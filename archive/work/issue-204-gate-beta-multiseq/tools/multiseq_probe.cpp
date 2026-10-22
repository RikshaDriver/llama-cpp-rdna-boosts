// Multi-sequence decode probe for the GDN gate/beta fusion (issue #204).
//
// Decodes N distinct sequences in LOCKSTEP: one token per sequence in a single
// batch, so the graph carries a genuine multi-sequence ubatch (n_seq_tokens=1,
// n_seqs=N).  Nothing else in the process varies between arms, so two runs of
// the same arm are bit-identical and two arms can be compared exactly.
//
// Prints, per step and per sequence: the greedy token, the argmax and an FNV-1a
// hash of the full logit row.  Compare the output of
//     ./multiseq_probe model.gguf 2 32
//     GGML_CUDA_FUSE_GATE_BETA_BF16=0 ./multiseq_probe model.gguf 2 32
//
// build: see build_multiseq_probe.sh
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void quiet_log(ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf n_seqs n_steps\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int    n_seqs     = atoi(argv[2]);
    const int    n_steps    = atoi(argv[3]);
    if (n_seqs < 1 || n_seqs > 8) { fprintf(stderr, "n_seqs 1..8\n"); return 1; }

    const char * texts[8] = {
        "The capital of France is Paris, and the capital of Germany is",
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot =",
        "The three laws of thermodynamics state that energy is conserved, entropy",
        "Once upon a time in a small village by the sea, there lived a",
        "Translate the following sentence into French: the cat sat on the mat,",
        "A list of the prime numbers below fifty is 2, 3, 5, 7, 11, 13,",
        "The mitochondrion is the powerhouse of the cell because it produces",
        "In the winter of 1812, the Grande Armee marched east toward Moscow and",
    };

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    const char * split = getenv("SPLIT");
    if (split && !strcmp(split, "tensor"))   { mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR; }
    if (split && !strcmp(split, "none"))     { mparams.split_mode = LLAMA_SPLIT_MODE_NONE; }
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 512 * n_seqs;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 32;
    cparams.n_seq_max = n_seqs;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // prefill one sequence at a time (single-sequence ubatches: the multi-sequence shape is what we probe)
    std::vector<int32_t> next(n_seqs, 0);
    for (int s = 0; s < n_seqs; ++s) {
        std::vector<llama_token> toks(512);
        const int nt = llama_tokenize(vocab, texts[s], (int) strlen(texts[s]), toks.data(), (int) toks.size(), true, true);
        if (nt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.resize(nt);

        llama_batch batch = llama_batch_init(nt, 0, 1);
        for (int i = 0; i < nt; ++i) {

            batch.token   [batch.n_tokens] = toks[i];
            batch.pos     [batch.n_tokens] = i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = s;
            batch.logits  [batch.n_tokens] = (i == nt - 1);
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
        llama_batch_free(batch);
        // greedy next token for this sequence from the last prefill row
        const float * logits = llama_get_logits_ith(ctx, nt - 1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) { if (logits[v] > logits[best]) best = v; }
        next[s] = best;
    }

    // lockstep decode: one token per sequence, one batch, n_steps times
    const int n_pos = 512;  // position base is irrelevant for the probe; keep the sequences aligned
    for (int step = 0; step < n_steps; ++step) {
        llama_batch batch = llama_batch_init(n_seqs, 0, 1);
        for (int s = 0; s < n_seqs; ++s) {
            batch.token   [batch.n_tokens] = next[s];
            batch.pos     [batch.n_tokens] = n_pos + step;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = s;
            batch.logits  [batch.n_tokens] = true;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at step %d\n", step); return 1; }
        printf("step %2d", step);
        for (int s = 0; s < n_seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx, s);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) { if (logits[v] > logits[best]) best = v; }
            printf(" | s%d tok %6d argmax %6d l0 %12.6f hash %016llx", s, next[s], best, logits[0],
                   (unsigned long long) fnv1a(logits, (size_t) n_vocab * sizeof(float)));
            next[s] = best;
        }
        printf("\n");
        llama_batch_free(batch);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
