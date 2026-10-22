# PR #205 (martino2k6) — `wip/spec-draft-mtp-vocab`, as submitted

This directory is the **record of the contribution as it arrived**, merged so that the author keeps GitHub
authorship (merge commit of `4534e32e03`, PR #205, `martino2k6/llama-cpp-rdna-boosts:wip/spec-draft-mtp-vocab`).
It is **reference only — nothing here is applied**.

- `README.md` — the PR's own campaign README, verbatim (sha256 `7acf3d4040ecbb348978c1117e43fd8f91b56521e26158c3351eada18167595e`). Its six patches are the
  files listed there; **they are byte-identical** to `archive/work/spec-draft-mtp-vocab/0001-0006`, verified by
  sha256 at import time:
  - `0001` `77a20529aac462680cac741008760529f360b1f651f0a6177d1c588138f10fdc`
  - `0002` `a746a68e0367056a0cc11e2e54c8abf6e7466faa970b8fcc19ba4d3f9fc641d3`
  - `0003` `6f8e5f7b29afd33c2e7394993b1eeded2c4a1330ad43348af11e6940e63eb087`
  - `0004` `20dcc9035d9bc100277491fa000d7abb48af2eb24760c73a4de7565d714a5b9e`
  - `0005` `c4b02955a90996e9380e8921a753339a46e8da8f27d5edca7af6546162baa63e`
  - `0006` `be79cfcc7f0dedf9839b213c7072b447c49fd9f61862324ad3ea787d7573c1ac`
  They are not duplicated here for that reason; the copies under `archive/work/spec-draft-mtp-vocab/` are the
  unmodified originals, so this directory plus that one is the complete submission.

**Canonical adaptation:** [`archive/work/spec-draft-mtp-vocab/`](../../archive/work/spec-draft-mtp-vocab/README.md) — the same six
patches plus three campaign patches (a `ggml-backend-meta` row-range read fix that makes `-sm tensor` work, the
`gemma4-assistant` arm, and a merge-order-guard fix), gated on gfx1201.  Where the two READMEs disagree, the
campaign one is authoritative: it carries numbers from our hardware, the `-sm tensor` result, and the
measurements that show gemma4 is supported but not beneficial.

Shipped as **block 0012** — see `WORKLOG.md` for the release record that names this PR.
