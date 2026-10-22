#!/usr/bin/env python3
"""Offline MoE expert-cache policy replayer (TODO #61 P1).

Scores expert-cache admission/placement policies against a REAL routing trace, so a policy change can
be answered from one GPU run instead of one benchmark per hypothesis.  The trace is produced by the
delivery's own cache:

    MOE_EXPERT_TRACE=/tmp/trace.txt ./llama-cli ... -n 300 --temp 0 --single-turn

and this tool replays it (stdlib only, no model, no GPU).  See `ENVIRONMENT.md`
(`MOE_EXPERT_TRACE`) for the file format, and `benchmarks/moe-cache-coverage/` for the study that
this tool was built for.

Policies scored (all on the same trace, same arena):

  shipped   the delivered policy: per-table uniform slots, F-LRU eviction with a decay period,
            `admit=touch` doorkeeper, misses served cold.  This is the REFERENCE: its hit rate must
            land on the run's logged `h=`, otherwise the model is wrong and no comparison means
            anything.
  static    an oracle static placement: each table pre-filled with its globally hottest experts and
            never evicted.  It is an upper bound for any placement that cannot re-rank at run time.
  quota     the shipped policy, but the per-table slot count comes from the trace's per-LAYER routed
            mass instead of being uniform (campaign P2: "per-layer capacity quota").  A floor keeps
            the whole decode/verify band resident (`--min-slots`, default 8 = `--spec-draft-n-max 7`
            + 1), because a table that cannot hold a full verify batch declines the cache entirely.
  window    the shipped policy plus pass-start pre-admission: at the start of each pass the whole
            distinct expert set that pass will route is admitted up front (bounded by slots), which
            is what a prefetch-on-miss scheme can hope to emulate.

Accounting caveats (read before quoting a number):

  * The trace dedupes the identical record a tensor split produces per device, so the table set is ONE
    device's; `--arena-mib` is therefore a PER-DEVICE budget.
  * An IDENTITY (fully resident) table is not traced on the decode band (its routing is never read
    back).  Such a table cannot miss, so a trace from a fully-resident run has no decode records at
    all -- trace a partially-resident config (e.g. `MOE_EXPERT_CACHE_MIB=4096`).
  * The trace records the routing, not the timing.  `h` is comparable to the logged `h=`; throughput is
    not.
"""

import argparse
import collections
import json
import sys


# ---------------------------------------------------------------------------------------------
# trace parsing
# ---------------------------------------------------------------------------------------------

class Table:
    __slots__ = ("layer", "role", "n_experts", "expert_bytes", "freq", "slots")

    def __init__(self, layer, role, n_experts, expert_bytes):
        self.layer = layer
        self.role = role
        self.n_experts = n_experts
        self.expert_bytes = expert_bytes
        self.freq = collections.Counter()   # expert -> touches in the whole trace
        self.slots = 0


class Rec:
    __slots__ = ("band", "layer", "role", "n_tok", "n_used", "ids")

    def __init__(self, band, layer, role, n_tok, n_used, ids):
        self.band = band
        self.layer = layer
        self.role = role
        self.n_tok = n_tok
        self.n_used = n_used
        self.ids = ids


def parse(path, mass_band="all"):
    tables = {}
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first.startswith("# moe-trace v1"):
            sys.exit("not a moe-trace v1 file: %s" % path)
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if parts[0] == "T":
                if len(parts) != 5:
                    continue
                layer, role, n_exp, eb = int(parts[1]), parts[2], int(parts[3]), int(parts[4])
                tables[(layer, role)] = Table(layer, role, n_exp, eb)
            elif parts[0] in ("P", "D"):
                if len(parts) < 6:
                    continue
                layer, role = int(parts[2]), parts[3]
                n_tok, n_used = int(parts[4]), int(parts[5])
                ids = [int(x) for x in parts[6:]]
                if len(ids) != n_tok * n_used:
                    continue   # a truncated final line
                recs.append(Rec(parts[0], layer, role, n_tok, n_used, ids))
                if mass_band != "all" and parts[0] != ("D" if mass_band == "decode" else "P"):
                    continue
                t = tables.get((layer, role))
                if t is not None:
                    for e in ids:
                        if 0 <= e < t.n_experts:
                            t.freq[e] += 1
    return tables, recs


def segment_passes(recs):
    """Split the ordered records into passes.

    A pass is one sweep of the model: the same (layer, role) table is never visited twice in it.  The
    trace is written in dispatch order, so a repeat of a table means the next sweep started.  This is
    how a decode token (or a verify batch, or a prefill ubatch) is recovered without a pass id in the
    file.
    """
    passes = []
    cur = []
    seen = set()
    for r in recs:
        key = (r.layer, r.role)
        if key in seen:
            passes.append(cur)
            cur = []
            seen = set()
        seen.add(key)
        cur.append(r)
    if cur:
        passes.append(cur)
    return passes


# ---------------------------------------------------------------------------------------------
# the cache model
# ---------------------------------------------------------------------------------------------

class TableState:
    __slots__ = ("slots", "slot_expert", "expert_slot", "count", "ghost", "last", "clock",
                 "last_decay", "hits", "misses", "cold", "fills", "evictions")

    def __init__(self, t, slots):
        self.slots = min(slots, t.n_experts)
        self.slot_expert = [-1] * self.slots
        self.expert_slot = {}
        self.count = collections.Counter()
        self.ghost = collections.Counter()
        self.last = collections.Counter()
        self.clock = 0
        self.last_decay = 0
        self.hits = 0
        self.misses = 0
        self.cold = 0
        self.fills = 0
        self.evictions = 0


def decay(st, period):
    if st.clock - st.last_decay < period:
        return
    for d in (st.count, st.ghost):
        for k in list(d.keys()):
            d[k] >>= 1
    st.last_decay = st.clock - (st.clock % period)


def access(st, expert, period, admit, touch, allow_cold=True, protect=None):
    """Mirror of `access_locked`: F-LRU eviction + the fill-vs-cold admission rule.

    `protect` is the record's own expert set: an expert admitted earlier in the same token/pass must
    not be chosen as this miss's victim (the real hook passes exactly that list), which is what keeps
    one token's experts from evicting each other.
    """
    st.clock += 1
    decay(st, period)
    if expert in st.expert_slot:
        st.hits += 1
        st.count[expert] += 1
        st.last[expert] = st.clock
        return True
    st.misses += 1
    st.ghost[expert] += 1
    if st.slots <= 0:
        if allow_cold:
            st.cold += 1
        return False
    slot = -1
    for s in range(st.slots):
        if st.slot_expert[s] < 0:
            slot = s
            break
    victim = None
    if slot < 0:
        for s in range(st.slots):
            e = st.slot_expert[s]
            if e < 0:
                continue
            if protect is not None and e in protect:
                continue
            if victim is None or st.count[e] < st.count[victim] or \
                    (st.count[e] == st.count[victim] and st.last[e] < st.last[victim]):
                victim = e
                slot = s
        if slot < 0:
            if allow_cold:
                st.cold += 1
            return False
        if allow_cold:
            if admit == "value" and st.ghost[expert] <= st.count[victim]:
                st.cold += 1
                return False
            if admit == "touch" and st.ghost[expert] < touch:
                st.cold += 1
                return False
    if victim is not None:
        del st.expert_slot[victim]
        st.slot_expert[slot] = -1
        st.evictions += 1
    st.slot_expert[slot] = expert
    st.expert_slot[expert] = slot
    st.fills += 1
    st.count[expert] += 1           # the real code counts the use that admitted it
    st.last[expert] = st.clock
    return True


def resident_bytes(st, t):
    return len(st.expert_slot) * t.expert_bytes


# ---------------------------------------------------------------------------------------------
# slot allocation
# ---------------------------------------------------------------------------------------------

def uniform_slots(tables, arena_bytes):
    one = sum(t.expert_bytes for t in tables.values())
    return max(0, int(arena_bytes / one)) if one else 0


def quota_slots(tables, arena_bytes, mass_by_layer, min_slots):
    """Per-layer slot counts sized by routed mass (campaign P2), with a verify-band floor.

    `mass_by_layer[l]` is the routed mass of layer `l` (summed over its tables).  Solve for the scale
    `alpha` such that `sum_l slots_l * bytes_l == arena`, with `slots_l = clamp(alpha*mass_l/bytes_l,
    floor_l, cap_l)`.  `floor_l` = min_slots and `cap_l` = the layer's smallest `n_experts`, so
    gate/up/down stay aligned (the C++ rearm path keys on a per-layer slot count) and a layer can
    always hold a full decode/verify batch.
    """
    layers = collections.defaultdict(list)
    for t in tables.values():
        layers[t.layer].append(t)
    bytes_l = {l: sum(t.expert_bytes for t in ts) for l, ts in layers.items()}
    cap_l = {l: min(t.n_experts for t in ts) for l, ts in layers.items()}
    floor_l = {l: min(min_slots, cap_l[l]) for l in layers}

    def alloc(alpha):
        return {l: int(min(cap_l[l], max(floor_l[l], alpha * mass_by_layer.get(l, 0) / bytes_l[l])))
                for l in layers}

    def total(a):
        return sum(a[l] * bytes_l[l] for l in a)

    lo, hi = 0.0, 1.0
    while total(alloc(hi)) <= arena_bytes and hi < 1e12:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(alloc(mid)) <= arena_bytes:
            lo = mid
        else:
            hi = mid
    a = alloc(lo)
    # Hand any rounding leftovers to the hottest layers, largest remaining mass first, while they fit.
    order = sorted(layers, key=lambda l: -mass_by_layer.get(l, 0))
    for l in order:
        while a[l] < cap_l[l] and total(a) + bytes_l[l] <= arena_bytes:
            a[l] += 1
    return a


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------

def score(tables, passes, slots_by_layer=None, arena_bytes=0, period=32, admit="touch", touch=2,
          policy="shipped"):
    states = {}
    for key, t in tables.items():
        slots = slots_by_layer.get(t.layer, 0) if slots_by_layer is not None else t.slots
        states[key] = TableState(t, slots)

    if policy == "static":
        # oracle: each table pre-filled with its globally hottest experts, never evicted
        for key, t in tables.items():
            st = states[key]
            top = [e for e, _ in t.freq.most_common(st.slots)]
            for s, e in enumerate(top):
                st.slot_expert[s] = e
                st.expert_slot[e] = s
                st.last[e] = 0

    tokens = 0
    for p in passes:
        tokens += p[0].n_tok if p else 0
        if policy == "window":
            # pass-start pre-admission of the whole distinct routed set, hottest-miss first
            want = collections.defaultdict(set)
            for r in p:
                want[(r.layer, r.role)].update(r.ids)
            for key, experts in want.items():
                st = states.get(key)
                t = tables.get(key)
                if st is None or t is None:
                    continue
                have = len(st.expert_slot)
                if have >= st.slots:
                    continue
                for e in experts:
                    if have >= st.slots:
                        break
                    if e in st.expert_slot or not (0 <= e < t.n_experts):
                        continue
                    slot = next((s for s in range(st.slots) if st.slot_expert[s] < 0), -1)
                    if slot < 0:
                        break
                    st.slot_expert[slot] = e
                    st.expert_slot[e] = slot
                    have += 1
        for r in p:
            key = (r.layer, r.role)
            st = states.get(key)
            t = tables.get(key)
            if st is None or t is None:
                continue
            protect = set(r.ids)
            for e in r.ids:
                if 0 <= e < t.n_experts:
                    if policy == "static":
                        st.clock += 1
                        if e in st.expert_slot:
                            st.hits += 1
                        else:
                            st.misses += 1
                    else:
                        access(st, e, period, admit, touch, protect=protect)

    hits = sum(s.hits for s in states.values())
    misses = sum(s.misses for s in states.values())
    cold = sum(s.cold for s in states.values())
    fills = sum(s.fills for s in states.values())
    evict = sum(s.evictions for s in states.values())
    reaches = hits + misses
    # Traffic: a fill and a cold read are both one expert's bytes over the host link.
    traffic = sum((s.fills + s.cold) * tables[k].expert_bytes for k, s in states.items())
    arena = sum(resident_bytes(s, tables[k]) for k, s in states.items())
    # Routed-mass coverage of the FINAL resident set, on this trace's routing.
    total_mass = sum(sum(t.freq.values()) for t in tables.values())
    covered = sum(t.freq[e] for k, t in tables.items() for e in states[k].expert_slot)
    return {
        "policy": policy,
        "hits": hits, "misses": misses, "reaches": reaches,
        "h": (hits / reaches) if reaches else 0.0,
        "cold": cold, "fills": fills, "evictions": evict,
        "tokens": tokens,
        "mib_per_token": (traffic / tokens / 1048576.0) if tokens else 0.0,
        "resident_mib": arena / 1048576.0,
        "covered_mass_pct": (100.0 * covered / total_mass) if total_mass else 0.0,
        "slots": {str(k): s.slots for k, s in states.items()},
    }


def static_bound(tables, arena_bytes):
    """Routed-mass coverage of a GLOBAL top-f resident set of the same total slot count."""
    flat = []
    for t in tables.values():
        flat.extend(t.freq.values())
    n_slots = sum(t.slots for t in tables.values())
    flat.sort(reverse=True)
    total = sum(flat)
    if total == 0:
        return 0.0
    return 100.0 * sum(flat[:max(0, min(n_slots, len(flat)))]) / total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--arena-mib", type=float, default=None,
                    help="per-device arena budget in MiB (default: derive the uniform slot count from "
                         "--slots, else 4096)")
    ap.add_argument("--slots", type=int, default=None, help="force the uniform per-table slot count")
    ap.add_argument("--devices", type=int, default=1,
                    help="tensor-split device count.  The trace records the FULL host expert size, while "
                         "the arena holds one device's slice, so the per-device expert bytes are "
                         "expert_bytes/devices (default 1 = unsplit)")
    ap.add_argument("--period", type=int, default=32, help="F-LRU decay period (MOE_EXPERT_CACHE_PERIOD)")
    ap.add_argument("--admit", choices=("always", "value", "touch"), default="touch",
                    help="admission rule (MOE_EXPERT_CACHE_ADMIT); 'always' = 0, 'value' = 1, 'touch' = 2")
    ap.add_argument("--touch", type=int, default=2, help="MOE_EXPERT_CACHE_TOUCH")
    ap.add_argument("--min-slots", type=int, default=8,
                    help="quota floor: the whole decode/verify band must fit (default 8 = n_max 7 + 1)")
    ap.add_argument("--mass-band", choices=("all", "decode", "prefill"), default="decode",
                    help="which band's routed mass drives the quota policy (default decode: that is the "
                         "band the arena serves; 'all' is dominated by the prefill, which routes ~n_tok "
                         "experts per layer in one pass, and a prefill pass can also lose one table to "
                         "the staged path, which would distort the ranking)")
    ap.add_argument("--policies", default="shipped,static,quota,window")
    ap.add_argument("--measured-h", type=float, default=None,
                    help="the run's logged `MoE expert cache: h=` -- printed beside the model's own "
                         "`shipped` h so the model's error is visible (it is a model, not the cache)")
    ap.add_argument("--json", metavar="PATH", help="write the scored table as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    tables, recs = parse(args.trace, args.mass_band)
    if not tables or not recs:
        sys.exit("empty trace (is it a fully-resident run? see the caveat in --help)")
    if args.devices > 1:
        for t in tables.values():
            t.expert_bytes = t.expert_bytes // args.devices

    # Drop any table that never routed: it cannot affect a hit rate, and the sizing in the log only
    # counted tables that registered (all of them), so keep it for the byte accounting either way.
    passes = segment_passes(recs)
    bands = collections.Counter(r.band for r in recs)
    tokens = sum(r.n_tok for p in passes for r in p if r is p[0])

    one = sum(t.expert_bytes for t in tables.values())
    if args.slots is not None:
        uni = args.slots
    elif args.arena_mib is not None:
        uni = uniform_slots(tables, args.arena_mib * 1048576.0)
    else:
        args.arena_mib = 4096.0
        uni = uniform_slots(tables, args.arena_mib * 1048576.0)
    arena_bytes = args.arena_mib * 1048576.0 if args.arena_mib is not None else uni * one
    for t in tables.values():
        t.slots = min(uni, t.n_experts)

    if not args.quiet:
        print("trace      : %s" % args.trace)
        print("records    : %d (%s), %d passes, ~%d tokens"
              % (len(recs), ", ".join("%s=%d" % kv for kv in sorted(bands.items())), len(passes), tokens))
        print("tables     : %d (one device's set), %d experts, %.1f MiB per slot-set (one slot per table)"
              % (len(tables), sum(t.n_experts for t in tables.values()), one / 1048576.0))
        print("arena      : %.0f MiB budget -> uniform %d slots/table (%.1f%% of the expert set)"
              % (arena_bytes / 1048576.0, uni,
                 100.0 * sum(min(uni, t.n_experts) for t in tables.values()) /
                 max(1, sum(t.n_experts for t in tables.values()))))
        print()

    results = {}
    for pol in [p.strip() for p in args.policies.split(",") if p.strip()]:
        if pol == "quota":
            mass_by_layer = collections.Counter()
            for t in tables.values():
                mass_by_layer[t.layer] += sum(t.freq.values())
            sq = quota_slots(tables, arena_bytes, mass_by_layer, args.min_slots)
            r = score(tables, passes, slots_by_layer=sq, arena_bytes=arena_bytes,
                      period=args.period, admit=args.admit, touch=args.touch, policy="quota")
            r["layer_slots"] = {str(k): v for k, v in sorted(sq.items())}
        else:
            r = score(tables, passes, slots_by_layer=None, arena_bytes=arena_bytes,
                      period=args.period, admit=args.admit, touch=args.touch, policy=pol)
        results[pol] = r

    bound = static_bound(tables, arena_bytes)
    if not args.quiet:
        hdr = "%-8s %8s %9s %7s %9s %11s %8s" % ("policy", "h", "misses", "cold", "MiB/tok",
                                                 "resident MiB", "covered%")
        print(hdr)
        print("-" * len(hdr))
        for pol in results:
            r = results[pol]
            print("%-8s %8.4f %9d %7d %9.2f %11.1f %8.2f"
                  % (pol, r["h"], r["misses"], r["cold"], r["mib_per_token"], r["resident_mib"],
                     r["covered_mass_pct"]))
        print()
        print("static bound (global top-f, same slot count): %.2f%% of the routed mass" % bound)
        sh = results.get("shipped")
        if sh and args.measured_h is not None:
            print("calibration: model 'shipped' h=%.4f vs the run's logged h=%.4f (error %+.4f)"
                  % (sh["h"], args.measured_h, sh["h"] - args.measured_h))
            print("             (the model has no persisted heat prior, no seed and no arena shortfall, so")
            print("              it is usually a little optimistic; judge POLICIES by their delta, not by")
            print("              the absolute h)")
        if sh and "quota" in results:
            q = results["quota"]
            print("quota vs shipped: h %+.4f, MiB/token %+.1f%% (%s)"
                  % (q["h"] - sh["h"],
                     100.0 * (q["mib_per_token"] / sh["mib_per_token"] - 1.0) if sh["mib_per_token"] else 0.0,
                     "quota WINNER" if q["h"] > sh["h"] else "shipped wins/ties"))
            print("layer slots (quota): %s" % ", ".join("%s:%d" % kv for kv in
                                                        sorted(results["quota"]["layer_slots"].items(),
                                                               key=lambda kv: int(kv[0]))))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"trace": args.trace, "arena_mib": arena_bytes / 1048576.0,
                       "uniform_slots": uni, "passes": len(passes), "tokens": tokens,
                       "static_bound_pct": bound, "results": results}, f, indent=2)
        if not args.quiet:
            print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
