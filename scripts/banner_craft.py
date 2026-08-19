"""Banner state, emblem multipliers, and what a crafting operation is worth.

Multipliers are additive and were verified against all nine emblems on the
client screenshot of 2026-08-03:

    emblem multiplier = 100% + quality% + net trait effect on that emblem

Quality:  Tier I 10 / II 30 / III 60 / IV 100 / V 150.
Adjacency is a chain -- ①-②, ②-③, and ① is NOT adjacent to ③.
Traits:
    Fractal        +60 to itself if all three qualities differ
    Benevolent     +20 to each adjacent emblem, nothing to itself
    Vampiric       +50 to itself, -10 to each adjacent emblem
    Unique         +30 to itself if it is the only Unique on the banner
    Friendly       +50 to itself if the banner holds >= 3 Friendly (i.e. all 3)
    Incorruptible  quality below Tier III counts as Tier III
    Base           nothing

A roll only ever touches the currently selected banner (confirmed in client,
2026-08-03), so each banner is an independent crafting problem.

The one thing still unmeasured is the roll distribution -- how likely each
quality tier and each trait is when rerolled. Those are parameters here, with
the sensitivity reported rather than a single number pretended to be known.
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402
from fantasy_model import role_game_scores  # noqa: E402
from banner_value import evaluate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUALITY = {1: 10, 2: 30, 3: 60, 4: 100, 5: 150}
# The client's help screen lists these five under "all available traits".
# dota_english.txt also carries Base and Incorruptible, but neither appears in
# the TI2026 pool -- same situation as the title pools, where the file holds
# entries from earlier events. Not one emblem has ever rolled either.
TRAITS = ["fractal", "benevolent", "vampiric", "unique", "friendly"]
TRAIT_CN = {"fractal": "分形", "benevolent": "仁爱", "vampiric": "吸血鬼",
            "unique": "唯一", "friendly": "友好"}
def adj(n):
    """Adjacency is a chain 1-2-...-n; the ends have one neighbour, the rest two.

    Period 1 banners held 3 emblems, period 2 holds 5 (client, 2026-08-13).
    All 15 period-2 emblems check out against this chain, so the rule is the
    same shape at both sizes -- only the length changed.
    """
    return {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}


ADJ = adj(3)   # kept for callers that still assume the period-1 layout


def multipliers(banner):
    """banner = [(stat, tier, trait), ...] in positional order -> [w0, w1, ...]."""
    tiers = [b[1] for b in banner]
    traits = [b[2] for b in banner]
    ADJ = adj(len(banner))
    # Fractal wants *every* quality on the banner to differ. At 5 emblems that
    # means an exact permutation of tiers I..V, which is why it reads 0 on the
    # 2026-08-13 mid banner (tiers 5,1,4,2,1 -- the two Tier I's kill it).
    # The competing reading "no other emblem shares my tier" would have paid
    # out there, so that one is falsified.
    distinct = len(set(tiers)) == len(tiers)
    n_unique = traits.count("unique")
    n_friendly = traits.count("friendly")
    out = []
    for i, (_, tier, trait) in enumerate(banner):
        q = QUALITY[tier]
        if trait == "incorruptible":
            q = max(q, QUALITY[3])
        t = 0
        if trait == "fractal" and distinct:
            t += 60
        elif trait == "vampiric":
            t += 50
        elif trait == "unique" and n_unique == 1:
            t += 30
        elif trait == "friendly" and n_friendly >= 3:
            t += 50
        for j in ADJ[i]:
            if traits[j] == "benevolent":
                t += 20
            elif traits[j] == "vampiric":
                t -= 10
        out.append(1 + (q + t) / 100.0)
    return out


def value(rows, bucket_col, banners, n_sims, seed=7):
    """E[period score] for a list of banner states, sharing one set of draws."""
    sets, weights = [], []
    for b in banners:
        w = multipliers(b)
        sets.append(tuple(x[0] for x in b))
        weights.append({x[0]: wi for x, wi in zip(b, w)})
    # evaluate() takes one weight dict, so run one banner at a time but with an
    # identical seed -- the draws line up, so the comparison stays paired
    return np.array([evaluate(rows, bucket_col, [s], weights=w,
                              n_sims=n_sims, seed=seed).mean()
                     for s, w in zip(sets, weights)])


def reachable(banner, k):
    """States reachable with k more single-tier quality increases."""
    out = {banner}
    for d in itertools.product(range(k + 1), repeat=len(banner)):
        if sum(d) <= k:
            out.add(tuple((s, min(5, t + dd), tr)
                          for (s, t, tr), dd in zip(banner, d)))
    return sorted(out)


def lookahead(rows, bucket_col, banner, k, n_sims, seed=7):
    """V_k: the best this banner can become given k more quality increases.

    Comparing two candidate moves by their immediate value treats the state
    after the move as terminal, which is wrong while tokens remain: a broken
    Fractal can be repaired, a low tier can be raised again. With 30+ tokens
    left the right yardstick is the reachable optimum, not the current value.
    Quality increases are monotone, so this correction is large for them and
    small for the non-monotone rerolls.
    """
    states = reachable(banner, k)
    v = value(rows, bucket_col, states, n_sims, seed)
    j = int(np.argmax(v))
    return float(v[j]), states[j]


def reroll_stat(banner, pos, pool):
    """Every outcome of rerolling one emblem's stat: guaranteed a new stat,
    and no stat may appear twice on the banner."""
    held = {b[0] for b in banner}
    out = []
    for s in pool:
        if s in held:
            continue
        nb = list(banner)
        nb[pos] = (s, banner[pos][1], banner[pos][2])
        out.append(tuple(nb))
    return out


def reroll_trait(banner, pos):
    """Rerolling a trait guarantees a different trait."""
    out = []
    for t in TRAITS:
        if t == banner[pos][2]:
            continue
        nb = list(banner)
        nb[pos] = (banner[pos][0], banner[pos][1], t)
        out.append(tuple(nb))
    return out


def reroll_quality(banner, positions, dist):
    """Reroll quality on the given positions. dist maps tier -> probability.
    Returns (banner, probability) pairs over the joint outcome."""
    out = []
    for combo in itertools.product(*[list(dist) for _ in positions]):
        p = 1.0
        nb = list(banner)
        for pos, tier in zip(positions, combo):
            p *= dist[tier]
            nb[pos] = (banner[pos][0], tier, banner[pos][2])
        out.append((tuple(nb), p))
    return out


# candidate quality distributions -- "larger boosts are more rare" is all the
# client says, so bracket it rather than guess one
QDIST = {
    "均匀": {1: .2, 2: .2, 3: .2, 4: .2, 5: .2},
    "温和递减": {1: .30, 2: .27, 3: .21, 4: .15, 5: .07},
    "陡峭递减": {1: .45, 2: .28, 3: .16, 4: .08, 5: .03},
}


def report(label, rows, bcol, cur, outcomes, n_sims, probs=None):
    banners = [cur] + [o for o in outcomes]
    v = value(rows, bcol, banners, n_sims)
    base, alt = v[0], v[1:]
    if probs is None:
        probs = np.full(len(alt), 1.0 / len(alt))
    probs = np.asarray(probs, dtype=float)
    ev = float((alt * probs).sum())
    print(f"\n--- {label}")
    print(f"    当前 {base:>11,.0f}")
    order = np.argsort(-alt)
    for k in order:
        b = outcomes[k]
        desc = "  ".join(f"{s}/{TRAIT_CN[t]}/{q}阶" for s, q, t in b)
        print(f"    {alt[k]:>11,.0f}  ({alt[k]-base:>+8,.0f})  p={probs[k]:.3f}  {desc}")
    print(f"    E[Δ] = {ev - base:+,.0f}  ({100*(ev-base)/base:+.2f}%)   "
          f"变好概率 {float(probs[alt > base].sum()):.0%}")
    return base, ev
