"""Final slot estimates: clean + synthetic combined, then shrunk, with errors.

Measured on the 21 slots that have >=150 games of the pair actually playing
together, the synthetic estimator built from their separate games runs
-81 points low (-2.1%) with a slot-to-slot spread of ~204. That spread, not the
number of synthetic draws, is what limits it: inverse-variance weighting makes
the whole synthetic pool worth about

    n_effective = var(per-game) / 204^2  ~=  8 real games

So separate-game evidence is real but small, and cannot rescue a slot whose
pair has only played 26 games together. Those slots instead get shrunk toward
their role's mean by empirical Bayes, which is the honest way to say "we don't
know" without inventing a number.

Anything reported here comes with a standard error, and slots that are not
separated by 2 SE are called indistinguishable rather than ranked.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slot_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calibrate(data):
    """Bias and structural spread of the synthetic estimator, from rich slots."""
    d, c = [], []
    for v in data.values():
        if v and v["n_clean"] >= 150 and len(v["synth"]):
            d.append(v["synth"].mean() - v["clean"].mean())
            c.append(v["clean"].std(ddof=1) / np.sqrt(v["n_clean"]))
    d = np.array(d)
    c = np.array(c)
    # remove the part of the spread that is just noise in the clean estimate
    sigma = float(np.sqrt(max(d.var(ddof=1) - (c**2).mean(), 1.0)))
    return float(d.mean()), sigma, len(d)


def main():
    data = slot_data.build()
    bias, sigma_s, n_ref = calibrate(data)
    print(f"合成估计校准（{n_ref} 个 n_clean>=150 的槽位）")
    print(f"  偏差 {bias:+.0f}，去除 clean 噪声后的结构性误差 σ = {sigma_s:.0f}\n")

    rows = {}
    for (team, role), v in data.items():
        if not v or v["n_clean"] < 5:
            rows[(team, role)] = None
            continue
        c = v["clean"]
        n = len(c)
        var_game = float(c.var(ddof=1))
        se_c = np.sqrt(var_game / n)
        est, se = c.mean(), se_c
        n_eff_synth = 0.0
        if len(v["synth"]):
            s_est = v["synth"].mean() - bias           # de-biased
            wc, ws = 1 / se_c**2, 1 / sigma_s**2
            est = (c.mean() * wc + s_est * ws) / (wc + ws)
            se = np.sqrt(1 / (wc + ws))
            n_eff_synth = var_game / sigma_s**2
        rows[(team, role)] = {"est": est, "se": se, "n": n,
                              "n_eff_synth": n_eff_synth, "stats": v["stats"]}

    # empirical-Bayes shrinkage within each role
    print(f"{'slot':<26}{'n_clean':>8}{'合成≈场':>8}{'估计':>9}{'±se':>7}"
          f"{'收缩后':>9}{'±se':>7}  统计项")
    final = {}
    for role in ("core", "mid", "support"):
        ks = [k for k in rows if k[1] == role and rows[k]]
        if len(ks) < 4:
            continue
        x = np.array([rows[k]["est"] for k in ks])
        s2 = np.array([rows[k]["se"] ** 2 for k in ks])
        mu = x.mean()
        tau2 = max(x.var(ddof=1) - s2.mean(), 1.0)
        print(f"--- {role}   角色均值 {mu:,.0f}，槽位间真实标准差 τ={np.sqrt(tau2):,.0f}")
        for k in sorted(ks, key=lambda k: -rows[k]["est"]):
            r = rows[k]
            w = tau2 / (tau2 + r["se"] ** 2)
            sh = mu + w * (r["est"] - mu)
            sh_se = np.sqrt(tau2 * r["se"] ** 2 / (tau2 + r["se"] ** 2))
            final[k] = (sh, sh_se)
            print(f"{k[0] + '/' + role:<26}{r['n']:>8}{r['n_eff_synth']:>8.0f}"
                  f"{r['est']:>9,.0f}{r['se']:>7,.0f}{sh:>9,.0f}{sh_se:>7,.0f}  "
                  f"{'+'.join(r['stats'])}")
        print()

    print("=== 能不能区分？（收缩后，差值 vs 2×合并标准误）")
    for role in ("core", "mid", "support"):
        ks = [k for k in final if k[1] == role]
        ks.sort(key=lambda k: -final[k][0])
        top = ks[0]
        print(f"\n{role}：最高 {top[0]} ({final[top][0]:,.0f})")
        n_tied = 0
        for k in ks[1:]:
            diff = final[top][0] - final[k][0]
            sd = np.sqrt(final[top][1] ** 2 + final[k][1] ** 2)
            if diff < 2 * sd:
                n_tied += 1
            else:
                break
        if n_tied:
            print(f"  与之统计上无法区分的还有 {n_tied} 个：" +
                  ", ".join(k[0] for k in ks[1:1 + n_tied]))
        clear = ks[1 + n_tied:]
        if clear:
            print(f"  能确定高于：{', '.join(k[0] for k in clear[:5])}"
                  + (" …" if len(clear) > 5 else ""))

    json.dump({f"{k[0]}|{k[1]}": {"est": v[0], "se": v[1]} for k, v in final.items()},
              open(os.path.join(ROOT, "data", "slot_estimates.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
