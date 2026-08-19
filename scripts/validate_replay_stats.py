"""把我们从 OpenDota 推出来的 18 项，和 Valve 录像解析出来的同一批逐项对账。

第三方数据不能因为「有莲花」就整份信了。石油杯那 157 场我们两边都有，同一个
(match_id, account_id) 可以逐行比。对得上的项说明我们的推导没问题、可以继续混用；
对不上的项要么是我们的代理量错了（比如狂石用 madstone_bundle 的使用次数代替拾取数），
要么是 OpenDota 自己的估计和 Valve 的服务端口径不同（团战参与最可疑）。

输出每一项的：覆盖行数、相关系数、平均绝对差、我们的均值 / Valve 的均值 / 比值。
比值显著偏离 1 的项，之前所有用到它的结论都要重新看。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fantasy_stats as FS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = FS.COLOR["red"] + FS.COLOR["blue"] + FS.COLOR["green"]


def main():
    rep = json.load(open(os.path.join(ROOT, "data", "raw", "replay_stats.json")))
    by_key = {(r["match_id"], r["account_id"]): r for r in rep["rows"]}
    ours = json.load(open(os.path.join(ROOT, "data", "raw",
                                       "player_matches.json")))

    pairs = {s: ([], []) for s in STATS}
    n_match, n_row = set(), 0
    for r in ours:
        k = (r["match_id"], r["account_id"])
        v = by_key.get(k)
        if v is None:
            continue
        n_match.add(r["match_id"])
        n_row += 1
        q = FS.game_stats(r)
        for s in STATS:
            a, b = q[s], v.get(s)
            if b is None or (isinstance(a, float) and np.isnan(a)):
                continue
            pairs[s][0].append(float(a))
            pairs[s][1].append(float(b))

    print(f"重叠 {len(n_match)} 场比赛 / {n_row} 个选手-局\n")
    print(f"{'统计项':<10}{'行数':>7}{'相关':>8}{'完全相同%':>10}"
          f"{'我们均值':>11}{'Valve均值':>11}{'比值':>8}   判定")
    verdict = {}
    for s in STATS:
        a, b = np.array(pairs[s][0]), np.array(pairs[s][1])
        if len(a) == 0:
            print(f"{FS_CN.get(s, s):<10}{0:>7}    我们这边没有这一项")
            verdict[s] = "缺"
            continue
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        same = float(np.mean(np.isclose(a, b, rtol=1e-3, atol=1e-6)))
        ratio = a.mean() / b.mean() if b.mean() else float("nan")
        if same > 0.99:
            v = "一致"
        elif r > 0.98 and abs(ratio - 1) < 0.02:
            v = "等价"
        elif r > 0.9:
            v = "偏差"
        else:
            v = "❌ 不一致"
        verdict[s] = v
        print(f"{FS_CN.get(s, s):<10}{len(a):>7}{r:>8.3f}{100*same:>9.0f}%"
              f"{a.mean():>11,.2f}{b.mean():>11,.2f}{ratio:>8.3f}   {v}")

    bad = [s for s, v in verdict.items() if v.startswith("❌") or v == "偏差"]
    print(f"\n需要复查的项：{', '.join(FS_CN.get(s, s) for s in bad) or '（无）'}")
    print("只有 Valve 侧有的：" + ", ".join(
        FS_CN.get(s, s) for s in STATS if verdict.get(s) == "缺"))


FS_CN = {"kills": "击杀", "deaths": "死亡", "cs": "正反补", "gpm": "GPM",
         "madstone": "狂石", "towers": "推塔", "wards": "假眼", "stacks": "堆野怪",
         "runes": "拾神符", "watchers": "观察者", "smokes": "诡计之雾",
         "lotus": "莲花", "roshan": "肉山", "teamfight": "团战", "stuns": "眩晕",
         "tormentor": "痛苦魔方", "firstblood": "第一滴血", "courier": "杀信使"}

if __name__ == "__main__":
    main()
