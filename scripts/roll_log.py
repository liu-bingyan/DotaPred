"""Summarise the observed crafting options against a uniform-pool null.

The client always shows three operations and never says where they come from.
The enumerated space is 4 targeted verbs x 17 targets + 2 untargeted quality
operations = 70. Whether they are drawn uniformly is the open question, and it
is the only missing input to the token strategy, so every observed draw counts.

Reads data/roll_options_log.json.
"""

import collections
import itertools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETED = ["increase_quality", "roll_quality", "roll_trait", "roll_stat"]
UNTARGETED = ["increase_one_quality", "increase_two_decrease_one"]
TARGETS = ["choose_one", "all",
           "ruby", "sapphire", "emerald",
           "first_all", "first_ruby", "first_sapphire", "first_emerald",
           "last_all", "last_ruby", "last_sapphire", "last_emerald",
           "random_all", "random_ruby", "random_sapphire", "random_emerald"]
COLOUR_SCOPED = {t for t in TARGETS if t.endswith(("ruby", "sapphire", "emerald"))}
POOL = len(TARGETED) * len(TARGETS) + len(UNTARGETED)


def main():
    log = json.load(open(os.path.join(ROOT, "data", "roll_options_log.json")))
    all_draws = log["draws"]
    draws = [d for d in all_draws if not d.get("censored")]
    n_cens = len(all_draws) - len(draws)
    if n_cens:
        print(f"跳过 {n_cens} 条 censored 记录 —— 它们来自「拿不定主意才记」的规则，"
              f"含明显好选项的刷新被系统性删掉，用来计数会得出反向结论。")
    if not draws:
        print("没有无偏样本。需要：每次刷新出现后、决定之前一律记录。")
        return
    slots = [o for d in draws for o in d["options"]]
    n = len(slots)
    print(f"{len(draws)} 次观测到的刷新，共 {n} 个选项槽   （枚举空间 {POOL} 种）\n")

    # increase_quality 在池里（用户直接观察确认），但每次出现即被采用，
    # 从不进入 draws。所以它的计数恒为 0，且这个 0 会把其余动词的相对频率也带偏
    # —— 被拿来记录的刷新，是「不含提升品质」的条件样本。
    known = json.load(open(os.path.join(ROOT, "data",
                                        "roll_options_log.json"))).get(
        "known_in_pool", {})
    v = collections.Counter(o["verb"] for o in slots)
    print("动词分布   ⚠ 这是**以『本次刷新没有提升品质』为条件**的样本，")
    print("           不能当作池子的无条件频率；提升品质的真实频率无法从这里估计。")
    for k in TARGETED + UNTARGETED:
        exp = n * (len(TARGETS) / POOL if k in TARGETED else 1 / POOL)
        flag = ""
        if k in known and known[k].get("confirmed") and v[k] == 0:
            flag = "   ← 已确认在池里，计数为 0 是删失造成的"
        print(f"  {k:<28}{v[k]:>3}   均匀假设下期望 {exp:.1f}{flag}")

    t = collections.Counter(o["target"] for o in slots)
    print("\n目标分布（只列出现过的）")
    for k, c in t.most_common():
        print(f"  {k:<28}{c:>3}")
    unseen = [k for k in TARGETS if k not in t]
    print(f"  未出现过：{', '.join(unseen)}")
    ncol = sum(c for k, c in t.items() if k in COLOUR_SCOPED)
    p = len(COLOUR_SCOPED) / len(TARGETS)
    print(f"\n按颜色限定的目标 {ncol}/{n}   均匀假设下比例 {p:.2f}"
          f"   全部命中的概率 {p ** n:.3f}")

    # 同一个选项在不同次刷新里重复出现的次数
    sets = [{(o["verb"], o["target"]) for o in d["options"]} for d in draws]
    print("\n跨刷新的重复（均匀假设下任意两次的期望重叠 = 3x3/70 = 0.13 个）")
    for (i, a), (j, b) in itertools.combinations(list(enumerate(sets)), 2):
        ov = a & b
        if ov:
            print(f"  第{i+1}次 vs 第{j+1}次：重叠 {len(ov)} 个 -> "
                  + ", ".join(f"{x}/{y}" for x, y in sorted(ov)))
    print("\n样本还太小，不足以下结论；继续记录。")


if __name__ == "__main__":
    main()
