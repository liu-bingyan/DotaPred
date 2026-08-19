"""英雄「形容词」标签 —— 称号前缀的判定依据，直接来自客户端。

八个前缀（红/蓝/绿/紫/黄棕/水火冰/亡灵恶魔圣灵/披风面具）全部卡在同一件事上：
V 社自定义的英雄分类，任何公开 API 都没有。**但它就明文写在游戏本体里** ——
`pak01_dir.vpk` 里的 `scripts/npc/npc_heroes.txt`，每个英雄一个 `Adjectives` 块：

    "Adjectives"
    {
        "Wings" "0"  "Horns" "0"  "Cape" "0"  "Mask" "0"
        "Undead" "0" "Demon" "0"  "Spirit" "0"
        "Aquatic" "0" "Fiery" "0" "Icy" "0"
        "Blue" "0"   "Red" "0"    "Green" "0"  "Purple" "0"
        "Yellow" "0" "Brown" "0"  ...
    }

`scripts/fantasy_crafting.vdata_c` 里的前缀定义引用的就是这些键，
所以这是**判定表本身**，不是近似。

用法：
    python scripts/hero_adjectives.py                 # 自动找 Steam 安装
    python scripts/hero_adjectives.py <dota2 game/dota 目录>
输出 data/hero_adjectives.json。
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "hero_adjectives.json")

# $DOTA_GAME_DIR wins; otherwise guess the usual install locations.
CANDIDATES = [p for p in [
    os.environ.get("DOTA_GAME_DIR"),
    "/mnt/d/SteamLibrary/steamapps/common/dota 2 beta/game/dota",
    "/mnt/c/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota",
    os.path.expanduser("~/.steam/steam/steamapps/common/dota 2 beta/game/dota"),
] if p]

# 前缀 → 命中所需的标签（任一为真即命中）。编号是本地化串里
# DOTAShowFantasy2026PlayerMiniTooltip( N ) 的 N。
PREFIXES = {
    "red":         (1,  ["Red"]),
    "blue":        (2,  ["Blue"]),
    "green":       (3,  ["Green"]),
    "purple":      (4,  ["Purple"]),
    "golden":      (5,  ["Yellow", "Brown"]),
    "horned":      (6,  ["Horns", "Wings"]),        # 全池里的「有角或翅膀」
    "elemental":   (7,  ["Aquatic", "Fiery", "Icy"]),
    "otherworldly":(8,  ["Undead", "Demon", "Spirit"]),
    "hairy":       (9,  ["Bearded", "Fuzzy"]),      # 全池里的「有胡子或毛茸茸」
    # 判定用的统计名是 hero_cape / hero_mask，对应 Adjectives 里的 Cape / Mask。
    # 圣堂刺客那条写的是 "Masked"（拼写不一致，全表仅此一例），按引擎读 "Mask"
    # 理解应当**不算**盖世英雄 —— 这是全表唯一一个需要人工确认的英雄。
    "heroic":      (10, ["Cape", "Mask"]),
}


def find_dota():
    for c in CANDIDATES:
        if os.path.exists(os.path.join(c, "pak01_dir.vpk")):
            return c
    return None


def read_npc_heroes(gamedir):
    import vpk
    pak = vpk.open(os.path.join(gamedir, "pak01_dir.vpk"))
    return pak.get_file("scripts/npc/npc_heroes.txt").read().decode("utf-8", "replace")


def block(text, start):
    """从 start 处的 '{' 起做括号配对，返回块内文本。"""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError("unbalanced")


def parse(text):
    starts = [(m.start(), m.group(1))
              for m in re.finditer(r'^\t?"(npc_dota_hero_\w+)"', text, re.M)]
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        blk = text[pos:end]
        m = re.search(r'"Adjectives"', blk)
        if not m:
            continue
        adj = dict(re.findall(r'"(\w+)"\s+"(-?\w+)"', block(blk, m.end())))
        out[name] = {k: int(v) for k, v in adj.items() if v.lstrip("-").isdigit()}
    return out


def main():
    gamedir = sys.argv[1] if len(sys.argv) > 1 else find_dota()
    if not gamedir:
        sys.exit("找不到 Dota 2 安装目录，把 game/dota 路径作为参数传进来")
    adj = parse(read_npc_heroes(gamedir))
    base = adj.pop("npc_dota_hero_base", {})
    adj.pop("npc_dota_hero_target_dummy", None)

    # 英雄 id / 名字：V 社自己的 datafeed（公开，无需 key）
    import urllib.request
    with urllib.request.urlopen(
            "https://www.dota2.com/datafeed/herolist?language=schinese", timeout=30) as r:
        feed = json.load(r)["result"]["data"]["heroes"]

    rows = {}
    for h in feed:
        a = dict(base)
        a.update(adj.get(h["name"], {}))
        tags = sorted(k for k, v in a.items() if v and k not in ("Legs", "Nose"))
        rows[str(h["id"])] = {
            "name": h["name"].replace("npc_dota_hero_", ""),
            "en": h["name_english_loc"],
            "zh": h["name_loc"],
            "tags": tags,
            "prefix": sorted(p for p, (_, ks) in PREFIXES.items()
                             if any(a.get(k) for k in ks)),
        }
    missing = [h["name"] for h in feed if h["name"] not in adj]
    json.dump({"heroes": rows, "prefixes": {p: ks for p, (_, ks) in PREFIXES.items()}},
              open(OUT, "w"), ensure_ascii=False, indent=1)

    from collections import Counter
    c = Counter(p for r in rows.values() for p in r["prefix"])
    print(f"{len(rows)} 个英雄，缺 Adjectives 的：{missing}")
    for p, (n, ks) in sorted(PREFIXES.items(), key=lambda x: -c[x[0]]):
        print(f"  {p:<13} ({'+'.join(ks):<22}) {c[p]:3d} 个  {c[p]/len(rows):5.1%}")
    print("→", OUT)


if __name__ == "__main__":
    main()
