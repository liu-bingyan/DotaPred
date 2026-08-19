"""梦幻打造的规则表本体，从客户端 `scripts/fantasy_crafting.vdata_c` 里取。

这一份文件是 V 社自己用来判分的定义，比帮助页面的文案权威。里面有：

* `m_vecPrefixes` / `m_vecSuffixes` —— **全部 8 + 8**，每条带 `m_nBonus`（定值，
  不是每次生成随机 roll）和 `m_vecStats`（真正被判定的统计名）。
  文案与统计名有两处**对不上**，以统计名为准：
    - `Suffix_LateFirstBlood`：文案「10 分钟后」，实际 `first_blood_after_6_minutes`
    - `Suffix_FirstBlood`：文案「号角前」，实际 `before_the_horn` **或** `before_1_minute`
* `m_vecQualities` —— 品质加成与 roll 权重：+10%/+30%/+60%/+100%/+150%，
  权重 10/20/10/5/2
* `m_vecTablets` —— 三个定位的徽标槽布局（含高等级才开的第 4、5 槽）
* `m_vecGems` —— 每种颜色徽标可选的统计项（蓝色确实含观察者与莲花）
* `m_vecLeagues` —— **每届 TI 的官方选手名单与定位**，TI2026 是 80 人

需要游戏本体（读 vpk）与 `pip install vpk keyvalues3`。
输出 data/fantasy_crafting.json 与 data/ti2026_official_roles.json。
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "fantasy_crafting.json")
OUT_ROLES = os.path.join(ROOT, "data", "ti2026_official_roles.json")

# $DOTA_GAME_DIR wins; otherwise guess the usual install locations.
CANDIDATES = [p for p in [
    os.environ.get("DOTA_GAME_DIR"),
    "/mnt/d/SteamLibrary/steamapps/common/dota 2 beta/game/dota",
    "/mnt/c/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota",
    os.path.expanduser("~/.steam/steam/steamapps/common/dota 2 beta/game/dota"),
] if p]


def find_dota():
    for c in CANDIDATES:
        if os.path.exists(os.path.join(c, "pak01_dir.vpk")):
            return c
    return None


def data_block(raw):
    """Source 2 资源文件 -> DATA 块（这里是二进制 KV3 v5）。"""
    import struct
    _fs, _hv, _ver, boff, nblocks = struct.unpack_from("<IHHII", raw, 0)
    pos = 8 + boff
    for _ in range(nblocks):
        name = raw[pos:pos + 4]
        off, size = struct.unpack_from("<II", raw, pos + 4)
        if name == b"DATA":
            start = pos + 4 + off
            return raw[start:start + size]
        pos += 12
    raise ValueError("no DATA block")


def plain(o):
    """keyvalues3 的包装类型 -> 纯 python。"""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if hasattr(o, "value"):
        return plain(o.value)
    return o


def mojibake(s):
    """KV3 里的名字是 utf-8 字节被当 latin-1 读进来的。"""
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def main():
    import vpk
    import keyvalues3 as kv3

    gamedir = sys.argv[1] if len(sys.argv) > 1 else find_dota()
    if not gamedir:
        sys.exit("找不到 Dota 2 安装目录，把 game/dota 路径作为参数传进来")
    pak = vpk.open(os.path.join(gamedir, "pak01_dir.vpk"))
    raw = pak.get_file("scripts/fantasy_crafting.vdata_c").read()

    tmp = os.path.join(ROOT, "data", ".fantasy_crafting_data.bin")
    open(tmp, "wb").write(data_block(raw))
    try:
        doc = plain(kv3.read(tmp))
    finally:
        os.remove(tmp)
    json.dump(doc, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)

    setup = doc["default_fantasy_setup"]
    cs = setup["m_vecCraftingSetups"][0]
    print(f"前缀 {len(cs['m_vecPrefixes'])} 个 / 后缀 {len(cs['m_vecSuffixes'])} 个"
          f"  —— 这就是全池，重 roll 换不出别的")
    for key, tag in (("m_vecPrefixes", "前缀"), ("m_vecSuffixes", "后缀")):
        print(f"\n{tag}")
        for t in cs[key]:
            stats = " 或 ".join(s["m_sStatName"] for s in t["m_vecStats"])
            typ = t["m_vecStats"][0]["m_eStatType"].replace("k_eFantasyStatType_", "")
            name = t["m_sLocName"].split("_")[-1]
            print(f"  {name:<16}+{t['m_nBonus']:>3}%  [{typ:<6}] {stats}")

    print("\n品质：" + "  ".join(
        f"+{q['m_nBonus']}%(权重{q['m_nRollWeight']})" for q in cs["m_vecQualities"]))

    ti = [x for x in setup["m_vecLeagues"]
          if x["m_eEvent"] == "EVENT_ID_INTERNATIONAL_2026"]
    if ti:
        roles = {}
        for grp in ti[0]["m_vecPlayers"]:
            role = grp["m_eRole"].replace("FANTASY_ROLE_", "").lower()
            for p in grp["m_vecPlayers"]:
                roles[str(p["m_unAccountID"])] = {
                    "role": role, "name": mojibake(p["m_strPlayerName"]),
                    "team_id": p["m_unTeamID"], "valid": p.get("m_bIsValid", True)}
        json.dump(roles, open(OUT_ROLES, "w"), ensure_ascii=False, indent=1)
        print(f"\nTI2026 官方名单 {len(roles)} 人 -> {OUT_ROLES}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
