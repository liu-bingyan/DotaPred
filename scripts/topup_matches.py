"""增量补拉最近的比赛，合并进已有的原始文件。

全量重拉 `fetch_matches_rich.py` / `fetch_player_matches.py` 要半小时，而
TI 期间我们只缺最后几天。这个脚本按 match_id 去重合并，重复运行是安全的。

  python3 scripts/topup_matches.py            # 两个文件都补，从各自最新时间往前 3 天
  python3 scripts/topup_matches.py --since 2026-08-01
  python3 scripts/topup_matches.py --only rich
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RICH = os.path.join(ROOT, "data", "raw", "pro_matches_rich.json")
PM = os.path.join(ROOT, "data", "raw", "player_matches.json")
LIN = os.path.join(ROOT, "data", "raw", "lineups.json")

import fetch_extra as FE  # noqa: E402
import fetch_matches_rich as FR  # noqa: E402
import fetch_player_matches as FP  # noqa: E402


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["rows"]
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("query failed")


def query_span(make_sql, lo, hi):
    """跨度超过 45 天就按月分片 —— 一次查三年的 explorer 会超时。"""
    if hi - lo <= 45 * 86400:
        return make_sql(lo, hi)
    rows = []
    cur = dt.datetime.fromtimestamp(lo, dt.timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0)
    end = dt.datetime.fromtimestamp(hi, dt.timezone.utc)
    while cur < end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=6)).replace(day=1)
        a, b = max(int(cur.timestamp()), lo), min(int(nxt.timestamp()), hi)
        if a < b:
            got = make_sql(a, b)
            rows.extend(got)
            print(f"    {cur:%Y-%m} {len(got):6d}  (累计 {len(rows)})", flush=True)
            time.sleep(1.5)
        cur = nxt
    return rows


def merge(path, old, new, key):
    seen = {key(r) for r in new}
    kept = [r for r in old if key(r) not in seen]
    rows = kept + new
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"  {os.path.basename(path)}: {len(old)} -> {len(rows)} "
          f"(新 {len(new)}，覆盖 {len(old) - len(kept)})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="YYYY-MM-DD，默认 = 文件最新时间 -3 天")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD，默认 = 现在。回补历史时用")
    ap.add_argument("--only", choices=["rich", "players", "lineups"], default=None)
    args = ap.parse_args()

    now = (int(dt.datetime.fromisoformat(args.until)
                 .replace(tzinfo=dt.timezone.utc).timestamp())
           if args.until else int(dt.datetime.now(dt.timezone.utc).timestamp()))

    def lo_for(rows):
        if args.since:
            return int(dt.datetime.fromisoformat(args.since)
                       .replace(tzinfo=dt.timezone.utc).timestamp())
        return max(r["start_time"] for r in rows) - 3 * 86400

    if args.only in (None, "rich"):
        old = json.load(open(RICH))
        lo = lo_for(old)
        print(f"rich: 从 {dt.datetime.fromtimestamp(lo, dt.timezone.utc):%Y-%m-%d %H:%M} 起补")
        new = query_span(lambda a, b: query(f"""
        select {FR.COLS}
        from matches m left join leagues l on l.leagueid = m.leagueid
        where m.start_time >= {a} and m.start_time < {b}
          and m.radiant_team_id is not null and m.dire_team_id is not null
          and m.radiant_win is not null
        """), lo, now)
        merge(RICH, old, new, lambda r: r["match_id"])

    if args.only in (None, "lineups"):
        # lineups.json 里没有时间戳，用 rich 文件定起点，按 match 的开赛时间查
        old = json.load(open(LIN))
        lo = lo_for(json.load(open(RICH)))
        print(f"lineups: 从 {dt.datetime.fromtimestamp(lo, dt.timezone.utc):%Y-%m-%d %H:%M} 起补")
        new = query_span(lambda a, b: query(FE.LINEUP_SQL.format(lo=a, hi=b)),
                         lo, now)
        merge(LIN, old, new, lambda r: (r["match_id"], r["account_id"]))

    if args.only in (None, "players"):
        old = json.load(open(PM))
        lo = lo_for(old)
        accounts = open(os.path.join(ROOT, "data", "accounts.txt")).read().strip()
        print(f"players: 从 {dt.datetime.fromtimestamp(lo, dt.timezone.utc):%Y-%m-%d %H:%M} 起补")
        new = query_span(lambda a, b: query(f"""
        select {FP.COLS}
        from player_matches pm
        join matches m using(match_id)
        left join leagues l on l.leagueid = m.leagueid
        where pm.account_id in ({accounts})
          and m.start_time >= {a} and m.start_time < {b}
          and m.radiant_team_id is not null and m.dire_team_id is not null
        """), lo, now)
        merge(PM, old, new, lambda r: (r["account_id"], r["match_id"]))


if __name__ == "__main__":
    main()
