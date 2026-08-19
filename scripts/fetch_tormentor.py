"""Tormentor participation, which is not the same thing as the killing blow.

The client's per-emblem tooltip is more precise than the help screen:

    DOTA_PlayerCardBonusExplainer16:
      "{f:fantasy_stat} points when the player PARTICIPATES IN a Tormentor kill."
    DOTA_PlayerCardBonusExplainer5:
      "{f:fantasy_stat} points when the player gets the KILLING BLOW on Roshan."

So Tormentor is the one green stat credited to everyone who helped, while
Roshan, Courier and Tower stay with whoever landed the last hit. `killed`
counts last hits only and therefore undercounts Tormentor badly.

Participation is recoverable: OpenDota's parsed `damage` blob has an
`npc_dota_miniboss` entry, so any player who dealt damage to a Tormentor
took part. This pulls that column for the TI2026 players.

Writes data/raw/tormentor_damage.json
"""

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "tormentor_damage.json")
# same window as fantasy_stats.load(); the full-year query times out
START = dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc)


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.load(r)
            if out.get("err"):
                raise RuntimeError(out["err"])
            return out["rows"]
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("query failed")


def main():
    accounts = open(os.path.join(ROOT, "data", "accounts.txt")).read().strip()
    lo = int(START.timestamp())
    sql = f"""
    select pm.account_id, pm.match_id, pm.player_slot,
           (pm.damage->>'npc_dota_miniboss')::int  as mb_damage,
           (pm.killed->>'npc_dota_miniboss')::int   as mb_kills,
           m.start_time, m.radiant_win, l.tier
    from player_matches pm
    join matches m using(match_id)
    left join leagues l on l.leagueid = m.leagueid
    where pm.account_id in ({accounts})
      and m.start_time >= {lo}
      and m.radiant_team_id is not null and m.dire_team_id is not null
    """
    rows = query(sql)
    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
