"""Two datasets the current model doesn't have.

A) Phase curves. `radiant_gold_adv` is a per-minute array; so far we only used
   its last/max/min value. Sampling it at minutes 10/15/20/25/30 lets us rate a
   team's laning, mid-game and late-game separately, which is the only way a
   rating system can express "team A is strong early but folds late" -- a
   matchup effect no single scalar can represent.

B) Lineups. account_id for all 10 players of every top-tier game, so team
   strength can be modelled as a function of the five players rather than of an
   org name. Rosters move; org names don't track skill.

Writes data/raw/phase_curves.json and data/raw/lineups.json
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)


def query(sql):
    url = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaPred/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["rows"]
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(6 * (attempt + 1))
    raise RuntimeError("query failed")


def months(start, end):
    cur = start
    while cur < end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=6)).replace(day=1)
        yield int(cur.timestamp()), int(min(nxt, end).timestamp()), cur
        cur = nxt


PHASE_SQL = """
select m.match_id,
  m.radiant_gold_adv[11] as g10, m.radiant_gold_adv[16] as g15,
  m.radiant_gold_adv[21] as g20, m.radiant_gold_adv[26] as g25,
  m.radiant_gold_adv[31] as g30,
  m.radiant_xp_adv[11]   as x10, m.radiant_xp_adv[21] as x20,
  array_length(m.radiant_gold_adv,1) as npts
from matches m left join leagues l on l.leagueid = m.leagueid
where m.start_time >= {lo} and m.start_time < {hi}
  and m.radiant_team_id is not null and m.dire_team_id is not null
  and l.tier in ('premium','professional')
"""

LINEUP_SQL = """
select pm.match_id, pm.account_id, pm.player_slot
from player_matches pm join matches m using(match_id)
left join leagues l on l.leagueid = m.leagueid
where m.start_time >= {lo} and m.start_time < {hi}
  and m.radiant_team_id is not null and m.dire_team_id is not null
  and l.tier in ('premium','professional')
"""


def pull(name, sql_tmpl, out_path):
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for lo, hi, label in months(START, now):
        got = query(sql_tmpl.format(lo=lo, hi=hi))
        rows.extend(got)
        print(f"[{name}] {label:%Y-%m}  {len(got):6d}  (total {len(rows)})", flush=True)
        time.sleep(1.5)
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"[{name}] wrote {len(rows)} -> {out_path}\n", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "phase"):
        pull("phase", PHASE_SQL, os.path.join(ROOT, "data", "raw", "phase_curves.json"))
    if which in ("both", "lineup"):
        pull("lineup", LINEUP_SQL, os.path.join(ROOT, "data", "raw", "lineups.json"))
