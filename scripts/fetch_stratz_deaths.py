"""Death events with map coordinates, from STRATZ.

Two coach-title conditions need information OpenDota does not carry:

    残酷之人 the Cruel   +13%  "if a player is killed while in their own fountain"
    受难之人 the Tormented +23% "if any player dies to a Tormentor"

The second is recoverable from OpenDota's `killed_by`, but the first needs the
*location* of each death, and only STRATZ exposes that
(MatchPlayerStatsDeathEventType.positionX / positionY).

Titles are the one layer that costs no roll tokens and can be changed freely at
any time, so measuring which of the eight offered suffixes actually fires is
pure upside.

STRATZ's batch `matches(ids:)` is admin-only, so this walks matches one at a
time. Writes data/raw/stratz_deaths.json.
"""

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "stratz_deaths.json")
def _token():
    """STRATZ token: $STRATZ_TOKEN, else the gitignored api/straz file."""
    tok = os.environ.get("STRATZ_TOKEN", "").strip()
    if tok:
        return tok
    path = os.path.join(ROOT, "api", "straz")
    if os.path.exists(path):
        return open(path).read().strip()
    sys.exit("no STRATZ token: set $STRATZ_TOKEN or write one to api/straz "
             "(get one at https://stratz.com/api)")


TOKEN = _token()

Q = """{ match(id: %d) {
  id durationSeconds didRadiantWin parsedDateTime
  players { isRadiant playerSlot heroId
    stats { deathEvents { time positionX positionY byAbility byItem } } } } }"""


def gql(query, tries=4):
    req = urllib.request.Request(
        "https://api.stratz.com/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + TOKEN,
                 "User-Agent": "STRATZ_API"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 429:                      # rate limited, back off hard
                time.sleep(30 * (k + 1))
            else:
                time.sleep(3 * (k + 1))
    return None


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rich = json.load(open(os.path.join(ROOT, "data", "raw",
                                       "pro_matches_rich.json")))
    rich = [r for r in rich if r.get("tier") in ("premium", "professional")]
    rich.sort(key=lambda r: -r["start_time"])
    ids = [r["match_id"] for r in rich[:want]]

    done = {}
    if os.path.exists(OUT):
        done = {int(k): v for k, v in json.load(open(OUT)).items()}
        print(f"已有 {len(done)} 场，续拉")

    for i, mid in enumerate(ids):
        if mid in done:
            continue
        r = gql(Q % mid)
        m = ((r or {}).get("data") or {}).get("match")
        done[mid] = m
        if (i + 1) % 25 == 0:
            json.dump({str(k): v for k, v in done.items()}, open(OUT, "w"))
            got = sum(1 for v in done.values() if v)
            print(f"  {i+1}/{len(ids)}  有效 {got}", flush=True)
        time.sleep(0.35)

    json.dump({str(k): v for k, v in done.items()}, open(OUT, "w"))
    got = sum(1 for v in done.values() if v)
    print(f"完成：{len(done)} 场请求，{got} 场有数据 -> {OUT}")


if __name__ == "__main__":
    main()
