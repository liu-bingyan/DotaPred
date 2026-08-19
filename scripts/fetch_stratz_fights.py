"""Kill / death / assist event streams, for rebuilding teamfights ourselves.

Teamfight Participation is the highest-stakes uncalibrated input in the whole
banner model: it wins the green emblem in all 48 team-role slots, and the model
flips to Roshan or Tormentor if its true value is only 80% of what we measure.
We use OpenDota's `teamfight_participation`, which comes from OpenDota's own
clustering of deaths into fights and its own participation test. Valve computes
it server-side by some other rule.

We cannot learn Valve's rule. What we can learn is how much the number moves
when the *detection* thresholds move -- if a wide range of sensible definitions
all land within a few percent, the input is safe; if they scatter, the green
emblem is genuinely undecided.

STRATZ carries every kill, death and assist with a timestamp and map position,
which is enough to rebuild fights under any threshold. Writes
data/raw/stratz_fights.json.
"""

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "stratz_fights.json")
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
  id durationSeconds
  players { steamAccountId isRadiant playerSlot heroId
    stats {
      killEvents   { time positionX positionY }
      deathEvents  { time positionX positionY }
      assistEvents { time positionX positionY }
    } } } }"""


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
            time.sleep((30 if getattr(e, "code", None) == 429 else 3) * (k + 1))
    return None


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 300
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
        done[mid] = ((r or {}).get("data") or {}).get("match")
        if (i + 1) % 25 == 0:
            json.dump({str(k): v for k, v in done.items()}, open(OUT, "w"))
            print(f"  {i+1}/{len(ids)}  有效 {sum(1 for v in done.values() if v)}",
                  flush=True)
        time.sleep(0.35)

    json.dump({str(k): v for k, v in done.items()}, open(OUT, "w"))
    print(f"完成：{sum(1 for v in done.values() if v)}/{len(done)} 场 -> {OUT}")


if __name__ == "__main__":
    main()
