"""Resolve the 16 TI2026 teams on OpenDota and pull rosters + match history.

Writes:
  data/raw/teams/<team_id>_info.json
  data/raw/teams/<team_id>_players.json
  data/raw/teams/<team_id>_matches.json
  data/teams.json           -- resolved roster/id table
"""

import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "teams")
API = "https://api.opendota.com/api"

# Names as shown in the Dota 2 client Fantasy team picker (2026-07-31).
# Candidate OpenDota ids resolved from /api/teams; ambiguous ones listed and
# disambiguated by last_match_time.
CANDIDATES = {
    "Team Liquid": [2163],
    "BoomBoys": [8255888],
    "Xtreme Gaming": [8261500],
    "Team Falcons": [9247354],
    "Aurora Gaming": [9467224, 9255706],
    "Team Yandex": [9823272],
    "Iron Wing": [10150413],
    "Vici Gaming": [726228],
    "Team Resilience": [5017210],
    "LGD Gaming": [10150538, 15],
    "OG": [2586976],
    "GamerLegion": [9964962],
    "Nigma Galaxy": [7554697, 10136357],
    "Huligani": [10149530],
    "Team Vision": [9572001],
    "Team Spirit": [7119388],
}


def get(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": "DotaPred/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def cached(path, fname):
    dest = os.path.join(RAW, fname)
    if os.path.exists(dest):
        with open(dest) as f:
            return json.load(f)
    data = get(path)
    with open(dest, "w") as f:
        json.dump(data, f)
    time.sleep(1.2)  # stay under OpenDota's 60/min free tier
    return data


def main():
    os.makedirs(RAW, exist_ok=True)
    resolved = {}

    for name, ids in CANDIDATES.items():
        best = None
        for tid in ids:
            info = cached(f"/teams/{tid}", f"{tid}_info.json")
            if best is None or (info.get("last_match_time") or 0) > (
                best.get("last_match_time") or 0
            ):
                best = info
        tid = best["team_id"]
        players = cached(f"/teams/{tid}/players", f"{tid}_players.json")
        matches = cached(f"/teams/{tid}/matches", f"{tid}_matches.json")

        current = [p for p in players if p.get("is_current_team_member")]
        if not current:
            # Freshly-created org ids (TI rebrands) have no "current member" flag
            # set yet; fall back to whoever actually played for them.
            current = list(players)
        current.sort(key=lambda p: -(p.get("games_played") or 0))
        resolved[name] = {
            "team_id": tid,
            "od_name": best.get("name"),
            "tag": best.get("tag"),
            "rating": best.get("rating"),
            "last_match_time": best.get("last_match_time"),
            "n_matches": len(matches),
            "players": [
                {
                    "account_id": p["account_id"],
                    "name": p.get("name"),
                    "games_played": p.get("games_played"),
                    "wins": p.get("wins"),
                }
                for p in current[:8]
            ],
        }
        print(f"{name:18s} id={tid:9d} matches={len(matches):4d} roster={len(current)}")

    with open(os.path.join(ROOT, "data", "teams.json"), "w") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
