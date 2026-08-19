"""Map legacy team_ids onto the 16 TI2026 entities via roster overlap.

Orgs rebrand (Tundra -> Iron Wing) and rosters move wholesale between orgs.
Rating a team on its new team_id alone throws away its entire history, so we
merge any historical team_id that fielded >= MIN_OVERLAP of the team's current
five players into that team's canonical id.

Writes data/aliases.json  {legacy_team_id: canonical_team_id}
"""

import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_OVERLAP = 4        # players shared with the current roster (out of 5)
MIN_GAMES = 30         # games that legacy id played with that roster


def main():
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json")))
    with open(os.path.join(ROOT, "data", "raw", "player_matches.json")) as f:
        pm = json.load(f)

    # account -> team_id -> games
    played = collections.defaultdict(collections.Counter)
    for r in pm:
        tid = r["radiant_team_id"] if r["player_slot"] < 128 else r["dire_team_id"]
        if tid:
            played[r["account_id"]][tid] += 1

    canon_ids = {info["team_id"] for info in teams.values()}

    # legacy id -> best claim, so two TI teams can't both inherit the same history
    claims = {}
    for name, info in teams.items():
        canon = info["team_id"]
        roster = [p["account_id"] for p in info["players"][:5]]
        overlap = collections.Counter()
        games = collections.Counter()
        for acc in roster:
            for tid, n in played[acc].items():
                overlap[tid] += 1
                games[tid] += n
        for tid, ov in overlap.items():
            if tid == canon or tid in canon_ids:
                continue
            if ov >= MIN_OVERLAP and games[tid] >= MIN_GAMES:
                bid = (ov, games[tid])
                if tid not in claims or bid > claims[tid][0]:
                    claims[tid] = (bid, canon, name)

    aliases = {tid: canon for tid, (_, canon, _) in claims.items()}

    by_team = collections.defaultdict(list)
    for tid, ((ov, g), canon, name) in claims.items():
        by_team[name].append((tid, ov, g))
    print(f"{len(aliases)} legacy ids merged\n")
    for name, info in teams.items():
        merged = sorted(by_team.get(name, []), key=lambda x: -x[2])
        tail = ", ".join(f"{t}({ov}p,{g}g)" for t, ov, g in merged) or "(none)"
        print(f"{name:18s} <- {tail}")

    with open(os.path.join(ROOT, "data", "aliases.json"), "w") as f:
        json.dump({str(k): v for k, v in aliases.items()}, f, indent=2)


if __name__ == "__main__":
    main()
