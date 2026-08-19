# DotaPred — The International 2026 Predictions & Fantasy

Modelling for Valve's TI2026 compendium: the Swiss-stage Predictions slate and
the Fantasy roster. Goal is the 90th overall percentile, which is the lowest
tier that pays out a Terrain Token.

## What is here

```
docs/     the written record, in the order it was worked out
scripts/  data pulls, models, validation, optimisers
data/     derived results (raw API pulls are gitignored, see below)
```

## Results that survived validation

**Team strength.** Bradley–Terry ratings fit on gold-difference-per-minute
rather than win/loss. On a sealed holdout (2025-11 … 2026-06, monthly
non-overlapping windows, nothing tuned on it):

| model | log-loss | single game | Bo3 series |
|---|---:|---:|---:|
| win/loss baseline | 0.6644 | 60.3% | — |
| gold margin, team only | 0.6581 | 61.0% | — |
| **gold margin, team + player** | **0.6559** | **62.0%** | **65.3%** |

Paired improvement over the baseline is +0.0086 log-loss, t = +2.37 with
standard errors clustered by series.

**Ceiling.** A rating model fit on the very games it is scored on — an oracle —
reaches only 65% single-game accuracy. Half of top-tier pro games have a true
win probability near 0.5, so no pre-match team-strength model gets much past
that. 75% single-game accuracy is not available to anyone.

**Group-stage format is an identity.** 16 teams, 5 Swiss rounds, 4 wins
advance / 4 losses eliminate forces exactly
`1×(4-0) 2×(4-1) 5×(3-2) 5×(2-3) 2×(1-4) 1×(0-4)` — which is precisely the
1/2/5/5/2/1 capacity of the client's prediction buckets. `simulate.py` asserts
this every run.

**Sorting by strength is optimal for the prediction slate.** The capacity
constraints make the slate a permutation, and bucket probabilities are monotone
in strength, so the rearrangement inequality makes strength-order optimal for
the linear *and* the convex objective. Verified by exhausting all 240 swaps and
all 3-cycles, and by annealing. Convexity, rating source and uncertainty model
all fail to change the answer.

**The playoff bracket is small enough to solve exactly.** Fourteen series, each
node a binary choice, so both the space of legal bracket fillings and the space
of real outcomes are the same 2^14 = 16,384 element set. `E[score]` is a
16,384 × 16,384 exact sum — no Monte Carlo error. It scored 8/16 on the group
stage slate (2,520 points) before the bracket unlocked; the chosen filling is in
`docs/09-playoff.md`.

**Nothing changes across a 3-4 day LAN break.** Folklore says the gap between
group stage and playoffs resets everything. On 37 premium LANs (1,768 games),
in-event form's coefficient survives the break intact (interaction −0.082,
t = −0.66; per gap-day +0.001, t = +0.04) and still outweighs the long-run prior
after it. The decay *is* real across all 130 events including online leagues
(−0.170, t = −2.35) — but there a multi-day "gap" means the teams went and
played somewhere else. The premium CI is wide enough to contain the pooled
estimate, so this is "not detectable at LAN level", not "proven absent".

**Player form is real, survives the break, and is useless for prediction.**
Per-game individual production (10 per-minute stats, z-scored within a
net-worth-derived position, residualised against how the team's game went)
minus that player's own 365-day baseline gives a form variable that
autocorrelates at r = +0.43 within a group stage — and at r = +0.41 *across*
the break (decay −0.019, n = 1,306). That is a second, independent confirmation
of the no-break-effect result, from a completely different observable. But once
the team rating already contains the group stage, form adds +0.0003 log-loss
(t = +0.29 over 304 post-break series). It flags Team Yandex as the only team
at TI2026 whose five players are collectively below baseline — and at the fitted
coefficient, using it changes none of the 14 bracket picks.

**Head-to-head history adds nothing to a rating.** Over 385 post-break series,
the shrunk net head-to-head record has coefficient +0.039 (t = +0.12) on top of
the rating, and the within-event version is *negative* (−1.283, t = −1.13).
Betting the head-to-head alone hits 54.9% where the rating hits 66.9%; in the 84
series where the two disagree, the rating is right 64.3% of the time.

**The ambiguous scoring rule turned out not to matter.** The client says only
"pick the team you think is going to win each series", which leaves open whether
a pick still counts when the team arrives at that node by a different path.
Optimising for the loose reading and being wrong costs 1.7%; the 50/50 mixture
optimum *is* the loose optimum. Cross-evaluating both readings was cheaper than
resolving the ambiguity.

## Findings that were negative

Recorded because they cost real time and are worth not repeating:

- Phase-split ratings (laning / mid / late) made prediction **worse**.
- Weighting recent games more heavily made it worse at every dose — **for
  pre-tournament prediction**. Narrowed, not overturned: when the training set
  contains the event's own group stage, a 45-day half-life beats the production
  150-day one on that event's post-break playoff series by 0.015 series
  log-loss (t = +2.76, 534 series across 37 LANs). A first pass picked 30 days
  off a test set that mixed pre- and post-break games; the post-break-only test
  reversed it, because the team half-life cannot be tuned independently of the
  player term's. See `docs/09-playoff.md` §3.3. Production ratings still use
  150 days.
- **A better probability model is not automatically a better plan.** The
  45-vs-30-day half-life result is significant, and it changes 3 of the 14
  bracket picks — worth ±5 points out of ~3,400, because the improvement lands
  almost entirely on one 50/50 node. Model selection needs log-loss; plan
  selection needs a separate expected-score calculation. Using either one for
  both leads you wrong in opposite directions (`docs/09-playoff.md` §8.2).
- Weighting a team's history by roster continuity made it worse than throwing
  away the same volume of games *at random* — the selection is systematically
  biased against exactly the teams that changed.
- A hand-built 5-component dominance composite lost to plain gold-diff/minute.
- Filtering out low-level leagues also made it worse (t = -3.1), even though
  20% of the training set is one grind league no TI team ever enters. Those
  games are connective tissue in the rating graph, not noise. Two earlier
  versions of this test were themselves wrong: zeroing weights leaves the teams
  in the parameter vector and distorts the centring, and comparing filtered to
  unfiltered models scores them on different test games. Only the
  same-test-games comparison is meaningful.
- Joint team+player estimation lost to stacking two separate models.
- Fantasy per-game production barely differs between slots: the true
  between-slot spread is 3.4% (core), 3.2% (mid), 1.8% (support). Most slot
  rankings are statistical ties.

## Reproducing

Raw pulls are gitignored. To rebuild them (~30 min of OpenDota API calls):

```bash
python3 scripts/fetch_teams.py          # resolve the 16 orgs + rosters
python3 scripts/fetch_matches_rich.py   # 42k pro matches with margin detail
python3 scripts/fetch_player_matches.py # per-game stats for the 84 TI players
                                        # (--accounts ID --merge to top up one player)
python3 scripts/fetch_extra.py          # gold curves + 227k lineup rows
python3 scripts/build_aliases.py        # merge org renames via roster overlap
```

Once those exist, don't re-run them to catch up — `topup_matches.py` pulls only
what is missing and merges by id, which is seconds instead of half an hour:

```bash
python3 scripts/topup_matches.py --since 2026-08-01   # rich + lineups + players
python3 scripts/topup_matches.py --since 2022-01-01 --until 2025-01-01  # backfill
```

The history goes back to 2022-01 (124k matches, all five Internationals) —
needed for the schedule-break analysis, which wants many events with a real
group-to-playoff gap.

Roster moves that happen after those pulls (bans, stand-ins) go in
`data/roster_overrides.json`, which `fantasy_stats.apply_roster_overrides`
layers on top of the fetched `teams.json` — a refetch cannot silently drop them.

Then:

```bash
python3 scripts/validate.py --stage select    # pick hyperparameters
python3 scripts/final_holdout.py              # sealed holdout, run once
python3 scripts/fit_ratings_v3.py             # production ratings
python3 scripts/simulate_boot.py 100000       # group stage, bootstrap ratings
python3 scripts/optimize_groups.py            # the prediction slate
python3 scripts/slot_estimate.py              # fantasy slot values + errors
```

Once the group stage is over, the playoff bracket becomes its own pipeline:

```bash
python3 scripts/ti_results.py                 # results + the 14-node bracket
python3 scripts/weighting_backtest.py         # first pass at the time weighting
python3 scripts/break_effect.py --premium --min-games 40   # schedule-break effect
python3 scripts/postbreak_headtohead.py --premium          # settles the half-life
for hl in 45 60 150; do
  python3 scripts/fit_playoff_probs.py --hl $hl --player-hl 150 --boot 200 \
          --out data/playoff_probs_hl$hl.json
done
python3 scripts/bracket_model_average.py data/playoff_probs_hl{45,60,150}.json
python3 scripts/fetch_premium_player_stats.py  # 80k player-game rows, premium tier
python3 scripts/player_form.py                 # does player form survive the break?
python3 scripts/ti_player_form.py              # the 40 playoff players right now
python3 scripts/h2h_value.py                   # is head-to-head worth anything?
```

## Data sources

OpenDota (matches, player stats, lineups — no key needed), Valve's
`IDOTA2League/GetLeagueData` for the bracket, and the Dota 2 client
localisation files via `dotabuff/d2vpkr` for the scoring rules.

Two scripts need a **STRATZ API token** — `fetch_stratz_fights.py` and
`fetch_stratz_deaths.py`, for the kill/death map positions OpenDota does not
expose. Get one from <https://stratz.com/api>, then either

```bash
export STRATZ_TOKEN=...            # preferred
echo '...' > api/straz             # or drop it in this file
```

`api/` is gitignored; nothing else in the repo needs a key. Two scripts also
read the local Dota 2 install for client localisation strings — override the
guessed path with `DOTA_GAME_DIR=/path/to/dota 2 beta/game/dota`.

Hero names, emblem/title text and other client strings under `data/` are
Valve's, reproduced here only as the inputs the analysis needs; Dota 2 is a
trademark of Valve Corporation, and this project is unaffiliated with Valve
and with any team or player named in it. Roster and eligibility notes cite
their sources — where an official statement and press coverage disagree on
wording, the docs follow the official statement.

## Licence

MIT, see `LICENSE`. The Chinese write-ups under `docs/` are part of the same
grant — reuse them freely with attribution.
