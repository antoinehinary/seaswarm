# SeaSwarm evaluation summary

600 runs: 3 strategies x 4 fleets x 50 generated scenarios (seed N = scenario gen_N), 180 min limit. Found = first detection. Brackets: 95 % bootstrap interval over scenarios (2000 resamples, fixed seed).

## Outcomes

| Fleet | Strategy | Found <= 30 min | Found <= 60 min | Found <= 120 min | Median time to find | Recovered | POS_cum @ 60 min | Refusals / run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D+1U | lawnmower | 30% [18%, 42%] | 46% [32%, 60%] | 62% [48%, 76%] | 84.6 min | 29/50 | 0.54 | 0.0 |
| 2D+1U | greedy | 40% [28%, 54%] | 64% [50%, 76%] | 82% [70%, 92%] | 36.7 min | 37/50 | 0.75 | 0.1 |
| 2D+1U | auction | 36% [22%, 50%] | 68% [56%, 80%] | 82% [70%, 92%] | 34.4 min | 40/50 | 0.75 | 0.0 |
| 4D | lawnmower | 30% [18%, 42%] | 38% [24%, 52%] | 64% [50%, 78%] | 71.3 min | 0/50 | 0.51 | 1.0 |
| 4D | greedy | 46% [32%, 60%] | 60% [46%, 74%] | 72% [60%, 84%] | 36.0 min | 0/50 | 0.63 | 7.7 |
| 4D | auction | 40% [26%, 54%] | 52% [38%, 66%] | 66% [52%, 80%] | 41.0 min | 0/50 | 0.63 | 6.7 |
| 2U | lawnmower | 12% [4%, 22%] | 34% [22%, 48%] | 52% [38%, 66%] | 103.9 min | 28/50 | 0.54 | 1.0 |
| 2U | greedy | 20% [10%, 32%] | 70% [56%, 82%] | 86% [76%, 94%] | 40.0 min | 49/50 | 0.81 | 22.8 |
| 2U | auction | 22% [10%, 34%] | 74% [62%, 86%] | 90% [82%, 98%] | 36.8 min | 48/50 | 0.80 | 5.7 |
| 4D+2U | lawnmower | 36% [22%, 50%] | 54% [40%, 68%] | 74% [62%, 86%] | 47.4 min | 36/50 | 0.74 | 2.1 |
| 4D+2U | greedy | 48% [34%, 62%] | 76% [64%, 86%] | 88% [78%, 96%] | 31.2 min | 44/50 | 0.87 | 29.2 |
| 4D+2U | auction | 50% [36%, 64%] | 80% [68%, 90%] | 98% [94%, 100%] | 30.0 min | 44/50 | 0.87 | 20.5 |

## Paired comparison (same scenarios and seeds)

Difference in the fraction found within 60 min, A - B, with a bootstrap interval of the mean paired difference. Pairing removes the scenario-to-scenario variance.

| Fleet | A - B | Difference | A faster / B faster / tie (time to find) |
| --- | --- | --- | --- |
| 2D+1U | lawnmower - greedy | -18% [-34%, -2%] | 16 / 29 / 5 |
| 2D+1U | lawnmower - auction | -22% [-36%, -8%] | 19 / 25 / 6 |
| 2D+1U | greedy - auction | -4% [-18%, +10%] | 20 / 23 / 7 |
| 4D | lawnmower - greedy | -22% [-38%, -8%] | 12 / 31 / 7 |
| 4D | lawnmower - auction | -14% [-30%, +2%] | 16 / 25 / 9 |
| 4D | greedy - auction | +8% [-6%, +22%] | 23 / 18 / 9 |
| 2U | lawnmower - greedy | -36% [-56%, -16%] | 13 / 36 / 1 |
| 2U | lawnmower - auction | -40% [-56%, -24%] | 11 / 35 / 4 |
| 2U | greedy - auction | -4% [-18%, +8%] | 26 / 22 / 2 |
| 4D+2U | lawnmower - greedy | -22% [-36%, -8%] | 20 / 29 / 1 |
| 4D+2U | lawnmower - auction | -26% [-42%, -10%] | 12 / 37 / 1 |
| 4D+2U | greedy - auction | -4% [-16%, +8%] | 23 / 24 / 3 |

## Safety

Separation violations: 0. Geofence violations: 0. Stranded vehicles: 0. Refusals (all runs): 4842.

## Plots

![Fraction found vs time](plots/found_vs_time.png)

![POS_cum vs time](plots/pos_vs_time.png)

## Interpretation

Searching the Bayesian probability map finds the person sooner than a fixed lawnmower pattern:
on the same 50 scenarios, greedy finds 18 to 36 percentage points more persons within 60 minutes
in every fleet, the auction 22 to 40 points more in three fleets (its +14 points with four drones
are not significant), and the median time to find drops by a third to two thirds. Greedy and the
bundle auction are statistically tied (every paired interval includes zero), which tells me that
with a replan every two minutes, planning three tiles ahead adds little over taking the best tile
each time. Fleet composition matters as much as the algorithm: the USVs sweep the most area per
hour and are the only vehicles that can recover the person, the EO drones lose most of their
value at night, and bigger fleets need more coordination (up to 29 separation refusals per run),
all of it absorbed by the safety layer without a single violation in 600 runs.
