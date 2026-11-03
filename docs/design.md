# SeaSwarm design notes

Equations, assumptions and parameter sources. All parameters are illustrative values taken from
public literature; SeaSwarm is not an operational search-and-rescue tool.

## 1. Frames, units and conventions

- Internal units: metres, seconds, radians, m/s. Knots and compass bearings appear only in
  user-facing text and in scenario files (`seaswarm.units` converts them once).
- Frame: local ENU, origin at the **centre of the search area**, x east, y north. The base is a
  point in this frame and may lie outside the area.
- Yaw: radians, counter-clockwise from +x. Bearing: degrees, clockwise from north.
  `yaw = 90° - bearing`.
- Vector fields are stored as (u, v) "towards" components. Wind is given by the bearing it blows
  **from** (meteorological convention): wind from 270° moves air east, u > 0. Current is given by
  the bearing it flows **towards** (oceanographic convention).
- Wind is the 10 m wind W10, the reference height of the leeway coefficients.

## 2. Drift of a person in water (truth model)

### 2.1 Equations

Each particle (the true person, or one sample of a Monte Carlo cloud) moves with

```
v = U_current(x) + n_current(t) + L_d(t) e_dw + s(t) L_c(t) e_cw
L_d = a_d |W10| + b_d + eps_d(t)
L_c = a_c |W10| + b_c + eps_c(t)
```

- `e_dw`: unit vector downwind; `e_cw`: unit vector 90° to the right of downwind.
- `s(t) ∈ {+1, -1}`: crosswind side. Objects drift on one side of the downwind axis and
  occasionally "jibe" to the other; this produces the characteristic two-lobed leeway spread.
- `eps_d`, `eps_c`, the optional current noise `n_current` and the optional gust on `|W10|` are
  independent Ornstein-Uhlenbeck processes, drawn independently for every particle.
- Integration: explicit Euler at `dt = 1 s`.

### 2.2 Parameters (PIW-1)

From OpenDrift `opendrift/models/OBJECTPROP.DAT`, object PIW-1, "Person-in-water (PIW), unknown
state (mean values)", which implements the US Coast Guard leeway tables (Allen and Plourde 1999).

| Parameter | Value | Note |
| --- | --- | --- |
| Downwind slope `a_d` | 0.0096 | 0.96 % of W10 |
| Downwind offset `b_d` | 0 m/s | |
| Downwind noise std | 0.12 m/s | standard error of the regression |
| Crosswind slope `a_c` | 0.0054 | ±0.54 % of W10, symmetric |
| Crosswind offset `b_c` | 0 m/s | |
| Crosswind noise std | 0.094 m/s | |
| Jibe probability | 4 % per hour | OpenDrift default, consistent with Allen (2005) |
| Noise correlation time `tau` | 1800 s | scenario parameter |

A person therefore drifts at about 1 % of the wind speed relative to the water, plus the current.
`LeewayModel.simple(k)` keeps the textbook special case "current + k × wind" for teaching and tests.

### 2.3 Why Ornstein-Uhlenbeck noise

Leeway errors come from slowly changing causes (sea state, posture, clothing) and persist for tens
of minutes. White noise drawn every step averages out: after time T it spreads a particle by only
`sigma * sqrt(dt * T)`, about 12 m after 3 h at `dt = 1 s` for `sigma = 0.12 m/s`. An OU process
with correlation time `tau` has the exact displacement variance

```
Var[X(T)] = 2 sigma^2 tau (T - tau (1 - exp(-T / tau)))   ->   2 sigma^2 tau T   for T >> tau
```

so it behaves like a diffusion with `D = sigma^2 tau` at long times. This is the diffusion
coefficient the belief predict step uses (Phase 2).

### 2.4 Exact discretisation

Every OU process is advanced with

```
x <- x exp(-dt / tau) + sigma sqrt(1 - exp(-2 dt / tau)) N(0, 1)
```

which keeps the stationary standard deviation exactly `sigma` for any `dt` (Euler-Maruyama is
only correct for `dt << tau`). Processes start from their stationary distribution `N(0, sigma^2)`.
Jibing is a Poisson process: per step, `p_flip = 1 - (1 - 0.04) ** (dt / 3600)`.

### 2.5 Current and wind fields

| Field | Definition | Divergence |
| --- | --- | --- |
| Uniform current | constant (u, v) | 0 |
| Linear-gradient current | constant direction, speed `s0 + shear * (x · n_left)` | 0: gradient ⟂ flow |
| Gyre | stream function `psi = A exp(-r^2 / (2 R^2))`, `u = -dpsi/dy`, `v = dpsi/dx` | 0 by construction |
| Wind | uniform W10, from-bearing rotating at a constant rate; optional OU gusts on the speed, clipped at 0 | — |

The gyre speed is largest at `r = R`, where it equals `max_speed_mps` (`A = v_max R e^(1/2)`).

### 2.6 Randomness

One `numpy.random.SeedSequence(seed)` per scenario is spawned into named, independent generators:
`drift`, `sensors`, `initial_position`, `planner_tiebreak`, `particles` (in that order; new
streams are only ever appended). The true person uses `initial_position` (fall position around the
LKP) and `drift`; Monte Carlo clouds use `particles`. Adding draws in one stream never changes
another, so strategies compared on the same seed face the same person trajectory.

### 2.7 Validation (Phase 1 tests)

| Check | Result |
| --- | --- |
| Uniform current only, 3600 s | exact to 1e-6 m |
| Mean downwind leeway, 2000 particles, W10 = 10 m/s, 3 h | within 0.5-1.8 % of 0.096 m/s over 5 seeds (limit 5 %); no bias at 40 000 particles (−0.43 % and −0.05 %, ±0.33 % at 1 sigma) |
| Per-particle spread of the time-averaged leeway | 0.0634 m/s vs 0.0633 m/s from the OU formula |
| Jibe flips in 1 h | 3.9 %, matching the Poisson prediction |
| Wind from 270° | person moves east |
| Same seed | bit-identical trajectories |

## 3. Probability map (belief)

### 3.1 State and prior

`p[ny, nx]` holds the probability that the person is in each 100 m cell (row 0 at the south),
`p_out` the probability that they are off the grid. Invariant: `p.sum() + p_out == 1` to 1e-9,
kept by bookkeeping (mass that leaves the grid is added to `p_out`), never by renormalising.

The prior is the Gaussian `N(LKP, sigma_lkp^2)` integrated exactly over each cell: the 2D
Gaussian is separable, so a cell's mass is a product of two differences of normal CDFs. Predict
steps then carry it from the fall to the start of the search.

### 3.2 Motion model

The map knows the forecast (wind, current, leeway coefficients), never the true person.

- Mean drift: `v_bar = U_current(map mean) + (a_d |W10| + b_d) e_dw`. The crosswind mean is 0
  because the side `s` is +1 or -1 with equal probability.
- Spread: for a stationary velocity noise with std `sigma` and correlation time `tau`, the
  displacement variance is exactly `V(t) = 2 sigma^2 tau (t - tau (1 - exp(-t/tau)))`. Each
  predict step adds `V(t + dt) - V(t)` for every noise source, so the map's variance follows the
  truth model at all times (ballistic early, diffusive late), not only for `t >> tau`.
- Noise sources and their correlation times:

| Source | Axis | Std | Correlation time |
| --- | --- | --- | --- |
| Leeway noise `eps_d` | downwind | 0.12 m/s | `tau` |
| Crosswind side `s` times mean crosswind leeway | crosswind | `a_c |W10| + b_c` | `1 / (2 lam)` (telegraph process, `lam` = jibe rate, ~12 h) |
| `s * eps_c` | crosswind | 0.094 m/s | `1 / (1/tau + 2 lam)` |
| Gusts | both, scaled by `a_d`, `a_c` | `a sigma_gust` | `tau_gust` (crosswind: combined with `lam`) |
| Current noise | both | `sigma_n` | `tau_n` |

- The two crosswind lobes are replaced by one Gaussian of the same variance. Over the first
  hours the lobes are narrower than the noise spread, so the approximation is good (see 3.4).

### 3.3 Predict step

One convolution of the map with `N(d, S)`, `d = v_bar dt`, `S = covariance increment`, computed
in Fourier space where the Gaussian has a closed form:

```
P'(k) = P(k) exp(-i k.d) exp(-k^T S k / 2)
```

- Exact for any sub-cell shift and any anisotropic, rotated or tiny covariance.
- No numerical diffusion. The brief's `ndimage.shift(order=1)` adds `f (1 - f)` cells^2 of
  variance per step for a fractional shift `f`, which the tests measure: 21 cells^2 after 100
  steps of 0.3 cell, as large as the physical spread.
- Zero padding (shift + 6 sigma) prevents wrap-around; mass in the padding goes to `p_out`.
  Small negative values from ringing next to sharp features (after Bayes updates) are clipped
  and the mass rescaled to its value before the step.
- Cost: ~0.2 ms per step on 100 x 100, 0.65 ms for predict + update on 200 x 200.
- The current is evaluated at the map's mean position. A semi-Lagrangian predict for spatially
  varying currents (shear, gyre) is a stretch item.

### 3.4 Validation against the truth drift model

After 2 h, against Monte Carlo clouds of `seaswarm.world.drift`:

| Scenario | Centroid error | Std (map / 5000 particles) | Hellinger, 5000 particles, 100 m / 400 m | Hellinger, 100 000 particles, 100 m |
| --- | --- | --- | --- | --- |
| calm_single | 0.10-0.14 cell | 558 x 492 m / 552-560 x 489-493 m | 0.18 / 0.05 | 0.049 |
| windy_mixed | 0.05-0.11 cell | 671 x 731 m / 658-663 x 741-742 m | 0.23 / 0.08 | 0.074 |

Two independent 5000-particle clouds of the same distribution are 0.25-0.29 apart at 100 m cells
(sampling noise, `H^2 ~ K / 8N` for `K` occupied cells), so the map is closer to each cloud than
another cloud is. The 0.15 acceptance threshold is therefore checked at 400 m tiles with 5000
particles, and at 100 m cells with 100 000 particles (`bench`).

## 4. Sensors and Bayes updates

### 4.1 Detection model

Lateral-range model, per look: `POD(r) = pod_max exp(-r^2 / (2 sigma_r^2))` for `r <= r_max`,
else 0. **Illustrative values, not validated sweep widths.**

| Sensor | r_max | sigma_r | pod_max day / night | Period |
| --- | --- | --- | --- | --- |
| Drone EO camera, 60 m, FOV 70° | 60 tan 35° = 42 m | 25 m | 0.6 / 0.1 | 2 s |
| Drone thermal camera, 60 m, FOV 50° | 60 tan 25° = 28 m | 18 m | 0.7 / 0.8 | 2 s |
| USV mast camera + lookout, 3 m | 150 m | 70 m | 0.5 / 0.2 | 5 s |

Drone cameras: `r_max = h tan(FOV/2)` and `pod_max(h) = pod_max(60 m) exp(-(h - 60 m) / 100 m)`,
clipped to [0, 1].

Successive looks are treated as independent. This is optimistic: real misses are correlated
(same sea state, same posture of the person), so the belief clears searched cells faster than it
should.

### 4.2 Cell-averaged POD

The likelihood of "no detection" given the person is in cell `i` is the POD averaged over the
cell, `POD_i = (1/A) ∫_cell POD(|x - s|) dx`, computed with a midpoint rule of spacing
`r_max / 8`. Evaluating POD only at the cell centre fails when the footprint is smaller than a
cell (42 m and 28 m cameras on 100 m cells). Check: the footprint integrates to the sensor's
coverage area `pod_max 2 pi sigma_r^2 (1 - exp(-r_max^2 / (2 sigma_r^2)))` within 1.2 %.

### 4.3 Bayes updates and probability of success

No detection:

```
POS_k = sum_i p_i POD_i                       (probability that this look succeeds)
p_i   <- p_i (1 - POD_i) / (1 - POS_k)
p_out <- p_out / (1 - POS_k)                  (POD = 0 off the grid)
POS_cum = 1 - prod_k (1 - POS_k)
```

Detection (v1, no false alarms): the belief is replaced by a Gaussian around the measured
position (sigma 10 m for drones, 20 m for USVs). The truth side rolls each look as a Bernoulli
trial with the POD at the true distance, using the `sensors` random stream.

## 5. Vehicles, safety and planning

### 5.1 Vehicles

Unicycle kinematics with heading-rate, speed and acceleration limits, integrated at 1 s; USVs are
carried by the surface current, drones ignore wind (v1). The heading and speed first-order loops
are discretised exactly (the error shrinks by `exp(-k dt)` per step): forward Euler gives
`err[n+1] = (1 - k dt) err[n]`, which with the drone's `k dt = 2` oscillated forever. Drones fly in altitude layers 60 / 80 /
40 m by index, so drones launched together are vertically separated; footprint and peak POD follow
from the altitude. Energy drains at `(idle + (1 - idle) (v / v_cruise)^2) / endurance` (equal to
`1 / endurance` at cruise; `idle` = 0.7 for a multirotor, 0.2 for a USV). Vehicles idle at the base
are parked: no drain, no looks. After a return to base: 300 s battery swap (drone) or 1800 s
refuel (USV).

| | Drone | USV |
| --- | --- | --- |
| Cruise / max speed | 15 / 20 m/s | 4 / 6 m/s |
| Max turn rate | 60 deg/s | 10 deg/s |
| Endurance at cruise | 1500 s | 28 800 s |
| Reserve | `t_rtb (1 + 0.2) + 120 s` | `t_rtb (1 + 0.2) + 900 s` |

### 5.2 Safety layer

Every command goes through `SafetyGuard`, which refuses with a code and a one-sentence reason:

| Rule | Code |
| --- | --- |
| Every waypoint inside the geofence (rectangle around area and base + 1 km; convex, so legs too) | `GEOFENCE` |
| No predicted approach within 30 s below 30 m (drones, same layer) / 50 m (USVs) plus the braking distance `v^2 / 2a`; not within 100 m of the base; a pair already too close may always move apart | `SEPARATION` |
| Energy after the task still covers the return with the reserve | `ENERGY_RESERVE` |
| One searcher per tile (dispatch to a detection exempt) | `TILE_OCCUPIED` |
| Vehicle unknown, returning, in turnaround or stranded | `VEHICLE_UNAVAILABLE` |

Recalls are never refused. Every step the guard (1) forces a return to base if the reserve would
be broken within 10 s; (2) re-projects every constrained pair's remaining planned paths over 30 s
and, on a predicted conflict, makes one vehicle hold (one in transit rather than one returning;
among equals the one commanded last) and triggers a replan, because paths accepted at different
times can converge after the command-time horizon; (3) counts actual violations (separation,
geofence, stranded) per episode.

### 5.3 Search theory used by the planners

- **Effective sweep width** W (Koopman): integral over lateral offset of the cumulative detection
  probability of the looks taken while passing at cruise speed. EO drone at 60 m: 46 m; thermal:
  29 m; USV: 239 m. Sweep rate W v: 2.5 km^2/h (EO drone), 3.4 km^2/h (USV).
- **Track spacing** `W (1 - 0.1)`.
- **POD of one tile sweep**: random-search formula `1 - exp(-W L / A)`.
- **Effort-limited sweep**: L is the full lawnmower length if affordable, otherwise the track
  length the vehicle can still fly before it must go home with the reserve, spread evenly over
  the tile (spacing A / L). A tile is feasible if one track across it fits. Without this, a weak
  sensor (EO camera at night, W of a few metres) makes every full pattern longer than a battery.
- **Utility** `U = w_prio POD_sweep p(tile) / (t_travel + t_search)`, `-inf` if infeasible.
  Sorties are cut to the longest affordable prefix of the pattern.

### 5.4 Strategies

- **Lawnmower** (baseline): the rectangle holding 95 % of the prior is split into north-south
  strips proportional to each vehicle's sweep rate; each vehicle mows its strip in
  energy-limited sorties, resuming where it stopped, shifted by the forecast drift. It ignores
  the belief after the start.
- **Greedy**: every replan, vehicles in id order take the feasible unclaimed tile of highest
  utility (tiles being searched stay claimed).
- **Auction**: sequential single-item auction with bundles over the top 3 x fleet tiles. Each
  round every vehicle bids its best marginal gain for inserting a remaining tile into its bundle
  (insertion heuristic); the highest bid wins (seeded tie-break); bundles stop at 3 tiles or when
  nobody bids a positive gain. Bundle value is the time-discounted reward
  `sum_j gain_j exp(-t_j / tau)`, tau = 1 h (diminishing marginal gains, as CBBA requires). Tiles
  held by another vehicle are not on offer (no swap cycles). Receding horizon: only the first tile
  is executed. Written as bid/award messages so a decentralised CBBA is a small step.

### 5.5 Mission loop and evaluation mode

Each 1 s step: backend looks, Bayes updates per look, belief predict every 60 s, safety watch,
replan on triggers. Start, periodic (120 s), detection, low energy, operator priority, refusal,
conflict hold and return to service replan at the next step; "vehicle idle" is rate-limited to
every 10 s. A refused (vehicle, tile) choice is excluded for one replan period, so an immediate
replan never repeats it. After a detection the detecting drone holds over the belief
mean, the nearest USV is dispatched until it is within 25 m of the person (recovery), and the
other vehicles return. In `search_only` mode (evaluation) every look counts as unsuccessful and
the run continues to its time limit, so POS_cum measures the planned search alone; it is then
identical for every seed, which a test checks.

### 5.6 Evaluation method

- 60 random scenarios (`scripts/generate_scenarios.py --seed 0`): wind 2-15 m/s, current
  0-0.8 m/s, random directions, night 30 %, LKP placed upstream of the expected drift so the
  person stays in the area. Seed N is scenario gen_N.
- Per (strategy, fleet, seed): a normal run (detection, recovery, safety) and a 120 min
  search-only run (POS_cum every minute).
- Fraction found vs time: Kaplan-Meier (detections are right-censored at the 3 h limit; with a
  common limit it equals the empirical fraction, but it stays correct with mixed run lengths).
- Intervals: 95 % percentile bootstrap over scenarios, fixed seed. Strategy comparisons are
  paired on identical scenarios and seeds, which removes scenario-to-scenario variance.

### 5.7 Decentralised allocation: CBBA (stretch)

`CBBAPlanner` (Choi, Brunet and How 2009) allocates the auction's tasks with the same
time-discounted bundle score, but without an auctioneer: each vehicle only exchanges messages
with vehicles within radio range. Rounds alternate bundle construction (add tasks whose marginal
gain beats the best known bid), consensus (merge neighbours' winning bids, winners and
timestamps with the paper's update / reset / leave table) and release (a vehicle outbid on a task
drops it and every task it added afterwards). It stops when a round changes nothing. The belief
is shared (as if broadcast); only the allocation is decentralised.

Evaluation, 50 generated scenarios per fleet, `results/cbba/` (`--strategy auction,cbba@inf,
cbba@3000,cbba@1000,cbba@300`), zero safety violations in 500 runs:

| Fleet | Strategy | Found <= 60 min | vs central auction | Median time to find | Tile conflicts / run |
| --- | --- | --- | --- | --- | --- |
| 2D+1U | auction | 68% [56%, 80%] | — | 34.4 min | 0.0 |
| 2D+1U | cbba@inf | 68% [56%, 80%] | +0% [+0%, +0%] | 34.4 min | 0.0 |
| 2D+1U | cbba@3000 | 72% [60%, 84%] | +4% [-10%, +18%] | 33.6 min | 0.8 |
| 2D+1U | cbba@1000 | 68% [54%, 80%] | +0% [-12%, +10%] | 32.9 min | 2.3 |
| 2D+1U | cbba@300 | 70% [56%, 82%] | +2% [-10%, +16%] | 32.0 min | 4.9 |
| 4D+2U | auction | 80% [68%, 90%] | — | 30.0 min | 0.0 |
| 4D+2U | cbba@inf | 86% [76%, 94%] | +6% [+0%, +14%] | 26.5 min | 0.0 |
| 4D+2U | cbba@3000 | 78% [66%, 88%] | -2% [-12%, +6%] | 34.2 min | 1.8 |
| 4D+2U | cbba@1000 | 86% [76%, 94%] | +6% [-6%, +20%] | 24.3 min | 5.4 |
| 4D+2U | cbba@300 | 86% [76%, 94%] | +6% [-6%, +20%] | 28.0 min | 14.1 |

- With full communication CBBA matches the central auction (identical with 2 drones + 1 USV).
- Shorter radio range multiplies tile conflicts (vehicles that cannot hear each other plan onto
  the same tile), but outcomes do not degrade: every paired interval includes zero. The safety
  layer refuses the second vehicle (one searcher per tile) and the next replan resolves it, and
  vehicles searching the same high-probability area are usually within range of each other.
  The cost of poor communication is wasted proposals and replanning, not missed persons.

## 6. Operator agent and MCP server

### 6.1 Layers

```
operator (plain language) -> CodeAgent (qwen3:8b via Ollama) -> MCPClient (stdio)
  -> FastMCP server -> Operations -> SearchMission -> SafetyGuard -> Sim2DBackend
```

The agent never imports the simulator: it starts the server as a subprocess and only calls its
tools. Every vehicle command, from the agent or from the planner, goes through the same safety
guard.

### 6.2 Tools

Ten tools, SI units, documented limits, JSON objects out, never raising:
`get_situation`, `get_probability_map`, `set_priority_area`, `set_priority_sector` (3 x 3 split of
the search area), `clear_priority_areas`, `assign_vehicle`, `dispatch_to_detection`,
`recall_vehicle`, `set_strategy`, `advance_simulation` (at most 600 s per call: the operator
controls simulated time). Refused or invalid requests return
`{"ok": false, "refusal": {"code", "reason", "vehicle_id"}}`, with the safety layer's sentence
as the reason. Positions are also given as bearing and distance from the base.

### 6.3 Operator commands and the automatic planner

A vehicle under a direct operator command (`assign_vehicle`, `dispatch_to_detection`,
`recall_vehicle`) is left alone by the planner and the automatic recovery until that command ends
(task finished, back at base, or stranded); then it rejoins the automatic search. A USV the
operator dispatches to a detection becomes the recovery vehicle. Priority areas change planner
utility only, never the probability map.

### 6.4 Agent

- smolagents `CodeAgent`, `LiteLLMModel("ollama_chat/qwen3:8b", num_ctx=32768, temperature 0,
  think off)`, 12 steps per instruction; a cloud model only through `SEASWARM_MODEL_ID`.
- `MCPClient(..., structured_output=False)` returns tool results as JSON text; a thin wrapper
  parses it into a dict and maps positional arguments, so the agent's code reads
  `situation["vehicles"]`.
- System prompt: SAR coordinator; get_situation first; unit conversions written out (knots,
  nautical miles, bearing and distance to x, y); never invent ids; never repeat a refused call;
  verify with get_situation; 5-line situation report.
- Tests never call an LLM: `ScriptedModel` replays code blocks through the real MCP stack.

## 7. Limitations

- Leeway coefficients are fleet-average regressions for one object class; real persons differ.
- Fields are synthetic and spatially simple; no tides, Stokes drift or coastline.
- The crosswind term is symmetric (true for PIW-1, not for every object class).
- The belief approximates the two crosswind lobes by one Gaussian and uses the current at its
  mean position.
- Looks are independent and sensors raise no false alarms.
- Drones ignore wind; turning is not modelled in separation predictions (straight legs at
  constant speed).
- POS_cum is slightly optimistic against the truth (within 2 sigma over 100 seeds).
- Several vehicles launching together still produce separation refusals at the base (resolved by
  replanning); there is no launch or approach sequencing.
- Effort-limited sweeps assume the effort is spread evenly over a tile; bundle tasks after the
  first are scored with the vehicle's current energy.

## References

- H.-L. Choi, L. Brunet, J. P. How, "Consensus-Based Decentralized Auctions for Robust Task
  Allocation", IEEE Transactions on Robotics 25(4), 2009.
- E. L. Kaplan, P. Meier, "Nonparametric Estimation from Incomplete Observations", Journal of the
  American Statistical Association 53(282), 1958.

- B. O. Koopman, "Search and Screening: General Principles with Historical Applications",
  Pergamon, 1980.

- L. D. Stone, C. M. Keller, T. M. Kratzke, J. P. Strumpfer, "Search for the Wreckage of Air
  France Flight AF 447", Statistical Science 29(1), 2014 (arXiv:1405.4720).
- L. D. Stone, J. O. Royset, A. R. Washburn, "Optimal Search for Moving Targets", Springer, 2016.

- A. A. Allen, J. V. Plourde, "Review of Leeway: Field Experiments and Implementation",
  US Coast Guard report CG-D-08-99, 1999.
- Ø. Breivik, A. A. Allen, "An operational search and rescue model for the Norwegian Sea and the
  North Sea", Journal of Marine Systems 69, 2008.
- OpenDrift, `opendrift/models/OBJECTPROP.DAT` and the `Leeway` model.
