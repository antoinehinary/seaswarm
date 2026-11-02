# Decision log

One entry per real decision: date, decision, why, alternatives considered.

## 2026-11-01 — Licence: AGPL-3.0

- **Decision:** release under AGPL-3.0-only.
- **Why:** keeps the project open source while preventing anyone from turning it into a closed
  product or a closed hosted service: modified versions, including network services, must publish
  their source under the same licence. The author remains free to relicense their own code.
- **Alternatives:** MIT (used by skypilot-agent; allows closed commercial reuse), GPL-3.0 (no
  network-service clause), PolyForm Noncommercial (not open source).

## 2026-11-01 — Pin `mcp>=1.20,<2`

- **Decision:** stay on the MCP Python SDK 1.x line and use `mcp.server.fastmcp.FastMCP`.
- **Why:** mcp 2.x renamed FastMCP and smolagents' `MCPClient` (via mcpadapt) fails to import with
  it. mcpadapt declares no upper bound, so the pin must live in this project.
- **Alternatives:** mcp 2.x with a hand-written MCP client in place of smolagents' `MCPClient`.

## 2026-11-01 — No OpenDrift dependency in the core package

- **Decision:** implement our own leeway model with OpenDrift's published PIW-1 coefficients;
  OpenDrift is only used, if at all, as an external reference in a separate environment.
- **Why:** OpenDrift is GPL-2.0, pulls heavy geo dependencies (GDAL, cartopy, netCDF) and needs
  gridded forcing data; a 2D synthetic-field simulator needs none of that.
- **Alternatives:** depend on OpenDrift directly.

## 2026-11-01 — Exact discretisation of the Ornstein–Uhlenbeck noise

- **Decision:** advance every OU process (leeway noise, current noise, wind gusts) with the exact
  update `x <- x*exp(-dt/tau) + sigma*sqrt(1 - exp(-2*dt/tau))*N(0,1)`, starting from the
  stationary distribution `N(0, sigma^2)`.
- **Why:** it keeps the stationary standard deviation exactly `sigma` for any `dt`, so larger
  steps (for fast batch evaluation) do not change the drift statistics. At `dt = 1 s`,
  `tau = 1800 s` it matches Euler–Maruyama to within 0.03 %.
- **Alternatives:** Euler–Maruyama `x += -x/tau*dt + sigma*sqrt(2*dt/tau)*N(0,1)`, only accurate
  for `dt << tau`.

## 2026-11-01 — Wind gusts as an OU process, sharing the leeway noise code

- **Decision:** model gusts as OU fluctuations of the wind speed, using the same `OUProcess` class
  as the leeway and current noise. Off by default.
- **Why:** one tested component serves three uses. The effect on drift is small (leeway is about
  1 % of wind speed), so gusts stay optional.
- **Alternatives:** leave gusts out of v1.

## 2026-11-01 — Load `.env` with python-dotenv

- **Decision:** add `python-dotenv` as a dependency to load optional model settings and keys.
- **Why:** same pattern as skypilot-agent; lets the operator switch model without exporting
  variables in every shell. Environment variables already set always take precedence.
- **Alternatives:** read `os.environ` only.

## 2026-11-01 — ENU origin at the search-area centre, not at the base

- **Decision:** the local ENU frame has its origin at the centre of the search area; the base is
  a configurable point in that frame (for example `(0, -6000)`, 1 km south of a 10 km area).
- **Why:** scenario files place the base outside the area, which only makes sense with an
  area-centred origin. The grid is then symmetric about the origin and grid indexing needs no
  offset. Bearings and distances "from base" are computed relative to the base point.
- **Alternatives:** origin at the base, with an explicit area centre in every scenario.

## 2026-11-01 — Scenario schema lives in `config.py`

- **Decision:** the pydantic `Scenario` model is defined in `seaswarm.config` from Phase 1;
  `eval/scenario.py` (Phase 4) will only hold the random scenario generator.
- **Why:** the drift demo already needs scenarios, and keeping every YAML schema in one module
  makes the "no unknown keys" rule easy to audit.
- **Alternatives:** schema in `eval/scenario.py` imported by the world modules (makes the
  simulator depend on the evaluation package).

## 2026-11-02 — Belief spread follows the exact OU variance, D(t) instead of constant D

- **Decision:** the predict step adds, per step, the difference of the exact cumulative
  displacement variance `2 sigma^2 tau (t - tau (1 - exp(-t/tau)))` of every noise process
  (leeway OU noise, crosswind side as a telegraph process, gusts, current noise).
- **Why:** a constant `D = sigma^2 tau` is only valid for `t >> tau` (30 min); before that real
  particles spread ballistically and a constant D makes the map too wide in the first hour,
  exactly when the search matters most. The jibing crosswind side is a telegraph process with
  autocorrelation `exp(-2 lam |lag|)`, so the same formula covers the two crosswind lobes.
- **Alternatives:** constant D (brief's first version).

## 2026-11-02 — Spectral predict step instead of `ndimage.shift` + `gaussian_filter`

- **Decision:** one convolution with N(v_bar dt, S) per step, computed as a product in Fourier
  space with zero padding (mass reaching the padding goes to `p_out`).
- **Why:** linear-interpolation shifting adds `f (1 - f)` cells^2 of spurious variance per step
  (measured in `test_linear_interpolation_shift_adds_numerical_diffusion`: 21 cells^2 after 100
  steps of 0.3 cell, as large as the physical spread); and per-step physical variances (~0.05-0.3
  cells^2) are too small for a sampled Gaussian kernel to represent. The Fourier form is exact for
  any sub-cell shift and any anisotropic covariance, and costs ~0.2 ms on 100 x 100.
- **Alternatives:** the brief's shift + filter (numerically diffusive); integer shifts with a
  carried sub-cell remainder (no diffusion but up to half a cell of lag).

## 2026-11-02 — Spatially varying currents in the belief predict moved to stretch

- **Decision:** the belief uses the current at the map's mean position; the semi-Lagrangian
  `map_coordinates` predict for shear and gyre currents is on the Phase 6 stretch menu.
- **Why:** keeps Phase 2 within 16 h. For `windy_mixed` (0.02 m/s per km of shear) the
  belief-vs-particles check still matches the spread within ~2 %.
- **Alternatives:** implement it in Phase 2 (+1.5 h).

## 2026-11-02 — Belief-vs-particles acceptance test measured at tile resolution

- **Decision:** with 5000 particles the Hellinger criterion (< 0.15) is evaluated on 400 m tiles
  (the planner's resolution), and the map must also be closer to the particles than a second
  independent 5000-particle cloud is. A 100 000-particle check at 100 m cells is marked `bench`.
- **Why:** at 100 m cells, 5000 particles cannot reach 0.15 whatever the model: two independent
  clouds of the same distribution are 0.25-0.29 apart (sampling noise, H^2 ~ K / 8N for K
  occupied cells). Against 100 000 particles the map is 0.05 (calm) and 0.07 (windy) away at
  100 m cells, so the model error itself is well below 0.15.
- **Alternatives:** 100 000 particles in the default suite (too slow for the 60 s budget);
  kernel-smoothing the particles (adds its own bias).

## 2026-11-02 — Cell-averaged POD instead of POD at the cell centre

- **Decision:** the belief update uses, for each cell, the POD averaged over the cell area
  (midpoint rule with spacing r_max / 8), not the POD at the cell centre.
- **Why:** the likelihood of "no detection" given the person is in a cell must average over where
  in the cell they are. With 100 m cells and a 42 m (EO) or 28 m (thermal) camera footprint,
  centre sampling gives 0 or full POD depending on whether the drone happens to pass within r_max
  of a centre, so a drone flying between centres would clear nothing. The averaged footprint
  integrates to the sensor's coverage area `pod_max 2 pi sigma^2 (1 - exp(-r_max^2 / 2 sigma^2))`
  within 1.2 % for every sensor, at ~20 us per look. Spacing is tied to r_max, not to the cell,
  because the hard cut-off at r_max limits the accuracy (a fixed 5 x 5 rule was off by 19 % for
  the thermal camera).
- **Alternatives:** POD at the cell centre (brief's first version); finer grid cells (cost grows
  with the square of the resolution everywhere, not only under the sensors).

## 2026-11-02 — Drone energy drain keeps a hover term

- **Decision:** drain rate `(idle + (1 - idle) (v / v_cruise)^2) / endurance` with
  `idle = 0.7` for drones and `0.2` for USVs, instead of `(v / v_cruise)^2 / endurance`.
- **Why:** with the pure quadratic law a hovering multirotor uses no energy, so a drone told to
  hold would never need to return to base. A multirotor's hover power is close to its cruise
  power. The rate still equals `1 / endurance` at cruise, as in the brief.
- **Alternatives:** the brief's quadratic law (hover is free).

## 2026-11-02 — Drones fly in altitude layers 60 / 80 / 40 m by index

- **Decision:** drone i is assigned altitude layer i mod 3 in the order 60, 80, 40 m. Its sensor
  footprint and peak POD follow from that altitude.
- **Why:** all vehicles start together at the base. Two drones heading for neighbouring tiles
  diverge too slowly to be 30 m apart when they leave the 100 m base zone (they need ~17° between
  headings), so the predictive separation check refused simultaneous launches. Vertical layers
  remove the horizontal constraint between drones in different layers, as in the brief's rule.
- **Alternatives:** staggered launches (slower start), all drones at 60 m with a larger base zone
  (hides real conflicts).

## 2026-11-02 — Safety guard owns planned paths and tile ownership

- **Decision:** the guard records every accepted command's path (for the 30 s predictive
  separation check of others) and which vehicle searches which tile; a tile is held only while
  its vehicle is in transit. Recalls (RTB, Hold) are never refused. Separation is not enforced
  within 100 m of the base (deck operations). The geofence is the rectangle around the area and
  the base plus 1 km; being convex, checking the waypoints keeps every straight leg inside.
- **Why:** commands carry only waypoints, so the guard is the one place that sees every plan.
- **Alternatives:** have the planner report all paths to the guard (two sources of truth).

## 2026-11-02 — Sweep width and POD of a tile sweep from classical search theory

- **Decision:** each vehicle's effective sweep width W is computed numerically from its lateral
  range curve (cumulative POD of the looks taken while passing at cruise speed); lawnmower track
  spacing is `W (1 - overlap)`, i.e. `2 r_eff (1 - overlap)` with `r_eff = W / 2`; the POD of
  one sweep of a tile is Koopman's random-search formula `1 - exp(-W L / A)`.
- **Why:** gives every vehicle type a comparable, physically meaningful sweep rate (W v): EO drone
  at 60 m ~46 m (2.5 km^2/h), USV ~239 m (3.4 km^2/h). The random-search formula is the
  conservative textbook value for the expected POS gain used in the utility.
- **Alternatives:** `r_eff = r_max` (ignores how many looks a point gets and the POD shape);
  the definite-range law (optimistic).

## 2026-11-02 — Strategy comparison uses a search-only mode

- **Decision:** POS_cum and "detected by 60 min" are measured in `search_only` runs: detections
  are recorded but do not end the search, every look updates the belief as unsuccessful, and
  recovery does not stop the run. Safety criteria are checked on normal full runs.
- **Why:** in a normal run the belief collapses at the first detection, so a strategy that finds
  the person early would show a lower POS_cum. In search-only mode POS_cum is the success
  probability of the search as planned, identical for every seed (a run-time check of the
  truth/belief separation, now a test); the seeds only change where the person really is.
- **Alternatives:** POS_cum frozen at the detection time (penalises early success).

## 2026-11-02 — Phase 3 result: greedy vs lawnmower, fleet 2D+1U, seeds 0-19

| Scenario | Strategy | Detected <= 60 min | POS_cum @ 30 min | POS_cum @ 60 min | Median detection |
| --- | --- | --- | --- | --- | --- |
| windy_mixed | lawnmower | 11/20 | — | 0.615 | 47.5 min |
| windy_mixed | greedy | 16/20 | — | 0.775 | 30.2 min |
| calm_single | lawnmower | 14/20 | 0.673 | 0.908 | 28.3 min |
| calm_single | greedy | 20/20 | 0.715 | 0.906 | 16.4 min |

All 80 full runs: 0 separation or geofence violations, 0 stranded vehicles, every run within its
step limit.

**Why greedy does not beat lawnmower on POS_cum at 60 min in calm_single.** With light wind the
belief stays compact (about 1 km across), and by 60 min both strategies have swept nearly all of
it: POS_cum saturates near 0.91 for both. Greedy's advantage is *when* probability is cleared,
not how much by a late deadline: it visits the densest tiles first (0.715 vs 0.673 at 30 min),
which is what matters for a person in cold water, and it detects in every seed by 60 min (median
16 min vs 28 min). A fixed-horizon POS_cum hides this; Phase 4 reports the whole POS(t) curve and
the fraction found vs time instead. In windy_mixed the belief spreads faster than the fleet can
sweep it, and greedy's focus pays off on both metrics.

**Calibration.** Over 100 seeds per case (60 min, search-only), the observed detection fraction
is within 2 sigma of the model's POS_cum in all four cases (-1.7, +0.8, -1.3, -1.6 sigma). The
slight negative tendency is consistent with the belief's approximations (one Gaussian for the two
crosswind lobes, current at the map mean, predict up to 60 s behind the looks).

## 2026-11-02 — Auction: time-discounted bundle reward, receding horizon

- **Decision:** bundle value `sum_j gain_j exp(-t_j / tau)`, tau = 3600 s, t_j the time task j
  would be finished; bundles of up to 3 tiles over the top 3 x fleet tiles; each vehicle executes
  only its first task and the auction is rerun at every replan. Tiles held by another vehicle,
  or recently refused for this one, are not on offer.
- **Why:** the discounted reward is the standard CBBA score: it has diminishing marginal gains
  (needed for CBBA's convergence guarantees) and encodes that clearing probability later is worth
  less for a person in water. Ruling out held tiles prevents swap cycles that no submission order
  can resolve (seen as TILE_OCCUPIED refusals before the rule).
- **Alternatives:** bundle rate `sum gain / total time` (not submodular: bundles rarely grow).

## 2026-11-02 — Replan triggers are immediate except "vehicle idle"

- **Decision:** detection, low energy, operator priority, refusal, return to service and start
  replan at the next step; only "vehicle idle" is rate-limited (10 s). After a refusal the
  refused (vehicle, tile) choice is excluded for one replan period instead of being retried.
- **Why:** a refused proposal retried unchanged is refused again; excluding it lets the replan be
  immediate without thrashing. Same rule the operator agent follows in Phase 5.
- **Alternatives:** a cooldown on every event trigger (misses the one-step requirement).

## 2026-11-02 — Exact discretisation of the heading and speed loops (bug fix)

- **Decision:** the unicycle's heading and speed first-order loops shrink their error by
  `exp(-k dt)` per step (then rate/acceleration limits), instead of forward Euler.
- **Why:** with forward Euler, `err[n+1] = (1 - k dt) err[n]`; the drone's `k = 2 /s` at
  `dt = 1 s` gives `-err`: the heading flipped between two values forever (seen as a +-25 deg
  wobble in the 4-drone evaluation). The wobble made straight-line conflict predictions wrong and
  caused real separation violations. Phase 3's test missed it because a large turn is rate-limited
  and lands exactly on target; a small-error regression test now checks monotone decay.
- **Alternatives:** lower gains so that `k dt < 1` (stable, but still dt-dependent).

## 2026-11-02 — Per-step predictive conflict watch in the safety guard

- **Decision:** every step the guard re-projects each constrained pair's remaining planned paths
  over the 30 s horizon; on a predicted conflict one moving vehicle holds (a vehicle in transit
  rather than one returning to base; among equals the one commanded last) and the mission
  replans it (`conflict` trigger).
- **Why:** the command-time check only covers 30 s after each command; paths accepted at
  different times can converge later (two USVs turning on a shared tile edge, a drone launching
  as another returns). Before the watch, 4-drone and 2-USV fleets had separation violations;
  with the watch and the heading fix, 120 runs across all fleets and strategies had none.
- **Alternatives:** longer command-time horizon (still blind to later convergence), reactive
  avoidance manoeuvres (more realistic, harder to explain and to test).
