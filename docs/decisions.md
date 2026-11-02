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
