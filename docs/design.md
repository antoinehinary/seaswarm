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

## 3. Limitations

- Leeway coefficients are fleet-average regressions for one object class; real persons differ.
- Fields are synthetic and spatially simple; no tides, Stokes drift or coastline.
- The crosswind term is symmetric (true for PIW-1, not for every object class).

## References

- A. A. Allen, J. V. Plourde, "Review of Leeway: Field Experiments and Implementation",
  US Coast Guard report CG-D-08-99, 1999.
- Ø. Breivik, A. A. Allen, "An operational search and rescue model for the Norwegian Sea and the
  North Sea", Journal of Marine Systems 69, 2008.
- OpenDrift, `opendrift/models/OBJECTPROP.DAT` and the `Leeway` model.
