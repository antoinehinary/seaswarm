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
