================================================================================
PyPSA-Libya: Electricity-Only Capacity Expansion Study (No Sector Coupling)
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This study models Libya's power sector alone, using PyPSA-Earth without the
sector-coupling extension (no hydrogen export, no industry/heat/transport
demand). It serves as a baseline reference for the separate PyPSA-Libya
sector-coupled hydrogen export study, establishing what Libya's optimal
generation and storage mix looks like under pure electricity-sector
decarbonisation pressure, before layering hydrogen production and export
on top.

Two cases are compared, identical in every respect except the CO2 emissions
constraint, to show how sensitive the optimal generation mix and total
system cost are to climate ambition:

  CASE 1 — Limited CO2 (co2.limit: 0.0)
    A hard ZERO-EMISSIONS constraint. Total system CO2 emissions must not
    exceed zero, meaning NO fossil generation (OCGT, CCGT, oil) can be
    dispatched at all — the system must be served entirely by solar,
    onshore wind, and storage (battery and H2, both modelled here as
    simple round-trip StorageUnits, not as a hydrogen production/export
    chain — see NOTE ON STORAGE REPRESENTATION below).

  CASE 2 — No CO2 limit (co2.limit: 1.0e9)
    A non-binding placeholder cap, effectively unconstrained. No real
    Libyan national CO2 target currently exists to calibrate this
    figure against, so 1.0e9 is used specifically because it is far
    above anything the modelled system could plausibly emit, i.e. a
    genuine "no climate constraint" cost-minimal baseline.

Comparing these two cases in a single, otherwise-identical setup isolates
the cost and technology-mix impact of full decarbonisation specifically,
without also changing spatial resolution, time horizon, or any other
model dimension.


MODEL BASIS
--------------------------------------------------------------------------------
- Framework:        PyPSA-Earth (Snakemake-orchestrated workflow),
                     electricity-only target (solve_all_networks — NOT
                     solve_sector_networks, which is used only in the
                     separate hydrogen-export study).
- Geographic scope: Libya only (countries: ["LY"]).
- Spatial resolution: 36 clustered nodes.
- Temporal scope:   Full year 2013 (2013-01-01 to 2014-01-01), resampled
                     to 3-hour resolution (opts: Co2L-3h).
- Weather data:     ERA5 reanalysis, custom-built cutout restricted to
                     Libya's bounding box (reused from earlier project
                     stages; not rebuilt for this run).
- Cost year:        2030 (technology-data cost assumptions).
- Solver:           HiGHS (interior-point method, crossover off, relaxed
                     feasibility/optimality tolerances for tractability
                     at this problem size — a full year x 36 clusters is
                     a substantially larger LP than earlier single-day
                     test runs in this project's development history).
- Line expansion:   Enabled, with a guaranteed minimum of one connection
                     per node (augmented_line_connection.connectivity_
                     upgrade: 1) — added specifically because earlier
                     testing in this project found genuinely isolated,
                     zero-connection buses in Libya's OSM-derived
                     transmission data (see LIMITATIONS below).


NOTE ON STORAGE REPRESENTATION (important distinction from the H2-export study)
--------------------------------------------------------------------------------
This model represents both "battery" and "H2" as generic StorageUnit
components — i.e. simple round-trip electricity storage (charge from the
grid, discharge back to the grid later), NOT as a hydrogen production and
export chain. There is no electrolysis link, no H2 pipeline network, no
export demand, and no distinction between "hydrogen for storage" and
"hydrogen as a commodity" in this study. The "H2" label here should be
read as "a storage technology with H2-like round-trip efficiency and
cost characteristics," not as evidence of hydrogen production capacity.
This is a deliberate simplification to keep this study focused purely on
electricity-sector questions; the separate sector-coupled study is where
genuine hydrogen production/export modelling happens.


WHAT TO EXPECT / KEY MODELLING CONSIDERATIONS
--------------------------------------------------------------------------------
- A hard co2.limit: 0.0 is an extreme, maximally demanding constraint —
  it does not merely discourage fossil generation, it prohibits it
  entirely (any nonzero OCGT/CCGT/oil dispatch produces some positive
  emissions, which a strict zero cap disallows). Expect Case 1 to show
  substantially higher renewable and storage capacity buildout, and
  materially higher total system cost, than Case 2.

- Given this project's prior findings on Libya's OSM-derived transmission
  data (documented gaps, including at least one previously-identified
  fully isolated bus with zero line connections before the connectivity-
  upgrade setting was introduced), results — especially any load
  shedding — should be interpreted with that data-completeness caveat in
  mind, not assumed to reflect a purely economic or purely resource-
  driven outcome without checking the underlying network structure first.

- This project's development history includes at least one prior instance
  where an apparently large "load shedding" figure turned out to be a
  metric-interpretation error (conflating a fixed non-extendable capacity
  ceiling with actual dispatched shedding). Any load-shedding results from
  this study should be verified using actual dispatch (generators_t.p),
  not p_nom/p_nom_opt, before being reported as a real finding.


KEY LIMITATIONS AND OPEN QUESTIONS
--------------------------------------------------------------------------------
- TRANSMISSION DATA COMPLETENESS: as noted above, Libya's OSM-derived
  grid topology has known gaps. The connectivity-upgrade setting
  mitigates full isolation but does not guarantee the resulting network
  reflects Libya's true grid topology with confidence.

- WIND VS. SOLAR ECONOMICS: prior testing in this project (electricity-
  only, at various cluster counts) consistently found onshore wind
  priced out in favour of solar under 2030 technology-cost assumptions,
  despite wind having an unconstrained resource potential and a higher
  capacity factor in the underlying data. This is expected to hold here
  too, though has not been separately re-verified at 36 clusters/full
  year specifically for this study.

- COST YEAR SENSITIVITY: results reflect 2030 technology costs only; no
  2040/2050 sensitivity has been run for this electricity-only study.

- NO REAL LIBYAN CO2 TARGET: neither case's CO2 assumption is calibrated
  to an actual published Libyan climate commitment (none is known to
  exist with the specificity needed for this kind of modelling); both
  "zero" and "unconstrained" should be read as bounding cases for
  exploring sensitivity, not as policy-realistic scenarios.

- NOT DIRECTLY COMPARABLE TO THE SECTOR-COUPLED STUDY'S COST FIGURES:
  because this model excludes hydrogen production/export, industry, and
  all other coupled sectors, its total system costs are not on the same
  basis as the sector-coupled study's Net/Pro-Rata LCOH figures and
  should not be combined or compared without carefully accounting for
  scope differences.


ANALYSIS AND VISUALIZATION
--------------------------------------------------------------------------------
Results are processed with visualize_libya_year.py, which produces
capacity-by-carrier breakdowns and spatial capacity maps for each case,
enabling direct visual comparison of how the CO2 constraint reshapes the
optimal generation and storage mix.


REPRODUCING THIS STUDY
--------------------------------------------------------------------------------
1. Clone/fork pypsa-meets-earth/pypsa-earth as the base repository.
2. Replace config.yaml with the desired case:
   - No_Sector_Coupling_36Clusters_LimitedCO2_config.yaml   (Case 1)
   - No_Sector_Coupling_36Clusters_NoLimitedCO2_config.yaml (Case 2)
3. Run: snakemake --cores <N> solve_all_networks
4. Run visualize_libya_year.py against the resulting solved network to
   generate capacity breakdowns and maps.

================================================================================
Document generated as part of PyPSA-Libya project development.
================================================================================
