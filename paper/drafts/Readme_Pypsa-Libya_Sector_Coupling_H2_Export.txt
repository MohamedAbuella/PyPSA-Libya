================================================================================
PyPSA-Libya: Sector-Coupled Hydrogen Export Study
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project adapts PyPSA-Earth (an open-source energy system modelling
framework, https://github.com/pypsa-meets-earth/pypsa-earth) into a
country-specific model of Libya's power and hydrogen system, "PyPSA-Libya",
to evaluate the technical and economic feasibility of exporting green and/or
blue hydrogen from Libya to European markets.

The methodology follows the sector-coupled hydrogen-export approach
demonstrated for Egypt in:

    Abdel-Khalek, H. et al. (2025). "Green hydrogen production and export
    potential..." Applied Energy, 383, 125316.
    https://www.sciencedirect.com/science/article/pii/S0306261925000467

That paper compares three hydrogen-production technology scenarios and, for
each, runs the model once with a hydrogen export requirement and once
without, using the difference to isolate the additional system cost (and
therefore levelized cost) attributable specifically to exporting hydrogen.
PyPSA-Libya follows the same three-scenario structure, adapted to Libya's
own geography, resources, and grid data.


MODEL BASIS
--------------------------------------------------------------------------------
- Framework:        PyPSA-Earth (Snakemake-orchestrated workflow), with the
                     sector-coupling extension enabled for hydrogen.
- Geographic scope: Libya only (countries: ["LY"]).
- Spatial resolution: 12 clustered nodes (reduced from earlier 36/40-node
                     test runs to keep the sector-coupled LP computationally
                     tractable — see LIMITATIONS below).
- Temporal scope:   Full year 2013 (2013-01-01 to 2014-01-01), resampled to
                     6-hour resolution (opts: Co2L-6h).
- Weather data:     ERA5 reanalysis, custom-built cutout restricted to
                     Libya's bounding box (not the full-Africa bundled
                     default — see project history for why this distinction
                     matters for both correctness and download size).
- Cost year:        2030 (technology-data cost assumptions).
- Solver:           HiGHS (interior-point method, crossover off, relaxed
                     feasibility/optimality tolerances for tractability).
- Sectors enabled:  Electricity + hydrogen + industry. Heat, biomass,
                     shipping, aviation, land/rail transport, agriculture,
                     residential, and services sectors are all disabled,
                     since they are unrelated to the hydrogen-export
                     question this study is scoped to answer.


HYDROGEN EXPORT ASSUMPTIONS (common to all scenarios)
--------------------------------------------------------------------------------
- Export volume:        70 TWh/year of hydrogen.
- Export mode:          Shipping (export_profile: "ship"), not pipeline.
- Ship parameters:      0.4 (unit capacity), 288h travel time, 24h fill
                         time, 24h unload time per cycle.
- Underground storage:  Enabled for all scenarios (see LIMITATIONS — the
                         geological potential backing this assumption for
                         Libya specifically has not been independently
                         verified).


THE THREE SCENARIOS
--------------------------------------------------------------------------------
Each scenario differs in which hydrogen production technologies the
optimizer is allowed to choose from, mirroring the Egypt paper's structure:

  SCENARIO A — Electrolysis only
    production_technologies: [H2 Electrolysis]
    temporal_matching: "year"
    co2.limit: 1.0e7
    Represents a strictly green-hydrogen pathway: all H2 must come from
    water electrolysis, and the "yearly matching" policy requires exported
    hydrogen to be backed, on an annual basis, by an equivalent amount of
    renewable generation somewhere in the system (the loosest form of
    genuine additionality accounting — not hourly or bus-level matching).

  SCENARIO B — Electrolysis + SMR with carbon capture
    production_technologies: [H2 Electrolysis, SMR CC]
    temporal_matching: "no_temporal_matching"
    co2.limit: 1.0e7
    Allows the model to also build carbon-captured steam methane reforming
    (SMR CC) capacity — "blue hydrogen" — alongside electrolysis, with no
    green-hydrogen matching requirement imposed.

  SCENARIO C — Electrolysis + SMR + SMR with carbon capture
    production_technologies: [H2 Electrolysis, SMR, SMR CC]
    temporal_matching: "no_temporal_matching"
    co2.limit: 1.0e12
    All three technologies available, including uncontrolled-emissions SMR.
    This is the least-constrained scenario both technologically and on
    emissions, and is expected to yield the lowest-cost (but highest-
    emissions) hydrogen of the three.

NOTE ON CO2 LIMITS ACROSS SCENARIOS: Scenarios A and B share the same CO2
cap (1.0e7), but Scenario C uses a dramatically looser cap (1.0e12,
effectively non-binding). This is an intentional design choice reflecting
that Scenario C is meant to represent an "anything goes, cost-minimal"
case — but it does mean cost comparisons between C and A/B are not purely
about technology choice; the emissions constraint itself also differs.
This should be stated explicitly in any write-up of results, not left
implicit, to avoid the comparison being read as apples-to-apples when it
isn't quite.


COST METRICS COMPUTED
--------------------------------------------------------------------------------
Two metrics are reported, both computed using a FULL 6-RUN approach: each
of the three scenarios (A, B, C) is solved twice — once with the 70 TWh
hydrogen export demand, once with zero export demand, all else identical.

  1. NET LCOH (Levelized Cost of Hydrogen)
     Net LCOH = (Export Run Total Cost - No-Export Baseline Cost) /
     Exported H2 Energy. This isolates the INCREMENTAL/MARGINAL cost of
     the export requirement specifically, by differencing two otherwise-
     identical solves. This is the same methodology used in the Egypt
     reference paper, and is the primary hydrogen-cost metric adopted for
     this study.

  2. LCOE (Levelized Cost of Electricity)
     Reported alongside Net LCOH to characterise the underlying power
     system cost basis for each scenario (i.e. the cost of serving
     Libya's domestic electricity demand within each scenario's
     technology mix and CO2 constraint), computed from the same paired
     with/without-export solves.

An energy-throughput-based "pro-rata" allocation method (attributing a
share of shared generation/fuel infrastructure cost to hydrogen in
proportion to its energy consumption, computed from a single export-run
solve without needing the paired baseline) was evaluated during model
development as a potentially faster alternative requiring only 3 runs
instead of 6. It was NOT adopted for this study: it measures an
average-cost allocation rather than the incremental/marginal cost the
Net LCOH method captures, it does not match the Egypt reference paper's
methodology, and a fuel-cost attribution issue was identified during
testing (shared natural gas cost between SMR/SMR CC and CCGT was not
being split correctly, understating SMR-heavy scenarios' pro-rata LCOH
by roughly 2.5x). The full 6-run Net LCOH/LCOE approach is the sole
basis for all reported cost results in this study.

Computation and visualization scripts: Viz_compute_h2_scenario_costs.py
(and its monthly variant). Output includes stacked bar charts of
annualized system cost by technology (styled after Figure 9a/9b of the
Egypt reference paper) and LCOH/LCOE summary tables in EUR/MWh and
EUR/kg (both LHV and HHV bases: 0.03333 and 0.03941 MWh/kg respectively).


KEY LIMITATIONS AND OPEN QUESTIONS
--------------------------------------------------------------------------------
These are known, identified limitations from model development and
diagnostic testing. They should be disclosed in any external write-up
rather than treated as resolved:

- TRANSMISSION DATA COMPLETENESS: Libya's transmission network is derived
  from OpenStreetMap, which has real, confirmed gaps for Libya specifically
  (at least one fully islanded bus with zero line connections was found
  during electricity-only model testing). Line capacity expansion in this
  framework can widen existing corridors but cannot invent new ones where
  OSM data shows none — genuine grid connections that exist in reality but
  are unmapped in OSM will not appear in the model.

- UNDERGROUND HYDROGEN STORAGE POTENTIAL: The model's underground H2
  storage capacity limits for Libya, while technically finite in the
  input data, are large enough relative to what the optimizer actually
  uses (well under 1% utilization at most nodes in testing) that they are
  functionally unconstrained. No independently verified geological survey
  of Libyan salt cavern or depleted-reservoir hydrogen storage potential
  is known to back these figures. Results that depend heavily on cheap,
  abundant underground storage should be treated as an optimistic case
  pending better geological data.

- WIND VS. SOLAR ECONOMICS: Onshore wind is available with an
  effectively unconstrained resource potential in the input data but is
  consistently not built by the optimizer, because its capital cost
  (roughly 2.6x solar's, per 2030 technology-cost assumptions) outweighs
  its capacity-factor advantage. This is a legitimate model outcome given
  standard cost assumptions, not a data or configuration error, but it
  means the current results describe a solar-dominated system by
  economics, not by Libya lacking wind resource.

- COST YEAR SENSITIVITY: All results use 2030 technology costs. Given
  the anticipated capital cost decline for electrolyzers and storage over
  the 2030s, results should be understood as a near-term snapshot, not a
  long-run projection; comparison against 2040/2050 cost assumptions is a
  recommended sensitivity check not yet performed as of this writing.

- SPATIAL RESOLUTION: 12 clusters is a reduction from earlier 36/40-node
  electricity-only testing, made to keep the sector-coupled optimization
  computationally tractable within available memory (16 GB solver
  allocation). Results at higher spatial resolution have not yet been
  validated for consistency with this coarser resolution.

- SNAPSHOT WEIGHTING METHODOLOGY: Some earlier test runs used a single
  representative month (August 2013) rather than the full year, with
  snapshot weightings scaled to represent annual totals. Where this
  approach was used, results should be understood as an approximation
  based on one month's weather/demand pattern, not the full year's actual
  seasonal variation.

- METHODOLOGY CHOICE (Net LCOH vs. pro-rata allocation): as noted above,
  an energy-throughput pro-rata allocation method was evaluated during
  development but not adopted, partly because a fuel-cost attribution
  bug was found in it during testing. Since all reported results in this
  study use the full 6-run Net LCOH/LCOE method instead, that specific
  issue does not affect any figures reported here — it is documented
  only for completeness, as part of this project's development record.


REPRODUCING THIS STUDY
--------------------------------------------------------------------------------
1. Clone/fork pypsa-meets-earth/pypsa-earth as the base repository.
2. Replace config.yaml with the desired scenario config
   (config_scenario_{a,b,c}.yaml — see attached files).
3. Run the paired zero-export config for the same scenario (h2export:
   [0], all else identical) — required for Net LCOH/LCOE, the sole
   adopted cost metric for this study.
4. Run: snakemake --cores <N> solve_sector_networks
5. Once solved, run Viz_compute_h2_scenario_costs.py against the
   resulting postnetwork .nc file(s) to generate cost breakdowns, LCOH
   tables, and the Figure-9-style comparison charts.

================================================================================
Document generated as part of PyPSA-Libya project development.
================================================================================
