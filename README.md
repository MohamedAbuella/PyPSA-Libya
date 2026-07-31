# PyPSA-Libya

An open, high-resolution model of the Libyan electricity system, built on
[PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth), following the
national-fork methodology introduced by
[PyPSA-Spain](https://doi.org/10.1016/j.esr.2025.101764)
(Gallego-Castillo & Victoria, *Energy Strategy Reviews*, 2025).

## Why a national fork instead of PyPSA-Africa / PyPSA-Earth directly?

A single-country model allows finer spatial and temporal resolution for the
same compute budget than a continental model, and lets national datasets
(GECOL capacity data, historical generation, national cost assumptions)
override the continental defaults — the same rationale PyPSA-Spain gives for
forking PyPSA-Eur instead of using it directly on Spain alone.

## Status

🚧 Early scaffold — configuration and data-collection stage, not yet a
validated model. See `REPO_STRUCTURE.md` for the repository architecture and
what is/isn't tracked in git.

## Quickstart

```bash
mamba env create -f envs/environment.yaml
conda activate pypsa-earth
snakemake --configfile config.yaml config.libya.yaml -j4 solve_all_networks
```

## Data availability

Large regenerable artifacts (weather cutouts, solved network `.nc` files,
full result sets) are **not** stored in this repository. Pre-computed inputs
and published results are archived at:

> [ADD Zenodo/OSF DOI link here once published]

## Known limitations vs. PyPSA-Spain

Libya lacks equivalents of Spain's Datadis (hourly demand by region) and
ESIOS (capacity by region) APIs, so the renewable-profile correction and
demand-disaggregation methods from PyPSA-Spain are not yet transferable
as-is — see `REPO_STRUCTURE.md` §7 for details and what data would be needed
to close that gap.

## Citing

If you use this model, please cite this repository and the methodological
basis:

```
Gallego-Castillo, C., Victoria, M., 2025. PyPSA-Spain: An extension of
PyPSA-Eur to model the Spanish energy system. Energy Strategy Reviews 60,
101764. https://doi.org/10.1016/j.esr.2025.101764
```

## License

[Match the upstream PyPSA-Earth license — MIT/AGPL depending on component,
check LICENSES/ in the fork before publishing.]
