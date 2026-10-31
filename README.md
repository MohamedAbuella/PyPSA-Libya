# PyPSA-Libya: Simulation Results & Pathways

[![Live Dashboard Preview](https://img.shields.io/badge/📊_Explore_Interactive_Dashboard-Click_Here-blue?style=for-the-badge)](https://MohamedAbuella.github.io/PyPSA-Libya/)

The complete model code, input data time series, scenario configuration scripts, and optimization routines are publicly available in this repository. This project provides an open energy system model of the Libyan electricity system built on top of the global [PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth) framework.

---

## Why a national fork instead of PyPSA-Africa / PyPSA-Earth directly?

A single-country model allows finer spatial and temporal resolution for the
same compute budget than a continental model, and lets national datasets
(GECOL capacity data, historical generation, national cost assumptions)
override the continental defaults.

## Status

🟢 **Active / Pre-Print Release** — A fully operational, high-resolution energy system model of Libya's power and hydrogen export pathways. The complete analysis, simulation results, and methodology are detailed in our preprint. See `REPO_STRUCTURE.md` for repository architecture and data tracking details.

## Quickstart

```bash
mamba env create -f envs/environment.yaml
conda activate pypsa-earth
snakemake --configfile config.yaml config.libya.yaml -j4 solve_all_networks
```

## Data availability

The input data time series, custom demand files, network topologies, scenario configuration scripts, and optimization routines are fully tracked and available directly within this repository (see `data/`, `configs/`, and `scripts/`). Pre-computed result figures and summary reports can be found in the `Libya_H2/` folder.

## Limitations

Because high-resolution regional APIs (such as hourly subnational demand and regional capacity tracking) are unavailable for Libya, standard regional demand-disaggregation and profile-correction workflows cannot be fully automated out-of-the-box. Instead, the model relies on custom national datasets and proxy profiles (see `data/`). See `REPO_STRUCTURE.md` for full architectural details and the data required to expand regional granularity.

## Citing

If you use the PyPSA-Libya model, data, or findings in your research, please cite our preprint:

> Abuella, M. (2026). PyPSA-Libya: An Open-Source Energy System Model for National Decarbonization and Hydrogen Export Pathways. arXiv:2610.38094 [cs.CE]. https://doi.org/10.48550/arXiv.2610.38094

### BibTeX

```bibtex
@misc{abuella2026pypsalibya,
  author    = {Abuella, Mohamed},
  title     = {PyPSA-Libya: An Open-Source Energy System Model for National Decarbonization and Hydrogen Export Pathways},
  year      = {2026},
  eprint    = {2610.38094},
  archivePrefix = {arXiv},
  primaryClass = {cs.CE},
  doi       = {10.48550/arXiv.2610.38094},
  url       = {https://arxiv.org/abs/2610.38094}
}
```

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0-or-later), matching the upstream [PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth) codebase alignment.

See the [`LICENSE`](./LICENSE) file for the full legal text.
