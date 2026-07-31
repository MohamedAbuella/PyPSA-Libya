# PyPSA-Libya — Repository Architecture

## 1. Base repo

Fork **pypsa-meets-earth/pypsa-earth** (not pypsa-eur). Reasons:

| | PyPSA-Eur | PyPSA-Earth |
|---|---|---|
| Grid source | ENTSO-E static grid extract | OpenStreetMap (any country) |
| Land-use/renewable data | Copernicus (EU only) | Global (Copernicus Global, WDPA, GEBCO) |
| Demand proxy | Eurostat GDP/pop | GDP/pop rasters, global |
| Libya coverage | None | Yes, out of the box |

PyPSA-Spain is your **reference for methodology and paper structure only** — it
is a fork of PyPSA-Eur and its data connectors won't resolve Libya. You'll
reuse three ideas from it directly:
1. the nested-model pattern (national high-res model bounded by a coarser
   regional/continental one — for you: PyPSA-Libya nested inside PyPSA-Earth's
   North-Africa cutout instead of PyPSA-Eur)
2. national cost/parameter overrides via a single `data_LY/costs/costs_updates.csv`
3. the `/paper` reproducibility structure (see §4)

## 2. Directory layout after forking

```
pypsa-libya/
├── config.yaml                  # <- edit: countries: ["LY"]
├── config.default.yaml          # upstream default, leave untouched
├── Snakefile                    # upstream workflow DAG, leave untouched
├── envs/
│   └── environment.yaml         # conda/mamba env, commit as-is unless you add deps
├── scripts/                     # upstream Python scripts — DO NOT bulk-edit
│   └── ...                      # add new scripts here for LY-specific steps
├── data/                        # small, versionable inputs you author yourself
│   └── data_LY/
│       ├── costs/costs_updates.csv
│       ├── demand/ly_nuts_equivalent_demand.csv
│       └── capacity/ly_installed_capacity_2024.csv
├── data_LY/                     # (alt. convention, mirrors PyPSA-Spain's data_ES/)
├── cutouts/                     # atlite weather cutouts — GITIGNORE (large, regen.)
├── resources/                   # intermediate pipeline outputs — GITIGNORE
├── networks/                    # solved/unsolved PyPSA network .nc files — GITIGNORE
├── results/                     # figures, tables, summary .csv — GITIGNORE bulk,
│                                 #   commit only small final summary tables if desired
├── logs/, benchmarks/           # GITIGNORE
├── test/                        # upstream CI tests
├── doc/                         # upstream Sphinx docs
├── paper/                       # NEW — LaTeX dissemination (see §4)
│   ├── elsarticle.cls
│   ├── paper.tex
│   ├── figures/                 # exported by scripts/export_figures.py
│   └── Makefile
├── .gitignore
└── README.md
```

## 3. What to commit vs. ignore

**Commit (small, deterministic, human-authored):**
- `config.yaml`, any `config.*.yaml` scenario files
- everything under `data/data_LY/` — these are hand-curated national datasets
  (TSO capacity by region, demand shares, cost overrides), analogous to
  PyPSA-Spain's `data_ES/`
- `scripts/*.py` you add or modify
- `envs/environment.yaml` if you pin/add packages
- `paper/` LaTeX source and `paper/figures/*.pdf` (final, referenced figures only)
- `README.md`, `LICENSE`, `CITATION.cff`

**Ignore (large, regenerable, or environment-specific):**
- `cutouts/*.nc` — ERA5/weather cutouts, tens of GB, fully regenerable from config
- `resources/**` — intermediate Snakemake outputs
- `networks/**`, `results/networks/*.nc` — solved network files (can be 100 MB–several GB each)
- `.snakemake/`, `logs/`, `benchmarks/`
- local solver license files, `.env`

## 4. `/paper` — research-as-code

Following the PyPSA-Spain structural pattern (Elsevier *Energy Strategy
Reviews*, `elsarticle` class):

- `paper/elsarticle.cls` — obtain from CTAN (https://ctan.org/pkg/elsarticle);
  it's an Elsevier-distributed class file, not something to author yourself,
  and it should be **committed** so the paper builds without internet access.
- `paper/paper.tex` — skeleton provided below
- `paper/figures/` — never hand-place figures here; write a
  `scripts/export_figures.py` (or notebook, mirroring pypsa-Xplore in the
  PyPSA-Spain repo) that reads `networks/*.nc` and writes final PDFs/PNGs
  straight into `paper/figures/`, so every figure in the paper is
  regenerable from a committed script + a network file, not hand-edited.
- `paper/Makefile` — `make figures && make paper` reproduces the PDF end to end.

## 5. External storage for large files

Large regenerable artifacts (`cutouts/`, `networks/`, full `results/`) do not
belong in git. Options, cheapest/simplest first:
- **Zenodo** — free, gives a DOI, good for "final results as published in the
  paper" snapshots. Link it in the README badge.
- **OSF** — similar, more common in some academic fields.
- **S3 / cloud bucket** — better if you expect frequent re-uploads during
  active development; costs money at scale.

Put the DOI/link in `README.md` under "Data availability," exactly as the
PyPSA-Spain paper does in its "Code availability" section.

## 6. Recommended git workflow

```bash
# 1. Fork pypsa-meets-earth/pypsa-earth on GitHub, then:
git clone https://github.com/<you>/pypsa-earth.git pypsa-libya
cd pypsa-libya
git remote rename origin fork
git remote add upstream https://github.com/pypsa-meets-earth/pypsa-earth.git
git checkout -b main-libya

# 2. Configure
#   edit config.yaml -> countries: ["LY"]
git add config.yaml
git commit -m "Configure PyPSA-Earth for Libya (countries: [LY])"

# 3. Add Libya-specific data + scripts as you build them
git add data/data_LY/ scripts/ envs/environment.yaml
git commit -m "Add Libyan cost overrides, capacity and demand datasets"

# 4. Never commit large outputs — verify before every commit
git status --short | grep -E '^\?\?' | grep -E 'cutouts/|resources/|networks/'
#   (should return nothing once .gitignore is in place)

# 5. Push to your own repo (rename it pypsa-libya on GitHub first)
git push fork main-libya

# 6. Periodically rebase on upstream to pick up PyPSA-Earth bug fixes
git fetch upstream
git rebase upstream/main
```

## 7. Key differences you'll need vs. PyPSA-Spain's four functionalities

Libya has essentially none of Spain's data richness (no Datadis-equivalent
hourly demand-by-region, no ESIOS-equivalent capacity-by-region API), so the
paper's Sections 2.1–2.2 (Q2Q renewable correction, spatio-temporal demand
disaggregation) will likely need to be **simplified or skipped** unless you
can source historical generation data (e.g. from GECOL, the General
Electricity Company of Libya, or ENTSOE-adjacent Mediterranean interconnection
studies). What *will* transfer directly:
- Section 2.3's interconnection modelling — Libya's real interconnections
  (Tunisia, Egypt, and the long-discussed but unbuilt Italy HVDC link) map
  cleanly onto the border-node + virtual-generator pattern.
- Section 2.4.3's cost override mechanism — directly reusable via
  `data_LY/costs/costs_updates.csv`.
