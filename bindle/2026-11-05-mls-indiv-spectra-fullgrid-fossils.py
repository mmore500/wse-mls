# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.23.2",
#     "matplotlib==3.10.9",
#     "numpy==2.1.2",
#     "pandas==2.2.3",
#     "phyloframe[jit] @ git+https://github.com/mmore500/phyloframe.git@890559b39e4c1439ee5828b1c70d151cad402159",
#     "polars==1.36.1",
#     "pyarrow==16.1.0",
#     "requests==2.33.1",
#     "scipy==1.14.1",
#     "seaborn==0.13.2",
#     "teeplot==1.4.2",
#     "watermark==2.4.3",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import pathlib
    import re
    import time
    import warnings

    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from phyloframe import legacy as pfl
    import polars as pl
    from scipy import stats as scipy_stats
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    from pylib import osf_fetch

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*experimental.*")
    # Computer Modern math, e.g., for italic site class qualifiers
    plt.rcParams["mathtext.fontset"] = "cm"

    # samples of the same run's fossils, by OSF slug: uniform within
    # (layer, subgrid) strata, and from a subset of spatial k-d tree groups
    SAMPLES = {"uniform": "68tgv", "kdtree": "gqt5d"}
    NUM_SITES = 16
    ALPHA = 0.05
    # minimum gap enforced between a node's origin time and its parent's
    ORIGIN_TIME_EPSILON = 0.5
    NUM_BOOTSTRAP = 0
    SEED = 1
    # dark, middle, and light shades per site class; plots use the middle
    ETI_PALETTE = {
        "MLS": ["#004C7A", "#0072B2", "#8FC3E6"],
        "BENwg": ["#00664F", "#009E73", "#8ED6BE"],
        "DELbg": ["#9A3F00", "#D55E00", "#F2B48A"],
        "neutral": ["#6E6E6E", "#A6A6A6", "#DADADA"],
    }


@app.function
def gap_points(*artists):
    """Draw point-plot markers as gaps in their connecting lines, layered as:
    connecting line, white gap, error bar, white halo, then a small dot
    marking each position."""
    for artist in artists:
        if artist is None:
            continue
        for line in [*artist.get_lines()]:
            if line.get_marker() in ("None", None, ""):
                line.set_zorder(3)  # error bars (and reference lines)
                continue
            _is_axes = hasattr(artist, "plot")
            if not _is_axes:
                # legend entries: just a dot
                line.set_linestyle("none")
                line.set_markerfacecolor(line.get_color())
                line.set_markersize(7)
                line.set_markeredgewidth(0)
                continue
            # connecting line below a white gap
            line.set_zorder(1)
            if not len(line.get_xdata()):
                continue
            _xy = line.get_xdata(), line.get_ydata()
            artist.plot(
                *_xy,
                linestyle="none",
                marker="o",
                markersize=9,
                markeredgewidth=0,
                color="white",
                zorder=2,
            )
            line.set_marker("None")
            # white halo, then small dot, above the error bar
            for _size, _color, _z in (
                (3.6, "white", 4),
                (2.2, line.get_color(), 5),
            ):
                artist.plot(
                    *_xy,
                    linestyle="none",
                    marker="o",
                    markersize=_size,
                    markeredgewidth=0,
                    color=_color,
                    zorder=_z,
                )


@app.function
def effect_pct(effect):
    """Express an effect size as a percent label, passing "neu" through."""
    return (
        effect
        if effect in ("neu", "neutral")
        else f"{float(effect) * 100:.1f}"
    )


@app.function
def right_legend(g):
    """Replace a FacetGrid's legend with a single-column legend to the right
    of the panels, vertically centered on them."""
    _handles = [*g.legend.legend_handles]
    _labels = [_text.get_text() for _text in g.legend.get_texts()]
    g.legend.remove()
    g._legend = None
    g.tight_layout()
    _boxes = [_ax.get_position() for _ax in g.axes.flat]
    return g.figure.legend(
        _handles,
        _labels,
        loc="center left",
        bbox_to_anchor=(
            max(_b.x1 for _b in _boxes) + 0.02,
            (min(_b.y0 for _b in _boxes) + max(_b.y1 for _b in _boxes)) / 2,
        ),
        ncol=1,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.5,
        labelspacing=1.0,
    )


@app.function
def corner_xlabel(g, legend, text="Effect\nSize (%)"):
    """Place the x axis label at bottom right, left-aligned under the legend
    and level with the bottom of the bottom-left panel's tick labels."""
    _renderer = g.figure.canvas.get_renderer()
    _to_fig = g.figure.transFigure.inverted()
    _axes = g.axes.flat
    _ticks = [
        _t.get_window_extent(_renderer).transformed(_to_fig)
        for _t in _axes[len(_axes) - 1].get_xticklabels()
    ]
    g.figure.text(
        legend.get_window_extent(_renderer).transformed(_to_fig).x0,
        min(_b.y0 for _b in _ticks),
        text,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"],
    )


@app.function
def tilt_xticklabels(g):
    """Rotate x tick labels vertical, anchored right on their ticks."""
    for _a in g.axes.flat:
        plt.setp(
            _a.get_xticklabels(),
            rotation=90,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )


@app.cell(hide_code=True)
def do_watermark():
    mo.md(f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """)
    return


@app.cell(hide_code=True)
def describe_notebook():
    mo.md(f"""
    # MLS Individual-host Spectra, Full-grid Fossils: Uniform vs. k-d Tree Group Sampling

    Compares two samples of fossil genomes from one full-grid run, to ask whether sampling densely within local spatial clusters (k-d tree groups) yields more information about trait gain rates than sampling uniformly, and whether that improves power to detect a trait's gain rate differing from neutral sites'.

    Data were generated by [`wse-async-ga` `wsclust/2026-11-05-mls-indiv-spectra-fullgrid-fossils-kdtree`](https://github.com/mmore500/wse-async-ga/tree/290f36b2fe870864c2bcc54ab2e6f26583577e90/wsclust/2026-11-05-mls-indiv-spectra-fullgrid-fossils-kdtree) (in [mmore500/wse-async-ga#100](https://github.com/mmore500/wse-async-ga/pull/100)), with genome flavor `genome_mlsindivspectra_2026_11_03` and seed 1105, and are hosted at {", ".join(f"<https://osf.io/{slug}> ({sample})" for sample, slug in SAMPLES.items())}.
    Both files are no-collapse minified phylogenies, so every sampled fossil genome is its own leaf, reconstructed with hstrat v1.27.0's empirical-Bayes origin times:

    - **uniform** (`05-minify-phylo`): 4M fossil genomes sampled uniformly within (fossil layer, subgrid) strata, spread over the whole grid; and
    - **kdtree** (`07-minify-phylo-kdtree`): fossils partitioned into 1,000 spatially compact groups by k-d tree over PE row and column; groups taken in random order (at least 50, and more until 4M genomes are available), then 4M genomes sampled uniformly from the chosen groups.

    So, the kdtree sample has about as many genomes as the uniform one, packed into a few patches of the grid.
    Denser sampling within patches should resolve more closely related genomes, so more gains should be seen on short branches and within multi-leaf clades, but the two samples see different parts of the run.

    The genome carries three spectra of four sites each, with base effect sizes $s_i \\in [0.25, 0.05, 0.025, 0.01]$, plus four neutral sites:

    - **MLS** sites (`MLS`) both are favored within a host and reduce transmission,
    - **between-group deleterious** sites (`DELbg`) only reduce transmission, and
    - **within-group beneficial** sites (`BENwg`) are only favored within a host.

    Each population member is a host, and each sample's trait gains are fit with `alifestd_tgs_allmdc_estimate_polars` (mutationally defined clades of every size) and `alifestd_tgs_pendantmdc_estimate_polars` (pendant branches only) from [mmore500/phyloframe#131](https://github.com/mmore500/phyloframe/pull/131), as in `2026-11-03-mls-indiv-spectra-fullgrid-fossils`.
    Times are in `origin_time` units (two per generation).
    """)
    return


@app.cell(hide_code=True)
def delimit_get_data():
    mo.md("""
    ## Get Data
    """)
    return


@app.cell
def load_config():
    def _config_cols(names):
        return [
            name
            for name in names
            if name.startswith(("ASYNC_GA_", "CEREBRASLIB_", "dstream_algo"))
            or name.startswith(("mls_", "deleterious_", "beneficial_"))
            or name
            in (
                "config",
                "genomeFlavor",
                "globalSeed",
                "nCol",
                "nRow",
                "nColSubgrid",
                "nRowSubgrid",
                "nCycle",
                "tilePopSize",
                "replicate",
                "trait_focal",
            )
        ]

    _rows = {}
    for _sample, _slug in SAMPLES.items():
        _lazy = pl.scan_parquet(osf_fetch(_slug)).filter(
            pl.col("clade_size") > 0
        )
        assert _lazy.select(pl.col("config").n_unique()).collect().item() == 1
        _rows[_sample] = (
            _lazy.select(_config_cols(_lazy.collect_schema().names()))
            .head(1)
            .collect()
        )
    _first, *_others = _rows.values()
    # both samples are of one run
    for _other in _others:
        assert _first.equals(_other.select(_first.columns)), "runs differ"
    config_df = _first
    assert config_df["nColSubgrid"].item() == 0, "expected full grid"
    config_df.transpose(include_header=True)
    return (config_df,)


@app.cell
def tabulate_effect_sizes(config_df):
    # site class K's level-i site is flag byte B, bit b (trait site 8B + b),
    # with transmission reduction (0 where a class lacks it) and within-host
    # gain and loss probabilities
    _config = config_df.row(0, named=True)
    _site_cols = [
        (int(B) * 8 + int(b), kind, what, col)
        for col in config_df.columns
        for kind, what, B, b in re.findall(
            r"^(mls|deleterious|beneficial)"
            r"_(transmission_reduction|withinhost_gain|withinhost_loss)"
            r"_p_byte(\d+)_bit(\d+)$",
            col,
        )
    ]
    effect_sizes = (
        pd.DataFrame(
            [
                {"site": site, "kind": kind, f"{what}_p": _config[col]}
                for site, kind, what, col in _site_cols
            ],
        )
        .groupby("site", as_index=False)
        .first()
    )
    _txscale = _config["CEREBRASLIB_MLS_TRANSMISSION_REDUCTION_SCALE"]
    _neutral_p = _config["mls_withinhost_neutral_p"]
    # base effect size s_i is transmission reduction over its scale; within-
    # host beneficial sites have none, so take s_i from the MLS site with the
    # same within-host gain probability
    effect_sizes["effect_size"] = (
        effect_sizes["transmission_reduction_p"] / _txscale
    ).round(6)
    _is_mls = effect_sizes["kind"] == "mls"
    _gain_to_s = dict(
        zip(
            effect_sizes.loc[_is_mls, "withinhost_gain_p"],
            effect_sizes.loc[_is_mls, "effect_size"],
        ),
    )
    _is_ben = effect_sizes["kind"] == "beneficial"
    effect_sizes.loc[_is_ben, "effect_size"] = effect_sizes.loc[
        _is_ben, "withinhost_gain_p"
    ].map(_gain_to_s)
    assert effect_sizes["effect_size"].notna().all()
    # configured within-host gain probability relative to neutral sites'
    effect_sizes["configured_gain_ratio"] = (
        effect_sizes["withinhost_gain_p"] / _neutral_p
    )
    focal_sites = sorted(effect_sizes["site"].unique().tolist())
    _focal_mask = sum(1 << site for site in focal_sites)
    assert config_df["trait_focal"].item() == _focal_mask, "focal mismatch"
    neutral_sites = [
        site for site in range(NUM_SITES) if site not in focal_sites
    ]

    # ETI evaluation bar notation: between-group effect above, within-group
    # effect below, with null (crossed circle) for no effect
    kind_labels = {
        "MLS": r"MLS$|^{-}_{+}$",
        "DELbg": r"DEL$\mathit{bg}|^{-}_{\varnothing}$",
        "BENwg": r"BEN$\mathit{wg}|^{\varnothing}_{+}$",
        "NEU": r"NEU$|^{\varnothing}_{\varnothing}$",
    }
    _kind_of = {
        "mls": kind_labels["MLS"],
        "deleterious": kind_labels["DELbg"],
        "beneficial": kind_labels["BENwg"],
    }
    _info = effect_sizes.set_index("site")
    site_info = pd.DataFrame(
        {
            "site": range(NUM_SITES),
            "trait": [f"site{site}" for site in range(NUM_SITES)],
            "kind": [
                (
                    _kind_of[_info.loc[site, "kind"]]
                    if site in focal_sites
                    else kind_labels["NEU"]
                )
                for site in range(NUM_SITES)
            ],
            "effect_size": [
                (
                    effect_pct(_info.loc[site, "effect_size"])
                    if site in focal_sites
                    else "neu"
                )
                for site in range(NUM_SITES)
            ],
            "configured_gain_ratio": [
                (
                    _info.loc[site, "configured_gain_ratio"]
                    if site in focal_sites
                    else 1.0
                )
                for site in range(NUM_SITES)
            ],
        },
    )
    kind_order = [
        kind_labels[_kind] for _kind in ("NEU", "BENwg", "DELbg", "MLS")
    ]
    kind_palette = {
        kind_labels[_kind]: ETI_PALETTE[_key][1]
        for _kind, _key in (
            ("NEU", "neutral"),
            ("BENwg", "BENwg"),
            ("DELbg", "DELbg"),
            ("MLS", "MLS"),
        )
    }
    effect_size_order = ["neu"] + sorted(
        set(site_info["effect_size"]) - {"neu"}, key=float
    )
    mo.vstack([effect_sizes, site_info])
    return (
        effect_size_order,
        kind_order,
        kind_palette,
        neutral_sites,
        site_info,
    )


@app.function
def prep_phylogeny(data_path):
    """Load a no-collapse minified phylogeny, drop layer 0, prune, collapse
    unifurcations, and enforce positive branch lengths; mark clade sizes,
    survival times, and branch lengths. Returns the phylogeny (pandas) and
    a summary."""
    # unminify: rows are in depth-first order, with id = row index
    _raw = (
        pl.read_parquet(
            data_path,
            columns=[
                "ancestor_offset",
                "origin_time",
                "clade_size",
                "trait_present",
                "layer",
                "row",
                "col",
            ],
        )
        .with_row_index("id")
        .with_columns(
            pl.col("id").cast(pl.Int64),
            ancestor_id=(
                pl.col("id").cast(pl.Int64)
                - pl.col("ancestor_offset").cast(pl.Int64)
            ),
            trait_present=pl.col("trait_present").cast(pl.Int64),
            row=pl.col("row").cast(pl.Int64),
            col=pl.col("col").cast(pl.Int64),
            layer=pl.col("layer").cast(pl.Int64),
        )
        .drop("ancestor_offset")
        .to_pandas()
    )
    _num_raw = len(_raw)
    _is_genome = _raw["clade_size"] > 0
    assert (_raw.loc[_is_genome, "clade_size"] == 1).all(), "no-collapse"
    # a missing origin time (a root's) can be no later than the run start
    _no_time = _raw["origin_time"].isna()
    assert (
        _raw.loc[_no_time, "ancestor_id"] == _raw.loc[_no_time, "id"]
    ).all()
    _raw["origin_time"] = _raw["origin_time"].fillna(-1).astype(float)
    phylo = (
        _raw.drop(columns=["clade_size"])
        .assign(extant=_is_genome & (_raw["origin_time"] >= 0))
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_prune_extinct_lineages_asexual, mutate=True)
        .pipe(pfl.alifestd_collapse_unifurcations, mutate=True)
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_mark_leaves, mutate=True)
        .reset_index(drop=True)
    )
    del _raw
    assert (phylo["is_leaf"] == phylo["extant"]).all()
    phylo = phylo.drop(columns=["extant"])
    assert (phylo["id"] == np.arange(len(phylo))).all()
    _before = phylo["origin_time"].to_numpy().copy()
    phylo = pfl.alifestd_coerce_chronological_consistency(
        phylo, mutate=True, epsilon=ORIGIN_TIME_EPSILON
    )
    _after = phylo["origin_time"].to_numpy()
    phylo["num_leaves"] = phylo.pipe(
        pfl.alifestd_mark_num_leaves_asexual, mutate=True
    )["num_leaves"]
    phylo["max_descendant_origin_time"] = phylo.pipe(
        pfl.alifestd_mark_max_descendant_origin_time_asexual, mutate=True
    )["max_descendant_origin_time"]
    _anc = phylo["ancestor_id"].to_numpy()
    phylo["is_root"] = _anc == phylo["id"].to_numpy()
    phylo["branch_length"] = np.where(
        phylo["is_root"], np.nan, _after - _after[_anc]
    )
    phylo["survival_time"] = phylo["max_descendant_origin_time"] - _after
    summary = {
        "num_nodes_raw": _num_raw,
        "num_nodes": len(phylo),
        "num_leaves": int(phylo["is_leaf"].sum()),
        "num_inner": int((~phylo["is_leaf"]).sum()),
        "num_roots": int(phylo["is_root"].sum()),
        "num_origin_times_raised": int((_after != _before).sum()),
    }
    return phylo, summary


@app.function
def add_trait_transitions(phylo):
    """Add each site's state transition on the branch into each node, from
    leaf states under parsimony without loss (a node has a trait if and
    only if every leaf of its clade does). Returns the column names."""
    _is_leaf = phylo["is_leaf"].to_numpy()
    _anc = phylo["ancestor_id"].to_numpy()
    _is_root = phylo["is_root"].to_numpy()
    _num_leaves = phylo["num_leaves"].to_numpy()
    _present = phylo["trait_present"].fillna(0).to_numpy(dtype=np.int64)
    names = []
    for _site in range(NUM_SITES):
        _has = _is_leaf & (((_present >> _site) & 1) != 0)
        _count = (
            phylo[["id", "ancestor_id"]]
            .copy()
            .pipe(
                pfl.alifestd_calc_clade_trait_count_asexual,
                mutate=True,
                trait_mask=_has,
            )
        )
        _state = (_count == _num_leaves).astype(np.int8)
        _name = f"site{_site}"
        phylo[_name] = _state - np.where(_is_root, 0, _state[_anc])
        assert phylo[_name].isin([0, 1]).all(), "parsimony without loss"
        names.append(_name)
    return names


@app.function
def measure_sampling(phylo):
    """Per-PE sampled genome counts, pendant branch lengths, and the
    spatial distance between leaves of cherries (two-leaf clades)."""
    # inner nodes have no PE, so row and col are float (with nan); leaves'
    # are whole
    _leaves = phylo.loc[phylo["is_leaf"]].astype({"row": int, "col": int})
    pe_counts = (
        _leaves.groupby(["row", "col"])
        .size()
        .rename("num_genomes")
        .reset_index()
    )
    pendant_lengths = _leaves["branch_length"].to_numpy(dtype=np.float32)
    # cherries: inner nodes whose two children are both leaves
    _anc = phylo["ancestor_id"].to_numpy()
    _child = ~phylo["is_root"].to_numpy()
    _num_children = np.bincount(_anc[_child], minlength=len(phylo))
    _leaf_children = np.bincount(
        _anc[_child & phylo["is_leaf"].to_numpy()], minlength=len(phylo)
    )
    _is_cherry = (_num_children == 2) & (_leaf_children == 2)
    _cherry_leaves = _leaves[_is_cherry[_leaves["ancestor_id"].to_numpy()]]
    _pairs = _cherry_leaves.sort_values("ancestor_id").groupby("ancestor_id")
    _first, _last = _pairs.first(), _pairs.last()
    cherry_distances = np.hypot(
        (_first["row"] - _last["row"]).to_numpy(),
        (_first["col"] - _last["col"]).to_numpy(),
    ).astype(np.float32)
    summary = {
        "num_pes_sampled": len(pe_counts),
        "genomes_per_sampled_pe": pe_counts["num_genomes"].mean(),
        "pendant_length_median": float(np.median(pendant_lengths)),
        "num_cherries": int(_is_cherry.sum()),
        "leaves_in_cherries_frac": 2 * _is_cherry.sum() / len(_leaves),
        "cherry_distance_median": float(np.median(cherry_distances)),
        "cherry_same_pe_frac": float((cherry_distances == 0).mean()),
    }
    return pe_counts, pendant_lengths, cherry_distances, summary


@app.function
def tabulate_sample_mdcs(phylo, trait_names):
    """One row per MDC (gain on a non-root branch), with its stem branch
    length, survival time, and size (leaves)."""
    _is_root = phylo["is_root"].to_numpy()
    _cols = ["origin_time", "branch_length", "survival_time", "num_leaves"]
    return pd.concat(
        [
            phylo.loc[
                (phylo[_name].to_numpy() == 1) & ~_is_root, _cols
            ].assign(trait=_name)
            for _name in trait_names
        ],
        ignore_index=True,
    )


@app.function
def fit_sample_estimates(phylo, trait_names, neutral_traits, batches):
    """Fit allmdc and pendant estimators, raw and calibrated, per batch of
    focal traits together with the neutral traits (to bound memory);
    neutral traits' fits must agree across batches."""
    _df = pl.from_pandas(
        phylo[["id", "ancestor_id", "origin_time", *trait_names]],
    )
    _results = []
    for _method, _fn in (
        ("allmdc", pfl.alifestd_tgs_allmdc_estimate_polars),
        ("pendantmdc", pfl.alifestd_tgs_pendantmdc_estimate_polars),
    ):
        for _calibrate in (False, True):
            _start = time.time()
            _fits = [
                _fn(
                    _df,
                    trait_transitions=[*_batch, *neutral_traits],
                    neutral_traits=neutral_traits,
                    calibrate=_calibrate,
                    num_bootstrap=NUM_BOOTSTRAP,
                    seed=SEED,
                ).to_pandas()
                for _batch in batches
            ]
            _neutral_fits = [
                _fit.set_index("trait").loc[
                    neutral_traits, ["gain_rate", "selection"]
                ]
                for _fit in _fits
            ]
            for _other in _neutral_fits[1:]:
                assert np.allclose(_neutral_fits[0], _other, rtol=1e-6)
            print(
                f"  {_method} calibrate={_calibrate}: "
                f"{time.time() - _start:.0f}s",
            )
            _results.append(
                pd.concat(
                    [
                        _fits[0][_fits[0]["trait"].isin(neutral_traits)],
                        *(
                            _fit[~_fit["trait"].isin(neutral_traits)]
                            for _fit in _fits
                        ),
                    ],
                    ignore_index=True,
                ).assign(
                    method=_method,
                    calibration="calibrated" if _calibrate else "raw",
                ),
            )
    return pd.concat(_results, ignore_index=True)


@app.cell(hide_code=True)
def delimit_analyze():
    mo.md(f"""
    ## Analyze Samples

    Each sample is analyzed in turn, then its phylogeny is discarded to bound memory:

    1. Fossil layer 0 (the initial population, at `origin_time == -1`) is dropped, lineages without retained leaves are pruned, and unifurcations collapsed.
    2. Origin times are raised to at least the parent's plus {ORIGIN_TIME_EPSILON}, so every branch has positive length.
    3. Each site's trait state at every node is reconstructed under parsimony without loss: a node has a trait if and only if every leaf of its clade does.
    4. The allmdc and pendant estimators are fit with raw and with neutral-calibrated branch lengths.
    """)
    return


@app.cell
def analyze_samples(neutral_sites, site_info):
    import gc

    _neutral = [f"site{_site}" for _site in neutral_sites]
    _batches = [
        _part["trait"].tolist()
        for _kind, _part in site_info.groupby("kind", sort=False)
        if not _kind.startswith("NEU")
    ]
    _summaries, _pes, _pendants, _cherries, _mdcs, _estimates = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for _sample, _slug in SAMPLES.items():
        print(f"{_sample} ({_slug})")
        _start = time.time()
        _phylo, _summary = prep_phylogeny(osf_fetch(_slug))
        _names = add_trait_transitions(_phylo)
        _pe, _pend, _cher, _density = measure_sampling(_phylo)
        _summaries.append({"sample": _sample, **_summary, **_density})
        _pes.append(_pe.assign(sample=_sample))
        _pendants.append(
            pd.DataFrame({"branch_length": _pend, "sample": _sample})
        )
        _cherries.append(pd.DataFrame({"distance": _cher, "sample": _sample}))
        _mdcs.append(
            tabulate_sample_mdcs(_phylo, _names).assign(sample=_sample)
        )
        _estimates.append(
            fit_sample_estimates(_phylo, _names, _neutral, _batches).assign(
                sample=_sample
            )
        )
        del _phylo
        gc.collect()
        print(f"  total {time.time() - _start:.0f}s")
    sample_summary = pd.DataFrame(_summaries).set_index("sample").T
    pe_counts = pd.concat(_pes, ignore_index=True)
    pendant_lengths = pd.concat(_pendants, ignore_index=True)
    cherry_distances = pd.concat(_cherries, ignore_index=True)
    sample_mdcs = pd.concat(_mdcs, ignore_index=True).merge(
        site_info, on="trait"
    )
    sample_estimates = pd.concat(_estimates, ignore_index=True).merge(
        site_info, on="trait"
    )
    mo.Html(sample_summary.to_html(float_format="{:.4g}".format))
    return (
        cherry_distances,
        pe_counts,
        pendant_lengths,
        sample_estimates,
        sample_mdcs,
    )


@app.cell(hide_code=True)
def delimit_density():
    mo.md("""
    ## Sampling Density

    Where each sample's genomes lie on the grid, how many share a processing element (PE), and how closely related sampled genomes are: pendant branch length (time back to a leaf's nearest sampled branching point) and the spatial distance between the two leaves of each cherry (two-leaf clade).
    """)
    return


@app.cell
def plot_pe_coverage(config_df, pe_counts):
    _n_row, _n_col = int(config_df["nRow"].item()), int(
        config_df["nCol"].item()
    )
    # genomes per 15x15-PE block
    _block = 15

    def pe_coverage(data):
        _fig, _axes = plt.subplots(
            1, len(SAMPLES), figsize=(3.2 * len(SAMPLES), 3.2)
        )
        for _ax, _sample in zip(_axes, SAMPLES):
            _d = data[data["sample"] == _sample]
            _grid = np.zeros(
                (-(-_n_row // _block), -(-_n_col // _block)), dtype=float
            )
            np.add.at(
                _grid,
                (
                    (_d["row"].to_numpy() // _block),
                    (_d["col"].to_numpy() // _block),
                ),
                _d["num_genomes"].to_numpy(),
            )
            _im = _ax.imshow(
                np.where(_grid > 0, _grid, np.nan),
                cmap="viridis",
                norm=plt.matplotlib.colors.LogNorm(vmin=1),
                interpolation="nearest",
            )
            _ax.set_title(f"{_sample}")
            _ax.set_xlabel("PE Column Block")
            _ax.set_ylabel("PE Row Block")
            _fig.colorbar(
                _im,
                ax=_ax,
                shrink=0.7,
                label=f"Genomes per\n{_block}×{_block} PEs",
            )
        _fig.tight_layout()

    tp.tee(
        pe_coverage,
        data=pe_counts,
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    )
    plt.close("all")
    return


@app.cell
def plot_density_distributions(cherry_distances, pe_counts, pendant_lengths):
    _palette = dict(zip(SAMPLES, [ETI_PALETTE["neutral"][0], "#CC79A7"]))
    for _data, _x, _label, _what, _log in (
        (
            pe_counts,
            "num_genomes",
            "Genomes per\nSampled PE",
            "pe-genomes",
            (True, False),
        ),
        (
            pendant_lengths[pendant_lengths["branch_length"] > 0],
            "branch_length",
            "Pendant Branch Length",
            "pendant-lengths",
            (True, False),
        ),
        (
            cherry_distances.assign(distance=lambda df: df["distance"] + 1),
            "distance",
            "Cherry Leaves' PE\nDistance + 1",
            "cherry-distances",
            (True, False),
        ),
    ):
        with tp.teed(
            sns.displot,
            data=_data,
            x=_x,
            hue="sample",
            hue_order=[*SAMPLES],
            palette=_palette,
            kind="ecdf",
            log_scale=_log,
            height=1.8,
            aspect=1.4,
            teeplot_outattrs={"what": _what},
            teeplot_outexclude=["palette", "hue_order", "log_scale"],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            _g.set_axis_labels(_label, "Cumulative\nFraction")
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_sample_mdcs():
    mo.md("""
    ## MDC Yield

    Number of MDCs (gains observed on non-root branches), split into leaf MDCs (one leaf) and inner MDCs (several leaves), per site and sample.
    Gain rate estimates rest on these gains, so more of them should mean more power.
    """)
    return


@app.cell
def tabulate_sample_mdc_counts(sample_mdcs):
    sample_mdc_counts = (
        sample_mdcs.assign(is_inner=sample_mdcs["num_leaves"] > 1)
        .groupby(["sample", "trait", "kind", "effect_size"], sort=False)
        .agg(
            num_mdcs=("num_leaves", "size"),
            num_inner_mdcs=("is_inner", "sum"),
            mdc_size_mean=("num_leaves", "mean"),
            mdc_size_max=("num_leaves", "max"),
        )
        .reset_index()
    )
    sample_mdc_counts["num_leaf_mdcs"] = (
        sample_mdc_counts["num_mdcs"] - sample_mdc_counts["num_inner_mdcs"]
    )
    mo.Html(
        sample_mdc_counts.pivot_table(
            index=["trait", "kind", "effect_size"],
            columns="sample",
            values=["num_mdcs", "num_inner_mdcs", "mdc_size_mean"],
            sort=False,
        ).to_html(float_format="{:.3g}".format),
    )
    return (sample_mdc_counts,)


@app.cell
def plot_sample_mdc_counts(
    effect_size_order,
    kind_order,
    kind_palette,
    sample_mdc_counts,
):
    _what = {
        "num_mdcs": "all",
        "num_leaf_mdcs": "leaf",
        "num_inner_mdcs": "inner",
    }
    _df = sample_mdc_counts.melt(
        id_vars=["sample", "kind", "effect_size"],
        value_vars=[*_what],
        var_name="what",
        value_name="count",
    ).assign(
        what=lambda df: df["what"].map(_what),
        count=lambda df: df["count"].clip(lower=0.5),
    )
    with tp.teed(
        sns.catplot,
        data=_df,
        x="effect_size",
        y="count",
        order=effect_size_order,
        hue="kind",
        hue_order=kind_order,
        palette=kind_palette,
        col="what",
        col_order=[*_what.values()],
        row="sample",
        row_order=[*SAMPLES],
        kind="point",
        errorbar=("pi", 100),
        dodge=0.3,
        height=1.6,
        aspect=1.0,
        log_scale=(False, True),
        teeplot_outattrs={"what": "mdc-counts"},
        teeplot_outexclude=[
            "palette",
            "hue_order",
            "col_order",
            "row_order",
            "dodge",
            "order",
            "log_scale",
        ],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _g:
        gap_points(*_g.axes.flat, _g.legend)
        _g.set_axis_labels("", "Number of MDCs")
        _g.set_titles(template="{row_name}, {col_name} MDCs")
        tilt_xticklabels(_g)
        corner_xlabel(_g, right_legend(_g))
    plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_power():
    mo.md(f"""
    ## Gain Rate Estimates and Power

    Each site's gain ratio (gain rate relative to neutral sites') and selection, from each sample, and the evidence for gain ratio ≠ 1: the signed square root of the naive likelihood-ratio statistic, $z = \\mathrm{{sign}}(\\log \\mathrm{{gain\\ ratio}}) \\sqrt{{\\mathrm{{LRT}}}}$, which would be standard normal under the null if the likelihood were not composite.

    Nested branches share leaves, so the likelihood is composite and the naive statistic overstates the evidence, possibly by more in a densely sampled tree, which has more nested branches.
    So, each statistic is also shown scaled by its sample's mean neutral-site statistic (a rough overdispersion factor; under the null, the mean of a $\\chi^2_1$ statistic is 1): $z_\\mathrm{{adj}} = z / \\sqrt{{\\overline{{\\mathrm{{LRT}}}}_\\mathrm{{neutral}}}}$.
    The dashed lines mark $|z| = {scipy_stats.norm.isf(ALPHA / 2):.2f}$, two-sided $\\alpha = {ALPHA}$.
    """)
    return


@app.cell
def calc_power(sample_estimates):
    _df = sample_estimates.copy()
    for _stat in ("gain_ratio", "selection"):
        _sign = np.sign(
            np.log(_df["gain_ratio"]) if _stat == "gain_ratio" else _df[_stat]
        )
        _df[f"{_stat}_z"] = _sign * np.sqrt(
            _df[f"{_stat}_lrt_naive"].clip(lower=0)
        )
        _neutral = _df["kind"].str.startswith("NEU")
        _scale = (
            _df[_neutral]
            .groupby(["sample", "method", "calibration"])[f"{_stat}_lrt_naive"]
            .mean()
            .rename(f"{_stat}_neutral_lrt_mean")
        )
        _df = _df.merge(_scale, on=["sample", "method", "calibration"])
        _df[f"{_stat}_z_adj"] = _df[f"{_stat}_z"] / np.sqrt(
            _df[f"{_stat}_neutral_lrt_mean"].clip(lower=1.0)
        )
    sample_power = _df
    _crit = scipy_stats.norm.isf(ALPHA / 2)
    power_summary = (
        sample_power[~sample_power["kind"].str.startswith("NEU")]
        .assign(
            detected=lambda df: df["gain_ratio_z"].abs() > _crit,
            detected_adj=lambda df: df["gain_ratio_z_adj"].abs() > _crit,
        )
        .groupby(["method", "calibration", "sample"], sort=False)
        .agg(
            focal_detected=("detected", "sum"),
            focal_detected_adj=("detected_adj", "sum"),
            neutral_lrt_mean=("gain_ratio_neutral_lrt_mean", "first"),
            num_clade_gains=("num_clade_gains", "sum"),
            num_trait_present=("num_trait_present", "sum"),
        )
    )
    mo.vstack(
        [
            mo.Html(power_summary.to_html(float_format="{:.3g}".format)),
            mo.Html(
                sample_power[
                    [
                        "method",
                        "calibration",
                        "sample",
                        "trait",
                        "kind",
                        "effect_size",
                        "configured_gain_ratio",
                        "gain_ratio",
                        "gain_ratio_z",
                        "gain_ratio_z_adj",
                        "selection",
                        "selection_z",
                    ]
                ].to_html(float_format="{:.4g}".format),
            ),
        ],
    )
    return (sample_power,)


@app.cell
def plot_power(effect_size_order, kind_order, kind_palette, sample_power):
    _crit = scipy_stats.norm.isf(ALPHA / 2)
    _labels = {
        "gain_ratio": ("Gain Ratio", True, None),
        "gain_ratio_z": ("Gain Ratio z", False, _crit),
        "gain_ratio_z_adj": ("Gain Ratio z\n(neutral-scaled)", False, _crit),
        "selection": ("Selection", False, None),
        "selection_z": ("Selection z", False, _crit),
    }
    for (_method, _calibration), _df in sample_power.groupby(
        ["method", "calibration"], sort=False
    ):
        for _stat, (_label, _log, _ref) in _labels.items():
            with tp.teed(
                sns.catplot,
                data=_df,
                x="effect_size",
                y=_stat,
                order=effect_size_order,
                hue="kind",
                hue_order=kind_order,
                palette=kind_palette,
                col="sample",
                col_order=[*SAMPLES],
                kind="point",
                errorbar=("pi", 100),
                dodge=0.3,
                height=1.8,
                aspect=1.1,
                log_scale=(False, _log),
                teeplot_outattrs={
                    "method": _method,
                    "calibration": _calibration,
                    "stat": _stat.replace("_", "-"),
                },
                teeplot_outexclude=[
                    "palette",
                    "hue_order",
                    "col_order",
                    "dodge",
                    "order",
                    "log_scale",
                ],
                teeplot_show=True,
                teeplot_subdir=pathlib.Path(__file__).stem,
            ) as _g:
                gap_points(*_g.axes.flat, _g.legend)
                if _ref is not None:
                    for _y in (-_ref, _ref):
                        _g.refline(y=_y, color="gray", linestyle="--")
                elif _stat == "gain_ratio":
                    _g.refline(y=1, color="gray", linestyle="--")
                else:
                    _g.refline(y=0, color="gray", linestyle="--")
                _g.set_axis_labels("", _label)
                _g.set_titles(
                    col_template=f"{_method}, {_calibration}\n{{col_name}}"
                )
                tilt_xticklabels(_g)
                corner_xlabel(_g, right_legend(_g))
            plt.close("all")
    return


@app.cell
def plot_power_scatter(kind_order, kind_palette, sample_power):
    _uniform, _kdtree = [*SAMPLES]
    _wide = sample_power.pivot_table(
        index=["method", "calibration", "trait", "kind", "effect_size"],
        columns="sample",
        values=["gain_ratio_z", "gain_ratio_z_adj"],
    )
    _wide.columns = [f"{_stat} {_s}" for _stat, _s in _wide.columns]
    _wide = _wide.reset_index()
    _wide["panel"] = _wide["method"] + ", " + _wide["calibration"]
    for _stat in ("gain_ratio_z", "gain_ratio_z_adj"):
        with tp.teed(
            sns.relplot,
            data=_wide,
            x=f"{_stat} {_uniform}",
            y=f"{_stat} {_kdtree}",
            hue="kind",
            hue_order=kind_order,
            palette=kind_palette,
            col="panel",
            col_wrap=2,
            height=1.9,
            aspect=1.0,
            teeplot_outattrs={"stat": _stat.replace("_", "-")},
            teeplot_outexclude=["palette", "hue_order", "col_wrap"],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            for _ax in _g.axes.flat:
                _lo, _hi = (
                    min(_ax.get_xlim()[0], _ax.get_ylim()[0]),
                    max(_ax.get_xlim()[1], _ax.get_ylim()[1]),
                )
                _ax.plot(
                    [_lo, _hi],
                    [_lo, _hi],
                    color="gray",
                    linestyle="--",
                    zorder=0,
                )
                _ax.set_xscale("symlog", linthresh=2)
                _ax.set_yscale("symlog", linthresh=2)
            _name = "Gain Ratio z" + (
                " (scaled)" if _stat.endswith("adj") else ""
            )
            _g.set_axis_labels(
                f"{_name}\n({_uniform})", f"{_name}\n({_kdtree})"
            )
            _g.set_titles(col_template="{col_name}")
            right_legend(_g)
        plt.close("all")
    return


if __name__ == "__main__":
    app.run()
