# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.23.2",
#     "matplotlib==3.10.9",
#     "numpy==2.1.2",
#     "pandas==2.2.3",
#     "phyloframe[jit]==0.19.0",
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
    # calibration bins, per phyloframe's `alifestd_tgs_estimate_polars`
    # documented example (all branches), and for leaf (pendant) branches
    CALIBRATION_COLS = ["branch_length", "origin_time", "survival_duration"]
    LEAF_CALIBRATION_COLS = ["branch_length", "origin_time"]
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

    Each population member is a host, and each sample's trait gains are fit with `alifestd_tgs_estimate_polars` from [phyloframe 0.19.0](https://github.com/mmore500/phyloframe/releases/tag/v0.19.0), over all branches (allmdc, mutationally defined clades of every size) and over pendant branches only (pendantmdc), as in `2026-11-03-mls-indiv-spectra-fullgrid-fossils`.
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
    only if every leaf of its clade does). Roots are the reconstruction's
    placeholder ancestors, at the run start, so never carry a trait. Returns
    the column names."""
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
        _state[_is_root] = 0
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
def estimator_settings():
    """Keyword arguments to `alifestd_tgs_estimate_polars`, per method and
    branch length calibration.

    Calibration follows the estimator's documented example: all-branch
    (allmdc) fits calibrate within clade size classes, against neutral gains
    in neighborhoods of joint quantile bins of branch length, origin time,
    and survival duration; pendant (pendantmdc) fits calibrate leaf branches
    by bins of branch length and origin time.
    """
    return {
        ("allmdc", "raw"): {},
        ("allmdc", "calibrated"): dict(
            calibration_compare_by_bins="joint_quantile_bin",
            calibration_compare_by_vals=[
                f"{_col}_rank" for _col in CALIBRATION_COLS
            ],
            calibration_partition_by="clade_size_geom_class",
        ),
        ("pendantmdc", "raw"): dict(episode_mask="is_leaf"),
        ("pendantmdc", "calibrated"): dict(
            episode_mask="is_leaf",
            calibration_compare_by_bins="leaf_quantile_bin",
            calibration_compare_by_vals=[
                f"{_col}_leaf_rank" for _col in LEAF_CALIBRATION_COLS
            ],
        ),
    }


@app.function
def mark_calibration_columns(phylo, trait_names, neutral_traits):
    """The phylogeny as a polars dataframe, with the columns
    `estimator_settings` calibrates by."""
    # roots have no branch, so are not fitted
    _is_fitted = pl.col("id") != pl.col("ancestor_id")
    _is_fitted_leaf = _is_fitted & pl.col("is_leaf")
    return (
        pl.from_pandas(
            phylo[
                ["id", "ancestor_id", "origin_time", "is_leaf", *trait_names]
            ],
        )
        .pipe(
            pfl.alifestd_mark_origin_time_delta_polars,
            mark_as="branch_length",
        )
        .pipe(
            pfl.alifestd_mark_max_descendant_origin_time_polars,
            mark_as="max_descendant_origin_time",
        )
        .with_columns(
            survival_duration=pl.col("max_descendant_origin_time")
            - pl.col("origin_time"),
        )
        .pipe(
            pfl.alifestd_mark_clade_size_geom_classes_polars,
            mark_as="clade_size_geom_class",
            min_class_weight=20,
            weight_by=neutral_traits,
        )
        .pipe(
            pfl.alifestd_mark_joint_quantile_bins_polars,
            bin_by=CALIBRATION_COLS,
            mark_as="joint_quantile_bin",
            partition_by="clade_size_geom_class",
        )
        .pipe(
            pfl.alifestd_mark_joint_quantile_bins_polars,
            bin_by=LEAF_CALIBRATION_COLS,
            mark_as="leaf_quantile_bin",
            partition_by="is_leaf",
        )
        .with_columns(
            pl.when(_is_fitted)
            .then(pl.col(CALIBRATION_COLS))
            .rank()
            .name.suffix("_rank"),
            pl.when(_is_fitted_leaf)
            .then(pl.col(LEAF_CALIBRATION_COLS))
            .rank()
            .name.suffix("_leaf_rank"),
        )
    )


@app.function
def fit_estimates(calibration_df, trait_names, neutral_traits):
    """Fits every trait with each method and calibration of
    `estimator_settings`; neutral traits are fit leave-one-out."""
    _candidates = [_t for _t in trait_names if _t not in neutral_traits]
    _results = []
    for (_method, _calibration), _kwargs in estimator_settings().items():
        _start = time.time()
        _fit = pfl.alifestd_tgs_estimate_polars(
            calibration_df,
            trait_transitions_candidate=_candidates,
            trait_transitions_neutral=neutral_traits,
            **_kwargs,
        )
        print(f"{_method} {_calibration}: {time.time() - _start:.0f}s")
        _results.append(
            _fit.to_pandas().assign(method=_method, calibration=_calibration),
        )
    return pd.concat(_results, ignore_index=True).rename(
        columns={
            "trait_name": "trait",
            "num_leaf_episodes": "num_leaves",
            "num_leaf_gains": "num_trait_present",
            "num_inner_episodes": "num_clades",
            "num_inner_gains": "num_clade_gains",
            "selection_coef": "selection",
            "selection_coef_lrt_naive": "selection_lrt_naive",
        },
    )


@app.function
def compute_fit_diagnostics(
    calibration_df, trait_names, neutral_traits, num_bins=200
):
    """Observed and expected trait gains under the calibrated allmdc fit,
    with and without survival time, recomputed from the estimator's own
    exposure episodes and calibrated episode durations.

    Returns `diagnostic_fits`, each trait's fitted gain rate and selection,
    and `fit_diagnostics`, branches grouped by trait, clade size class, and
    calibrated length (`num_bins` quantile bins per trait and class).
    """
    from phyloframe.legacy import _impl as pfl_impl
    from scipy import special

    _settings = estimator_settings()[("allmdc", "calibrated")]
    _candidates = [_t for _t in trait_names if _t not in neutral_traits]
    _names = [*_candidates, *neutral_traits]
    _neutral_ids = list(range(len(_candidates), len(_names)))
    _exposure = pfl_impl.tgs_build_exposure_df_polars(
        calibration_df,
        trait_transitions=_names,
        episode_mask=None,
        **_settings,
    )
    _weights = np.ones(len(_exposure))
    _neighborhoods = pfl_impl.tgs_select_calibration_neighborhoods(
        _exposure,
        neighborhood_fractions=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        neutral_trait_ids=_neutral_ids,
    )
    _fit_kwargs = dict(
        weights=_weights,
        num_traits=len(_names),
        neutral_trait_ids=_neutral_ids,
        calibration_neighborhoods=_neighborhoods,
        nll_coarsegrain_rtol=1e-2,
        loss_rate=0.0,
        fit_selection_coef_starts=(0.0, -0.3, -1.8),
        fit_growth_exponent_bound=600.0,
        calc_lrts=False,
        progress_wrap=lambda _x: _x,
    )
    # without survival time, the same model with every survival duration 0
    _fits = {
        "with survival time": pfl_impl.tgs_fit_traits(
            _exposure, **_fit_kwargs
        ),
        "without survival time": pfl_impl.tgs_fit_traits(
            _exposure.with_columns(survival_duration=pl.lit(0.0)),
            **_fit_kwargs,
        ),
    }
    diagnostic_fits = pd.concat(
        [
            _fit.to_pandas()
            .rename(columns={"selection_coef": "selection"})
            .assign(trait=_names, fit=_label)
            for _label, _fit in _fits.items()
        ],
        ignore_index=True,
    )
    # reference gain rate: mean over neutral sites (other neutral sites,
    # for a neutral site), from the fit with survival time
    _with = diagnostic_fits[diagnostic_fits["fit"] == "with survival time"]
    _neu_rates = _with.set_index("trait").loc[neutral_traits, "gain_rate"]
    diagnostic_fits["neutral_gain_rate"] = [
        _neu_rates.drop(_t, errors="ignore").mean()
        for _t in diagnostic_fits["trait"]
    ]

    # clade size classes (calibration partitions), labeled by size range
    _partitions = _exposure["partition_id"].to_numpy()
    _clade_sizes = _exposure["clade_size"].to_numpy()
    _num_partitions = _partitions.max() + 1
    _size_lo = np.full(_num_partitions, np.iinfo(np.int64).max)
    np.minimum.at(_size_lo, _partitions, _clade_sizes)
    _size_hi = np.zeros(_num_partitions, dtype=np.int64)
    np.maximum.at(_size_hi, _partitions, _clade_sizes)
    _size_order = np.argsort(np.argsort(_size_lo))
    _labels = np.array(
        [
            f"{_lo}" if _lo == _hi else f"{_lo}–{_hi}"
            for _lo, _hi in zip(_size_lo, _size_hi)
        ],
    )

    _bin_counts = pfl_impl.tgs_count_trait_exposures_binwise(
        _exposure, trait_ids=_neutral_ids, weights=_weights
    )
    _durations = _exposure["episode_duration"].to_numpy()
    _survival = _exposure["survival_duration"].to_numpy()
    _bin_ids = _exposure["bin_id"].to_numpy()
    _diagnostics = []
    for _trait_id, _trait in enumerate(_names):
        _starts = _exposure[
            f"trait{_trait_id}_episode_starts_absent"
        ].to_numpy()
        _gained = ~_exposure[
            f"trait{_trait_id}_episode_ends_absent"
        ].to_numpy()
        _factors = pfl_impl.tgs_calc_calibration_rescaling(
            trait_bin_counts_df=_bin_counts,
            trait_id=_trait_id,
            neutral_trait_ids=_neutral_ids,
            calibration_neighborhoods=_neighborhoods,
        )
        _length = (_durations * _factors[_bin_ids])[_starts]
        _part = _partitions[_starts]
        # quantile bins of calibrated length within each size class
        _order = np.lexsort((_length, _part))
        _rank = np.empty(len(_order), dtype=np.int64)
        _rank[_order] = np.arange(len(_order))
        _part_start = np.searchsorted(_part[_order], _part)
        _part_count = np.bincount(_part, minlength=_num_partitions)[_part]
        _fine = (_rank - _part_start) * num_bins // _part_count
        _key = _part * num_bins + _fine
        _num_keys = _num_partitions * num_bins
        _columns = {
            "num_trials": np.bincount(_key, minlength=_num_keys),
            "num_successes": np.bincount(
                _key, _gained[_starts].astype(float), minlength=_num_keys
            ),
            "length": np.bincount(_key, _length, minlength=_num_keys),
        }
        for _label, _fit in _fits.items():
            _g = _fit["gain_rate"][_trait_id]
            _c = _fit["selection_coef"][_trait_id]
            if not _g > 0:
                _c = 0.0
            _effective = _length * special.exprel(
                np.minimum(_c * _length, 600.0)
            )
            _growth = np.exp(
                np.minimum(
                    _c
                    * (
                        _survival[_starts]
                        if _label == "with survival time"
                        else 0.0
                    ),
                    600.0,
                ),
            )
            _columns[f"expected {_label}"] = np.bincount(
                _key, _g * _effective * _growth, minlength=_num_keys
            )
        _df = pd.DataFrame(_columns).assign(
            trait=_trait,
            size_partition=np.repeat(_size_order, num_bins),
            clade_size=np.repeat(_labels, num_bins),
        )
        _df = _df[_df["num_trials"] > 0].copy()
        _df["length"] /= _df["num_trials"]
        _diagnostics.append(_df)
    fit_diagnostics = pd.concat(_diagnostics, ignore_index=True)
    return diagnostic_fits, fit_diagnostics


@app.function
def poisson_interval(count, alpha=0.05):
    """Exact (Garwood) Poisson confidence interval for an observed count."""
    count = np.asarray(count, dtype=float)
    lower = np.where(
        count > 0, scipy_stats.chi2.ppf(alpha / 2, 2 * count) / 2, 0.0
    )
    upper = scipy_stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / 2
    return lower, upper


@app.function
def shade_palette(kind_key, num):
    """Shades of a site class's ETI color, from dark to light."""
    _dark, __, _light = ETI_PALETTE[kind_key]
    return sns.blend_palette([_dark, _light], max(num, 2))[:num]


@app.function
def make_panel_layout(kind_order, site_info):
    """One row of panels per site class, one panel per effect size (largest
    first) or per neutral site: (trait, kind, palette key, label)."""
    # one row per site class, one column per effect size (largest first),
    # or per neutral site
    _kind_keys = dict(zip(kind_order, ["neutral", "BENwg", "DELbg", "MLS"]))
    panel_layout = []
    for _kind in kind_order:
        _sites = site_info[site_info["kind"] == _kind]
        if _kind.startswith("NEU"):
            _sites = _sites.sort_values("site")
        else:
            _sites = _sites.sort_values(
                "effect_size", key=lambda x: -x.astype(float)
            )
        panel_layout.append(
            [
                (
                    _row["trait"],
                    _kind,
                    _kind_keys[_kind],
                    (
                        f"NEU site {_row['site']}"
                        if _kind.startswith("NEU")
                        else f"s = {_row['effect_size']}%"
                    ),
                )
                for __, _row in _sites.iterrows()
            ],
        )
    return panel_layout


@app.function
def obs_exp_by_size(data, sizes, panel_layout):
    """Observed / expected gains by clade size, per site class."""
    _fig, _axes = plt.subplots(
        1,
        len(panel_layout),
        figsize=(2.1 * len(panel_layout), 3.1),
        sharey=True,
    )
    for _ax, _row in zip(_axes, panel_layout):
        _colors = shade_palette(_row[0][2], len(_row))
        for _j, ((_trait, _kind, __, _label), _color) in enumerate(
            zip(_row, _colors)
        ):
            _d = (
                data[data["trait"] == _trait]
                .set_index("clade_size")
                .reindex(sizes)
            )
            _x = np.arange(len(sizes)) + (_j - 1.5) * 0.06
            _obs = _d["num_successes"].fillna(0).to_numpy()
            _lo, _hi = poisson_interval(_obs)
            for _fit, _style, _marker in (
                ("with survival time", "-", "o"),
                ("without survival time", "--", "o"),
            ):
                _exp = _d[f"expected {_fit}"].to_numpy()
                _ratio = np.maximum(_obs, 0.5) / _exp
                _ax.errorbar(
                    _x,
                    _ratio,
                    yerr=[_ratio - _lo / _exp, _hi / _exp - _ratio],
                    color=_color,
                    linestyle=_style,
                    marker=_marker,
                    markersize=3.5,
                    markerfacecolor=(_color if _style == "-" else "white"),
                    linewidth=1,
                    elinewidth=0.6,
                    label=_label if _style == "-" else None,
                )
        _ax.axhline(1, color="gray", linestyle=":", zorder=0)
        _ax.set_yscale("log")
        _ax.set_xticks(range(len(sizes)), sizes)
        _ax.set_title(_row[0][1])
        _ax.set_xlabel("Clade Size\n(Leaves)")
        _ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.62),
            ncol=2,
            fontsize="x-small",
            frameon=False,
            handlelength=1.2,
            columnspacing=0.8,
        )
    _axes[0].set_ylabel("Observed ÷\nExpected Gains")
    _fig.legend(
        [
            plt.Line2D([], [], color="black", linestyle="-"),
            plt.Line2D([], [], color="black", linestyle="--"),
        ],
        ["with survival time", "without survival time"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    _fig.tight_layout()


@app.function
def bin_by_length(fit_diagnostics, diagnostic_fits):
    """Bin each trait's branches by calibrated length, for all branches
    and for each of the first three clade size groups."""
    _num_bins = 20
    _sizes = (
        fit_diagnostics.sort_values("size_partition")["clade_size"]
        .drop_duplicates()
        .tolist()[:3]
    )
    _cols = [
        "num_trials",
        "num_successes",
        "expected with survival time",
        "expected without survival time",
    ]
    _binned = []
    for _group, _df in [
        ("all branches", fit_diagnostics),
        *(
            (_size, fit_diagnostics[fit_diagnostics["clade_size"] == _size])
            for _size in _sizes
        ),
    ]:
        for _trait, _part in _df.groupby("trait"):
            _part = _part.sort_values("length")
            # bins of equal total calibrated length (so, about equal
            # expected neutral gains), rather than equal branch counts,
            # which would crowd short, uninformative branches into many
            # noisy bins
            _weight = _part["num_trials"] * _part["length"]
            _cum = _weight.cumsum() / _weight.sum()
            _bin = np.minimum((_cum * _num_bins).astype(int), _num_bins - 1)
            _agg = (
                _part.assign(
                    weighted_length=_part["length"] * _part["num_trials"],
                    bin=_bin.to_numpy(),
                )
                .groupby("bin")[[*_cols, "weighted_length"]]
                .sum()
            )
            _agg["length"] = _agg["weighted_length"] / _agg["num_trials"]
            _binned.append(
                _agg.drop(columns="weighted_length")
                .reset_index()
                .assign(trait=_trait, group=_group),
            )
    length_bins = pd.concat(_binned, ignore_index=True)
    _lo, _hi = poisson_interval(length_bins["num_successes"])
    length_bins["share"] = (
        length_bins["num_successes"] / length_bins["num_trials"]
    )
    length_bins["share_lo"] = _lo / length_bins["num_trials"]
    length_bins["share_hi"] = _hi / length_bins["num_trials"]
    for _fit in ("with survival time", "without survival time"):
        length_bins[f"share {_fit}"] = (
            length_bins[f"expected {_fit}"] / length_bins["num_trials"]
        )
    _rates = diagnostic_fits[
        diagnostic_fits["fit"] == "with survival time"
    ].set_index("trait")
    length_bins["share neutral"] = length_bins["length"] * length_bins[
        "trait"
    ].map(_rates["neutral_gain_rate"])
    length_bins["share own rate"] = length_bins["length"] * length_bins[
        "trait"
    ].map(_rates["gain_rate"])
    length_bin_groups = ["all branches", *_sizes]
    return length_bins, length_bin_groups


@app.function
def gain_share_by_length(data, groups, relative, log_axes, panel_layout):
    """Binned gain share vs. calibrated branch length, per site."""
    _shares = [
        "share",
        "share_lo",
        "share_hi",
        "share with survival time",
        "share without survival time",
        "share neutral",
        "share own rate",
    ]
    if relative:
        data = data.assign(
            **{_c: data[_c] / data["share neutral"] for _c in _shares},
        )
    _fig, _axes = plt.subplots(
        len(panel_layout),
        max(map(len, panel_layout)),
        figsize=(9, 8),
        sharex=True,
        # log axes share a scale; linear axes fit each panel, as strong
        # effects (e.g., large gain ratios) would flatten the others
        sharey=log_axes,
        squeeze=False,
    )
    for _axrow, _row in zip(_axes, panel_layout):
        for _ax, (_trait, _kind, _key, _label) in zip(_axrow, _row):
            _colors = shade_palette(_key, len(groups))
            if len(groups) == 1:
                _colors = [ETI_PALETTE[_key][1]]
            _d = data[data["trait"] == _trait]
            _first = _d[_d["group"] == groups[0]]
            for _col, _style in (
                ("share neutral", (0, (4, 2))),
                ("share own rate", (0, (6, 2, 1, 2))),
            ):
                _ax.plot(
                    _first["length"],
                    _first[_col],
                    color="gray",
                    linestyle=_style,
                    linewidth=0.9,
                    zorder=1,
                )
            for _group, _color in zip(groups, _colors):
                _g = _d[_d["group"] == _group]
                _ax.errorbar(
                    _g["length"],
                    _g["share"],
                    yerr=[
                        _g["share"] - _g["share_lo"],
                        _g["share_hi"] - _g["share"],
                    ],
                    color=_color,
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    elinewidth=0.6,
                    zorder=3,
                )
                for _fit, _style in (
                    ("with survival time", "-"),
                    ("without survival time", ":"),
                ):
                    _ax.plot(
                        _g["length"],
                        _g[f"share {_fit}"],
                        color=_color,
                        linestyle=_style,
                        linewidth=1,
                        zorder=2,
                    )
            _ax.set_title(f"{_kind} {_label}", fontsize="small")
            if log_axes:
                _ax.set_xscale("log")
                _ax.set_yscale("log")
            else:
                # fit each panel to its fitted and reference lines and
                # observed bins, not to the wide intervals of bins with
                # few expected gains (e.g., short branches)
                _shown = _d[_d["group"].isin(groups)]
                _lines = _shown[
                    [
                        "share with survival time",
                        "share without survival time",
                        "share neutral",
                        "share own rate",
                    ]
                ].to_numpy()
                _top = 1.15 * max(
                    np.nanmax(_lines),
                    np.nanquantile(_shown["share"], 0.95),
                )
                _ax.set_ylim(0, max(_top, 2.5) if relative else _top)
    for _ax in _axes[-1]:
        _ax.set_xlabel("Calibrated\nBranch Length")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(
            "Gain Share ÷\nNeutral" if relative else "Share\nGaining"
        )
    _handles = [
        *(
            plt.Line2D([], [], color=_c, marker="o", linestyle="none")
            for _c in (
                shade_palette("neutral", len(groups))
                if len(groups) > 1
                else [ETI_PALETTE["neutral"][1]]
            )
        ),
        plt.Line2D([], [], color="black", linestyle="-"),
        plt.Line2D([], [], color="black", linestyle=":"),
        plt.Line2D([], [], color="gray", linestyle=(0, (4, 2))),
        plt.Line2D([], [], color="gray", linestyle=(0, (6, 2, 1, 2))),
    ]
    _labels = [
        *(f"clade size {_g}" if _g != "all branches" else _g for _g in groups),
        "fit with survival time",
        "fit without survival time",
        "neutral rate × length",
        "own rate × length",
    ]
    _fig.legend(
        _handles,
        _labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize="small",
    )
    _fig.tight_layout()


@app.cell(hide_code=True)
def delimit_analyze():
    mo.md(f"""
    ## Analyze Samples

    Each sample is analyzed in turn, then its phylogeny is discarded to bound memory:

    1. Fossil layer 0 (the initial population, at `origin_time == -1`) is dropped, lineages without retained leaves are pruned, and unifurcations collapsed.
    2. Origin times are raised to at least the parent's plus {ORIGIN_TIME_EPSILON}, so every branch has positive length.
    3. Each site's trait state at every node is reconstructed under parsimony without loss: a node has a trait if and only if every leaf of its clade does, except roots (the reconstruction's placeholder ancestors at the run start), which never carry a trait.
    4. The allmdc and pendant estimators are fit with raw and with neutral-calibrated branch lengths (calibrated per the estimator's documented example).
    5. The calibrated allmdc fit's expected gains are recomputed per exposure episode, with and without survival time, for fit diagnostics.
    """)
    return


@app.cell
def analyze_samples(neutral_sites, site_info):
    import gc

    _neutral = [f"site{_site}" for _site in neutral_sites]
    _summaries, _pes, _pendants, _cherries, _mdcs, _estimates = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    _bins, _fits, _obs_exp = [], [], []
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
        _calibration_df = mark_calibration_columns(_phylo, _names, _neutral)
        del _phylo
        gc.collect()
        _estimates.append(
            fit_estimates(_calibration_df, _names, _neutral).assign(
                sample=_sample
            )
        )
        _start_diag = time.time()
        _dfits, _diag = compute_fit_diagnostics(
            _calibration_df, _names, _neutral
        )
        _sample_bins, length_bin_groups = bin_by_length(_diag, _dfits)
        _bins.append(_sample_bins.assign(sample=_sample))
        _fits.append(_dfits.assign(sample=_sample))
        _obs_exp.append(
            _diag[_diag["clade_size"].isin(length_bin_groups[1:])]
            .groupby(["trait", "clade_size"], sort=False)[
                [
                    "num_successes",
                    "expected with survival time",
                    "expected without survival time",
                ]
            ]
            .sum()
            .reset_index()
            .assign(sample=_sample)
        )
        print(f"  fit diagnostics: {time.time() - _start_diag:.0f}s")
        del _calibration_df, _diag
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
    sample_length_bins = pd.concat(_bins, ignore_index=True)
    sample_obs_exp = pd.concat(_obs_exp, ignore_index=True)
    sample_diagnostic_fits = pd.concat(_fits, ignore_index=True).merge(
        site_info, on="trait"
    )
    mo.Html(sample_summary.to_html(float_format="{:.4g}".format))
    return (
        cherry_distances,
        length_bin_groups,
        pe_counts,
        pendant_lengths,
        sample_estimates,
        sample_length_bins,
        sample_mdcs,
        sample_obs_exp,
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
def note_density():
    mo.md(r"""
    The kdtree sample is much denser locally.
    It packs its genomes into about 55,000 PEs, about 72 per sampled PE, where the uniform sample spreads about 4.5 per PE over about 875,000 PEs.
    Its leaves are more closely related: the median pendant branch is about 3.3 times shorter (about 500 vs. 1,700 `origin_time` units), and 81% of its cherries pair two genomes from the same PE (vs. 35%).
    """)
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
def note_mdc_yield():
    mo.md(r"""
    Dense local sampling trades many separate gains for fewer, larger clades.
    For neutral and `BENwg` sites, the kdtree sample has about 2.4 times fewer MDCs (e.g., about 2,850 vs. 6,900 per neutral site), but slightly more inner MDCs, and its MDCs average about 3 leaves (vs. 1.2).
    For costly MLS and `DELbg` sites, it sees as many or somewhat fewer MDCs, but many more inner MDCs (e.g., about 300 vs. 30 for MLS at $s = 5\%$), as close relatives are sampled before carrier lineages are lost.
    """)
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


@app.cell(hide_code=True)
def delimit_fit_diagnostics():
    mo.md("""
    ## Fit Diagnostics

    Per sample, observed trait gains compared with those expected under the calibrated allmdc fit, recomputed from the estimator's own exposure episodes, calibrated branch lengths, and survival times (as in `2026-11-03-mls-indiv-spectra-fullgrid-fossils`).
    Two fits are compared: **with survival time** (the allmdc model) and **without survival time** (refit with every branch's survival time set to 0).
    Branches whose parent lacks a trait are grouped by clade size class and binned by calibrated branch length, into bins of equal total length.
    For reference, the neutral expectation is the mean neutral gain rate (for a neutral site, the other three's) times calibrated length, and the trait's own expectation is its fitted gain rate times calibrated length.
    Error bars are 95% Poisson intervals.
    """)
    return


@app.cell
def plot_fit_diagnostics(
    kind_order,
    length_bin_groups,
    sample_length_bins,
    sample_obs_exp,
    site_info,
):
    _layout = make_panel_layout(kind_order, site_info)
    for _sample in SAMPLES:
        tp.tee(
            obs_exp_by_size,
            data=sample_obs_exp[sample_obs_exp["sample"] == _sample],
            sizes=length_bin_groups[1:],
            panel_layout=_layout,
            teeplot_outattrs={"fit": "calibrated-allmdc", "sample": _sample},
            teeplot_outexclude=["sizes", "panel_layout"],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        )
        plt.close("all")
        for _groups, _relative, _log in (
            (["all branches"], False, False),
            (["all branches"], True, False),
            (["all branches"], False, True),
            (length_bin_groups[1:], False, False),
            (length_bin_groups[1:], True, False),
        ):
            tp.tee(
                gain_share_by_length,
                data=sample_length_bins[
                    sample_length_bins["sample"] == _sample
                ],
                groups=_groups,
                relative=_relative,
                log_axes=_log,
                panel_layout=_layout,
                teeplot_outattrs={
                    "sample": _sample,
                    "branches": (
                        "all"
                        if _groups == ["all branches"]
                        else "by-clade-size"
                    ),
                    "y": "relative-to-neutral" if _relative else "share",
                    "axes": "log" if _log else "linear",
                },
                teeplot_outexclude=[
                    "groups",
                    "relative",
                    "log_axes",
                    "panel_layout",
                ],
                teeplot_show=True,
                teeplot_subdir=pathlib.Path(__file__).stem,
            )
            plt.close("all")
    return


@app.cell(hide_code=True)
def note_power():
    mo.md(r"""
    ### Notes

    - **MLS sites:** the kdtree sample gives more evidence that gain rate exceeds neutral sites'.
      With calibrated pendant fits, whose neutral sites' statistics are near their null scale in both samples (mean neutral LRT about 1.6–1.7), $z$ rises from 6.0 to 10.0 at $s = 5\%$ and from 2.5 to 4.4 at $s = 2.5\%$; with calibrated allmdc fits, from 8.0 to 17.0 and from 4.7 to 9.3.
      Its MLS gain ratio estimates are also closer to the configured ones (allmdc, calibrated: 1.78 vs. 1.54 at $s = 5\%$, configured 1.99; 68 vs. 42 at $s = 25\%$, configured 82.7), as gains are seen on short branches before between-host cost removes their carriers.
    - **`BENwg` sites:** the kdtree sample gives less evidence (e.g., calibrated pendant $z$ 15.0 vs. 20.5 at $s = 5\%$), as these sites carry no cost, so the uniform sample's 2.4-fold more independent gains count for more.
    - **Null calibration:** the allmdc statistics are overdispersed in the kdtree sample: its neutral sites' mean gain ratio LRT is 5.5 (calibrated) vs. 1.9 for the uniform sample, and one `DELbg` site reaches $z = 3.5$.
      Denser sampling nests more branches, so the composite likelihood overstates the evidence more.
      Scaled by the neutral mean, the kdtree's MLS advantage shrinks (e.g., $z_\mathrm{adj}$ 7.2 vs. 5.8 at $s = 5\%$) and its `BENwg` deficit grows (9.6 vs. 18.6); with four neutral sites, this scaling is rough.
    - **Selection:** the kdtree sample gives less evidence of between-host cost, especially for pendant fits (e.g., MLS at $s = 5\%$: $z = -9.4$ vs. $-25.9$), as its short pendant branches leave little time for cost to act.
    - So, k-d tree group sampling does sample local clusters densely, and that improves power to detect raised gain rates of costly (MLS) traits, but it costs power for cost-free (`BENwg`) traits and for detecting between-host cost, and allmdc's naive statistics need a null calibration to compare across samples.
    """)
    return


if __name__ == "__main__":
    app.run()
