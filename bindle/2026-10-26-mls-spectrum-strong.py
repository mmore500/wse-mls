import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import pathlib
    import random
    import re
    import warnings

    import alifedata_phyloinformatics_convert as apc
    from hstrat import _auxiliary_lib as hstrat_aux
    import iplotx as ipx
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

    from pylib import osf_fetch, percentileofscore_grouped

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Computer Modern math, e.g., for italic site class qualifiers
    plt.rcParams["mathtext.fontset"] = "cm"

    OSF_SLUGS = ("nj2de", "xtacr", "y5x6p", "a8g6h")
    NUM_SITES = 16
    ALPHA = 0.05
    MAX_TREE_LEAVES = 4000
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
    """Express an effect size label as a percent, passing "neu" through."""
    return (
        effect
        if effect in ("neu", "neutral")
        else f"{float(effect) * 100:.1f}"
    )


@app.function
def top_legend(g, title=None):
    """Replace a FacetGrid's legend with a single-row legend above the plot,
    with the title (if any) as a leading dummy entry."""
    _handles = [*g.legend.legend_handles]
    _labels = [_text.get_text() for _text in g.legend.get_texts()]
    g.legend.remove()
    g._legend = None
    # reclaim the space reserved for the old side legend
    g.tight_layout()
    if title is not None:
        _handles.insert(0, plt.Line2D([], [], linestyle="none"))
        _labels.insert(0, title)
    # center over the panels, rather than the whole figure
    _boxes = [_ax.get_position() for _ax in g.axes.flat]
    _left = min(_b.x0 for _b in _boxes)
    _right = max(_b.x1 for _b in _boxes)
    _renderer = g.figure.canvas.get_renderer()
    _span = (_right - _left) * g.figure.bbox.width
    # tighten entry spacing as needed so the legend doesn't overhang
    for _spacing in (2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4):
        _legend = g.figure.legend(
            _handles,
            _labels,
            loc="lower center",
            bbox_to_anchor=((_left + _right) / 2, 1.0),
            ncol=len(_handles),
            frameon=False,
            handletextpad=0.5,
            columnspacing=_spacing,
        )
        if title is not None:
            # collapse the dummy entry's empty handle slot, so the title
            # sits flush and the legend's visible content is centered
            _first_row = _legend._legend_handle_box.get_children()[
                0
            ].get_children()[0]
            _first_row.get_children()[0].width = 0
        if _legend.get_window_extent(_renderer).width <= _span:
            break
        _legend.remove()


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
    _ticks = [
        _t.get_window_extent(_renderer).transformed(_to_fig)
        for _t in g.axes.flat[-2].get_xticklabels()
    ]
    g.figure.text(
        legend.get_window_extent(_renderer).transformed(_to_fig).x0,
        min(_b.y0 for _b in _ticks),
        text,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"],
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
    # MLS Spectrum (Strong) Subgrid Fossils: Detectability by Population Size, Effect Size, and Site Class

    Applies the phylogeny-based multilevel selection (MLS) screen from *Extending a Phylogeny-based Method for Detecting Signatures of Multi-level Selection for Applications in Artificial Life* (ISAL 2025, <https://doi.org/10.1162/ISAL.a.916>; code at <https://github.com/mmore500/multilevel-selection-concept>) to wafer-scale evolution data, following `2026-10-25-mls-spectrum-subgrid-fossils` with stronger effect sizes.

    Data were generated by [`wse-async-ga` `wsclust/2026-10-26-mls-spectrum-strong-subgrid-fossils`](https://github.com/mmore500/wse-async-ga/tree/2e85bc34f8aa360f0118acde5d3b779d211b1b5b/wsclust/2026-10-26-mls-spectrum-strong-subgrid-fossils) (added in [mmore500/wse-async-ga#92](https://github.com/mmore500/wse-async-ga/pull/92)) and are hosted at {", ".join(f"<https://osf.io/{slug}>" for slug in OSF_SLUGS)}.
    Each file is one run's minified hstrat-reconstructed phylogeny (`06-minify-keep-pe-phylo` output), for subgrid sizes 0 (i.e., the full grid), 27, 81, and 243.
    This minification collapses monomorphic clades only among genomes on the same processing element (PE), so each tip keeps a single PE location.
    Fossils were collected by snapshot sampling.
    Runs stopped after roughly 1,200 to 1,600 generations (genome counter up to about 3,200, at two ticks per generation), fewer than the 2026-10-25 runs.
    These runs include the earlier fix in [`79a94ab`](https://github.com/mmore500/wse-async-ga/commit/79a94ab1f78e7b1239ecb2bb50ec3fd8809055e0), under which an empty tournament winner never displaces a living resident; without it, a recolonizing founder (placed in the first slot that selection overwrites) was usually lost, leaving its tile empty but closed to immigrants, and the metapopulation died out early in the run.

    Runs use genome flavor `genome_mlsspectrumstrong_2026_10_26`, a copy of `genome_mlsspectrum_2026_10_25` with stronger effect sizes.
    The genome carries three spectra of four sites each, with effect sizes $s_i \\in [0.25, 0.1, 0.05, 0.025]$, plus four neutral sites:

    - **MLS** sites (`MLS`) are both favored within a tile population and harmful to recolonizing other tiles,
    - **between-group deleterious** sites (`DELbg`) are only harmful to recolonizing other tiles, and
    - **within-group beneficial** sites (`BENwg`) are only favored within a tile population.

    Labels follow the ETI framework, with evaluation bar notation giving each trait's between-group effect above and within-group effect below, and $\\varnothing$ for no effect: MLS $|^{{-}}_{{+}}$, DELbg $|^{{-}}_{{\\varnothing}}$, BENwg $|^{{\\varnothing}}_{{+}}$, and neutral sites NEU $|^{{\\varnothing}}_{{\\varnothing}}$.

    Each tile population goes extinct with probability 0.1 per generation (vs. 0.01 in the 2026-10-25 runs), and only extinct tiles accept immigrants, so genes move between tiles only by recolonization.
    An immigrant carrying a between-group deleterious site is blocked (clobbered) from recolonizing with probability $s_i$ per site, leaving the tile open to later immigrants.
    A within-group beneficial site gives relative fitness $1 + s'_i$ in size-2 tournaments, with $s'_i = (1 + s_i)^{{\\log_2(N) / 25}} - 1$ rescaled to tile population size $N$.
    Runs sweep subgrid size, where subgrids are blocks of tiles between which migration is disabled.

    Each subgrid is an isolated metapopulation, so populations are analyzed independently rather than pooled: within each population, the screen asks whether each focal (MLS, deleterious, or beneficial) site

    1. gives rise to more mutation-defined clades (MDCs) than expected for a neutral site (mutation-frequency screen), and
    2. gives rise to MDCs that are smaller, shorter-lived, and/or spread over fewer PEs than neutral-site MDCs (clade-fitness screen).

    Detectability is the fraction of populations in which a screen is significant at $\\alpha = {ALPHA}$, broken down by population size, site class, and effect size.
    As a null calibration, the same screens are applied to each of the four neutral sites, against the other neutral sites.
    """)
    return


@app.cell(hide_code=True)
def delimit_get_data():
    mo.md("""
    ## Get Data
    """)
    return


@app.cell
def fetch_data():
    data_paths = [osf_fetch(slug) for slug in OSF_SLUGS]
    return (data_paths,)


@app.cell
def show_schema(data_paths):
    pl.scan_parquet(data_paths[0]).collect_schema()
    return


@app.cell
def load_config(data_paths):
    def _config_cols(names):
        return [
            name
            for name in names
            if name.startswith(("ASYNC_GA_", "CEREBRASLIB_", "dstream_algo"))
            or re.match(r"^(mls|deleterious|beneficial)_", name)
            or name
            in (
                "config",
                "genomeFlavor",
                "populationFlavor",
                "globalSeed",
                "nCol",
                "nRow",
                "nColSubgrid",
                "nRowSubgrid",
                "nWav",
                "nTrait",
                "nCycle",
                "tilePopSize",
                "tournSize",
                "replicate",
                "trait_focal",
                "dstream_S",
                "hstrat_differentia_bitwidth",
            )
        ]

    _rows = []
    for _path in data_paths:
        _lazy = pl.scan_parquet(_path).filter(pl.col("clade_size") > 0)
        # each file holds one run; check before taking its first row
        assert _lazy.select(pl.col("config").n_unique()).collect().item() == 1
        _rows.append(
            _lazy.select(_config_cols(_lazy.collect_schema().names()))
            .head(1)
            .collect()
        )
    config_df = pl.concat(_rows, how="diagonal_relaxed").sort("nRowSubgrid")
    assert config_df["config"].is_unique().all(), "expected one row per run"
    assert (config_df["nRowSubgrid"] == config_df["nColSubgrid"]).all()
    config_df.transpose(include_header=True)
    return (config_df,)


@app.cell
def tabulate_effect_sizes(config_df):
    # site class K's level-i site is flag byte B, bit b (trait site 8B + b),
    # with clobber and within-population selection probabilities (0 where a
    # class lacks that effect)
    _site_cols = [
        (int(B) * 8 + int(b), kind, what, col)
        for col in config_df.columns
        for kind, what, B, b in re.findall(
            r"^(mls|deleterious|beneficial)"
            r"_(clobber|selection)"
            r"_p_byte(\d+)_bit(\d+)$",
            col,
        )
    ]
    effect_sizes = (
        pd.DataFrame(
            [
                {
                    "config": row["config"],
                    "site": site,
                    "kind": kind,
                    f"{what}_p": row[col],
                    "clobber_scale": row["CEREBRASLIB_CLOBBER_IMMIGRANT_P"],
                }
                for row in config_df.iter_rows(named=True)
                for site, kind, what, col in _site_cols
            ],
        )
        .groupby(["config", "site"], as_index=False)
        .first()
    )
    # base effect size s_i is clobber chance over its scale; within-group
    # beneficial sites have none, so take s_i from the MLS site with the
    # same selection probability
    effect_sizes["effect_size"] = (
        effect_sizes["clobber_p"] / effect_sizes["clobber_scale"]
    ).round(6)
    _is_mls = effect_sizes["kind"] == "mls"
    _gain_to_s = dict(
        zip(
            effect_sizes.loc[_is_mls, "selection_p"],
            effect_sizes.loc[_is_mls, "effect_size"],
        ),
    )
    _is_ben = effect_sizes["kind"] == "beneficial"
    effect_sizes.loc[_is_ben, "effect_size"] = effect_sizes.loc[
        _is_ben, "selection_p"
    ].map(_gain_to_s)
    assert effect_sizes["effect_size"].notna().all()
    # effect sizes are the same across runs and site classes
    assert (effect_sizes.groupby("site")["effect_size"].nunique() == 1).all()
    assert (
        effect_sizes.groupby("kind")["effect_size"]
        .apply(lambda x: tuple(sorted(x.unique())))
        .nunique()
        == 1
    )
    focal_sites = sorted(effect_sizes["site"].unique().tolist())
    _focal_mask = sum(1 << site for site in focal_sites)
    assert (config_df["trait_focal"] == _focal_mask).all(), "focal mismatch"
    background_sites = [
        site for site in range(NUM_SITES) if site not in focal_sites
    ]
    # every neutral site serves as a null control, against the others
    control_sites = background_sites

    # ETI evaluation bar notation: between-group effect above, within-group
    # effect below, with null (crossed circle) for no effect
    kind_labels = {
        "MLS": r"MLS$|^{-}_{+}$",
        "DELbg": r"DEL$\mathit{bg}|^{-}_{\varnothing}$",
        "BENwg": r"BEN$\mathit{wg}|^{\varnothing}_{+}$",
        "NEU": r"NEU$|^{\varnothing}_{\varnothing}$",
    }
    _kind_labels = {
        "mls": kind_labels["MLS"],
        "deleterious": kind_labels["DELbg"],
        "beneficial": kind_labels["BENwg"],
    }
    _site_info = effect_sizes.groupby("site")[["kind", "effect_size"]].first()
    site_labels = pd.Series(
        {
            site: (
                f"{_kind_labels[_site_info.loc[site, 'kind']]}"
                f" s={_site_info.loc[site, 'effect_size']:g}"
                if site in focal_sites
                else kind_labels["NEU"]
            )
            for site in range(NUM_SITES)
        },
    )
    mutation_order = [
        f"{_kind_labels[_kind]} s={_s:g}"
        for _kind in _kind_labels
        for _s in sorted(_site_info["effect_size"].unique(), reverse=True)
    ] + [kind_labels["NEU"]]
    effect_sizes
    return (
        background_sites,
        control_sites,
        focal_sites,
        kind_labels,
        mutation_order,
        site_labels,
    )


@app.cell(hide_code=True)
def delimit_prep_data():
    mo.md("""
    ## Prep Data

    Minification collapsed each maximal clade of genomes sharing identical traits onto its root, which became a leaf; `clade_size` gives the number of fossil genomes it represents, and `last_origin_time` the latest among them.
    Genome counts below weight leaves by `clade_size`.

    Fossil layer 0 samples the initial population, at `origin_time == -1` before any lineage history accumulated (as in `2026-10-15-mls-strong-fossils`), so these genomes are dropped (collapsed leaves are kept if any member has `last_origin_time >= 0`).
    Some roots have no `origin_time`; as roots can be no later than the run start, it is set to -1.
    Lineages without retained leaves are pruned and unifurcations collapsed.
    """)
    return


@app.cell
def load_phylogeny(config_df, data_paths, site_labels):
    # runs are independent, so each run's file is loaded, checked, summarized,
    # pruned, and split into populations separately to bound memory use
    _config_rows = {
        row["config"]: (i, row)
        for i, row in enumerate(
            config_df.select(
                "config",
                "nRow",
                "nCol",
                "nRowSubgrid",
                "nColSubgrid",
                "tilePopSize",
            ).iter_rows(named=True),
        )
    }
    _phylos, _pops, _splits = [], [], []
    _tips, _site_freqs, _times = [], [], []
    _offset = 0
    for _path in data_paths:
        # unminify: rows are in depth-first order, with id = row index
        _raw = (
            pl.read_parquet(
                _path,
                columns=[
                    "ancestor_offset",
                    "origin_time",
                    "last_origin_time",
                    "clade_size",
                    "trait_present",
                    "config",
                    "layer",
                    "subgrid",
                    "row",
                    "col",
                    "nCol",
                ],
            )
            .with_row_index("id")
            .with_columns(
                pl.col("id").cast(pl.Int64),
                ancestor_id=(
                    pl.col("id").cast(pl.Int64)
                    - pl.col("ancestor_offset").cast(pl.Int64)
                ),
                config=pl.col("config").cast(pl.String),
                # the full-grid run also fills subgrid (0) on inner nodes
                subgrid=pl.when(pl.col("clade_size") > 0).then(
                    pl.col("subgrid").cast(pl.Int64)
                ),
                trait_present=pl.col("trait_present").cast(pl.Int64),
                # processing element (PE) of each genome tip
                pe=pl.when(pl.col("clade_size") > 0).then(
                    pl.col("row").cast(pl.Int64) * pl.col("nCol")
                    + pl.col("col").cast(pl.Int64)
                ),
            )
            .drop("ancestor_offset", "row", "col", "nCol")
            .to_pandas()
        )

        # some roots lack origin_time; roots can be no later than run start
        _no_time = _raw["origin_time"].isna()
        assert (
            _raw.loc[_no_time, "ancestor_id"] == _raw.loc[_no_time, "id"]
        ).all()
        _raw["origin_time"] = _raw["origin_time"].fillna(-1)
        _raw["last_origin_time"] = _raw["last_origin_time"].fillna(
            _raw["origin_time"]
        )

        _is_genome = _raw["clade_size"] > 0
        assert (_is_genome == _raw["trait_present"].notna()).all()
        assert (_is_genome == _raw["subgrid"].notna()).all()
        assert _raw["config"].notna().all()
        assert (_raw["ancestor_id"] <= _raw["id"]).all()
        _has_child = np.zeros(len(_raw), dtype=bool)
        _has_child[_raw["ancestor_id"]] = True
        _has_child[_raw["ancestor_id"] == _raw["id"]] = False
        # collapsed monomorphic clades and fossil genomes are leaves
        assert not (_has_child & _is_genome).any()

        _tips.append(
            _raw[_is_genome]
            .groupby("config")
            .agg(
                num_leaves=("id", "size"),
                num_genomes=("clade_size", "sum"),
                clade_size_mean=("clade_size", "mean"),
                clade_size_max=("clade_size", "max"),
                num_layers=("layer", "nunique"),
                origin_time_min=("origin_time", "min"),
                last_origin_time_max=("last_origin_time", "max"),
                negative_origin_frac=(
                    "last_origin_time",
                    lambda x: (x < 0).mean(),
                ),
            )
            .assign(num_nodes=len(_raw)),
        )
        _times.append(
            _raw.assign(is_genome=_is_genome)
            .groupby(["config", "is_genome"])[
                ["origin_time", "last_origin_time"]
            ]
            .describe(),
        )

        _extant = _is_genome & (_raw["last_origin_time"] >= 0)
        _weights = _raw.loc[_extant, "clade_size"].to_numpy(dtype=float)
        _present = _raw.loc[_extant, "trait_present"].to_numpy(dtype=np.uint64)
        _site_freqs.append(
            pd.Series(
                {
                    f"site {site} ({site_labels[site]})": (
                        ((_present >> np.uint64(site)) & np.uint64(1))
                        * _weights
                    ).sum()
                    / _weights.sum()
                    for site in range(NUM_SITES)
                },
                name=_raw["config"].iloc[0],
            ),
        )

        _part = (
            _raw.assign(extant=_extant)
            .pipe(pfl.alifestd_to_working_format, mutate=True)
            .pipe(pfl.alifestd_prune_extinct_lineages_asexual, mutate=True)
            .pipe(pfl.alifestd_collapse_unifurcations, mutate=True)
            .pipe(pfl.alifestd_to_working_format, mutate=True)
            .pipe(pfl.alifestd_mark_leaves, mutate=True)
            .reset_index(drop=True)
        )
        del _raw
        assert (_part["is_leaf"] == _part["extant"]).all()
        _part = _part.drop(columns=["extant", "layer"])
        _code, _config_row = _config_rows[_part["config"].iloc[0]]
        _phylo, _pop, _split = split_run_populations(_part, _code, _config_row)
        del _part
        # offset ids so they are unique across runs
        for _col in ("id", "ancestor_id", "root_id"):
            _phylo[_col] += _offset
        _offset += len(_phylo)
        _phylos.append(_phylo)
        _pops.append(_pop)
        _splits.append(_split)
        del _phylo

    phylo = pd.concat(_phylos, ignore_index=True)
    del _phylos
    assert (phylo["id"] == np.arange(len(phylo))).all()
    populations = pd.concat(_pops, ignore_index=True)
    pop_size_order = (
        populations.sort_values("pop_tiles")["pop_size"].unique().tolist()
    )
    split_summary = pd.DataFrame(_splits).sum().astype(int).to_frame("value")
    tips_summary = pd.concat(_tips).T
    times_summary = pd.concat(_times).T
    # genome-weighted fraction of fossil genomes carrying each site, by run
    site_frequencies = pd.DataFrame(_site_freqs).T
    phylo.info()
    return (
        phylo,
        pop_size_order,
        populations,
        site_frequencies,
        split_summary,
        times_summary,
        tips_summary,
    )


@app.cell
def tabulate_tips(tips_summary):
    tips_summary
    return


@app.cell
def describe_times(times_summary):
    times_summary
    return


@app.cell
def tabulate_site_frequencies(site_frequencies):
    # render in full; marimo's default display truncates to ten rows
    mo.Html(site_frequencies.to_html(float_format="{:.5f}".format))
    return


@app.cell(hide_code=True)
def note_site_frequencies():
    mo.md("""
    Within-group beneficial (`BENwg`) sites are far more common than neutral sites, increasingly so with effect size (roughly 20-fold at $s = 0.025$ to 150-fold at $s = 0.25$).
    Between-group deleterious (`DELbg`) sites are rarer than neutral sites at the larger effect sizes (roughly 0.3-fold at $s = 0.25$ and 0.55-fold at $s = 0.1$), and about as common at $s \\le 0.05$.
    MLS sites track `BENwg` sites at $s \\le 0.1$, but at $s = 0.25$ fall well below them (roughly 0.45-fold), so the recolonization cost partly offsets their within-population advantage only at the largest effect size.
    """)
    return


@app.cell(hide_code=True)
def delimit_split_populations():
    mo.md("""
    ## Split Populations

    Subgrids share no migration, so no true common ancestry links genomes from different subgrids (or runs); each run's tree was reconstructed independently, but hstrat reconstruction can still join unrelated lineages at spurious deep nodes.
    To analyze each population (run × subgrid) independently, nodes whose descendant leaves span more than one population are deleted, making each maximal single-population clade a root.
    Each population's forest is then analyzed separately.

    Edge subgrids are truncated by the grid boundary (e.g., 755 columns is not a multiple of 27), so only full-size subgrids are retained.
    """)
    return


@app.function
def split_run_populations(part, config_code, config_row):
    """Split one run's pruned tree into its populations (subgrids), keeping
    full-size ones, and mark per-node clade stats used downstream."""
    _is_leaf = part["is_leaf"].to_numpy()
    _pop_code = np.where(
        _is_leaf,
        config_code * 2**32
        + part["subgrid"].fillna(0).to_numpy(dtype=np.int64),
        -1,
    )
    _pure = pfl.alifestd_mask_monomorphic_clades_asexual(
        part[["id", "ancestor_id"]],
        trait_mask=_is_leaf,
        trait_values=_pop_code,
    )["alifestd_mask_monomorphic_clades_asexual"].to_numpy()
    _ancestor_ids = part["ancestor_id"].to_numpy()
    _is_root = _ancestor_ids == part["id"].to_numpy()
    split_summary = pd.Series(
        {
            "num_nodes": len(part),
            "num_multipopulation_nodes": int((~_pure).sum()),
            "num_roots_before": int(_is_root.sum()),
            "num_roots_after": int(
                (_pure & (_is_root | ~_pure[_ancestor_ids])).sum()
            ),
        },
    )
    _phylo_split = (
        part[_pure]
        .drop(columns=["config"])
        .assign(
            ancestor_id=np.where(
                _pure[_ancestor_ids], _ancestor_ids, part["id"]
            )[_pure],
            pop_code=_pop_code[_pure],
        )
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_mark_roots, mutate=True)
        .pipe(pfl.alifestd_mark_root_id, mutate=True)
        .reset_index(drop=True)
    )
    del _pure, _pop_code, _ancestor_ids, _is_root
    _root_pops = (
        _phylo_split.loc[_phylo_split["is_leaf"]]
        .groupby("root_id")["pop_code"]
        .agg(["first", "nunique"])
    )
    assert (_root_pops["nunique"] == 1).all()
    _phylo_split["pop_code"] = _phylo_split["root_id"].map(_root_pops["first"])
    assert (_phylo_split["pop_code"] >= 0).all()
    _phylo_split["subgrid"] = (_phylo_split["pop_code"] % 2**32).astype(int)
    _leaves = _phylo_split[_phylo_split["is_leaf"]]
    populations = (
        _leaves.groupby("pop_code")
        .agg(
            subgrid=("subgrid", "first"),
            num_leaves=("id", "size"),
            num_genomes=("clade_size", "sum"),
            num_roots=("root_id", "nunique"),
        )
        .reset_index()
        .assign(**config_row)
    )
    del _leaves
    _has_subgrids = populations["nRowSubgrid"] > 0
    _nrow_sub = populations["nRowSubgrid"].where(
        _has_subgrids, populations["nRow"]
    )
    _ncol_sub = populations["nColSubgrid"].where(
        _has_subgrids, populations["nCol"]
    )
    _nsub_cols = -(-populations["nCol"] // _ncol_sub)
    _sub_row = populations["subgrid"] // _nsub_cols
    _sub_col = populations["subgrid"] % _nsub_cols
    populations["is_full_size"] = (
        (_sub_row + 1) * _nrow_sub <= populations["nRow"]
    ) & ((_sub_col + 1) * _ncol_sub <= populations["nCol"])
    populations["pop_tiles"] = (_nrow_sub * _ncol_sub).astype(int)
    populations["pop_size"] = [
        f"{r}x{c}"
        for r, c in zip(_nrow_sub.astype(int), _ncol_sub.astype(int))
    ]
    _full_pops = populations.loc[populations["is_full_size"], "pop_code"]
    phylo = (
        _phylo_split[_phylo_split["pop_code"].isin(_full_pops)]
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_mark_roots, mutate=True)
        .pipe(pfl.alifestd_mark_root_id, mutate=True)
        .pipe(pfl.alifestd_mark_num_leaves_asexual, mutate=True)
        .reset_index(drop=True)
    )
    del _phylo_split
    phylo["trait_present"] = (
        phylo["trait_present"].fillna(0).to_numpy(dtype=np.uint64)
    )
    _is_leaf = phylo["is_leaf"].to_numpy()
    phylo["num_genomes"] = pfl.alifestd_calc_clade_trait_count_asexual(
        phylo[["id", "ancestor_id"]],
        # passing counts rather than a mask sums genomes over descendants
        trait_mask=np.where(_is_leaf, phylo["clade_size"], 0),
    )
    # collapsed leaves' clades last until their latest member genome
    phylo["max_descendant_origin_time"] = (
        phylo[["id", "ancestor_id"]]
        .assign(
            origin_time=np.where(
                _is_leaf, phylo["last_origin_time"], phylo["origin_time"]
            ),
        )
        .pipe(pfl.alifestd_mark_max_descendant_origin_time_asexual)[
            "max_descendant_origin_time"
        ]
        .to_numpy()
    )
    phylo = phylo.pipe(pfl.alifestd_mark_clade_duration_asexual, mutate=True)
    return phylo, populations, split_summary


@app.cell
def tabulate_split(split_summary):
    split_summary
    return


@app.cell
def tabulate_populations(populations):
    populations.groupby(["pop_size", "is_full_size"], sort=False).agg(
        num_populations=("pop_code", "size"),
        pop_tiles=("pop_tiles", "first"),
        tile_pop_size=("tilePopSize", "first"),
        num_leaves_median=("num_leaves", "median"),
        num_genomes_median=("num_genomes", "median"),
        num_genomes_min=("num_genomes", "min"),
        num_genomes_max=("num_genomes", "max"),
        num_roots_median=("num_roots", "median"),
    ).T
    return


@app.cell
def check_clade_duration(phylo):
    _bad = ~(phylo["clade_duration"] >= 0)
    phylo.loc[
        _bad,
        [
            "id",
            "ancestor_id",
            "is_leaf",
            "origin_time",
            "last_origin_time",
            "max_descendant_origin_time",
            "clade_duration",
            "clade_size",
            "pop_code",
        ],
    ].describe(include="all").T.assign(num_bad=int(_bad.sum()))
    return


@app.cell(hide_code=True)
def delimit_tree_plots():
    mo.md(f"""
    ## Example Phylogenies

    Radial tree drawings in the style of the reference work's wafer-scale figure, one per population size.
    For each, a population is sampled at random and its largest tree with at most {MAX_TREE_LEAVES} leaves is drawn.
    Diamonds mark tips carrying any focal (MLS, deleterious, or beneficial) mutation; circles mark tips carrying any neutral (background) mutation.
    """)
    return


@app.cell
def plot_trees(focal_sites, phylo, pop_size_order, populations):
    _focal_mask = np.uint64(sum(1 << site for site in focal_sites))
    _roots = phylo.loc[phylo["is_root"], ["id", "num_leaves", "pop_code"]]
    _roots = _roots[_roots["num_leaves"] <= MAX_TREE_LEAVES]
    _rng = random.Random(SEED)
    for _pop_size in pop_size_order:
        _pops = populations.loc[
            populations["is_full_size"]
            & (populations["pop_size"] == _pop_size)
            & populations["pop_code"].isin(_roots["pop_code"]),
            "pop_code",
        ].tolist()
        if not _pops:
            continue
        _pop_code = _rng.choice(_pops)
        _root_id = (
            _roots.loc[_roots["pop_code"] == _pop_code]
            .nlargest(1, "num_leaves")["id"]
            .item()
        )

        # mark traits on the subtree only, to avoid copying the whole forest
        _subtree = phylo[phylo["root_id"] == _root_id]
        _trait_present = _subtree["trait_present"].to_numpy(dtype=np.uint64)
        _is_leaf = _subtree["is_leaf"].to_numpy()
        _subtree = (
            _subtree.assign(
                has_mls=np.where(
                    _is_leaf, (_trait_present & _focal_mask) != 0, np.nan
                ),
                has_neutral=np.where(
                    _is_leaf, (_trait_present & ~_focal_mask) != 0, np.nan
                ),
            )
            .pipe(pfl.alifestd_assign_contiguous_ids)
            .pipe(pfl.alifestd_try_add_ancestor_list_col)
        )
        _tree = apc.alife_dataframe_to_dendropy_tree(
            _subtree,
            setattrs=["has_mls", "has_neutral"],
        )
        with hstrat_aux.RngStateContext(6):
            _tree.ladderize()
            _tree.reorder(key=lambda *args: max(random.random(), 0.89))

        with tp.teed(
            ipx.plotting.tree,
            _tree,
            aspect=0.7,
            edge_color="darkgray",
            edge_linewidth=0.9,
            ladderize=True,
            layout="radial",
            layout_angular=True,
            layout_start=0.25,
            margins=0,
            teeplot_outattrs={"pop_size": _pop_size, "root_id": _root_id},
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _teed:
            plt.gcf().set_size_inches(6, 6)

            for _n, (_x, _y) in _teed.get_layout().T.items():
                _x_, _y_ = _x * np.cos(_y), _x * np.sin(_y)
                if _n.has_mls == 1.0:
                    plt.plot(
                        _x_,
                        _y_,
                        "d",
                        markersize=8,
                        color="#E2FFC4",
                        alpha=0.8,
                        zorder=0,
                    )
                    plt.plot(
                        _x_,
                        _y_,
                        "d",
                        markersize=5,
                        color="#6AB81C",
                        alpha=1,
                        zorder=2,
                    )
                if _n.has_neutral == 1.0:
                    plt.plot(
                        _x_,
                        _y_,
                        "o",
                        markersize=5,
                        color="#C5A3FF",
                        alpha=0.3,
                        zorder=1,
                    )
                    plt.plot(
                        _x_,
                        _y_,
                        "o",
                        markersize=2.5,
                        color="#C5A3FF",
                        alpha=1.0,
                        zorder=3,
                    )
                if _n is _tree.seed_node:
                    for _size in 3.5, 5.5:
                        plt.plot(
                            _x_,
                            _y_,
                            "o",
                            markersize=_size,
                            color="black",
                            alpha=1,
                            fillstyle="none",
                        )

            plt.gca().margins(x=0.02, y=0.02)
            plt.title(f"{_pop_size} subgrid")

            (_h0,) = plt.plot(
                [], [], "d", markersize=0, color="#FFFFFF", label="Mutation | "
            )
            (_h1,) = plt.plot(
                [], [], "d", markersize=8, color="#6AB81C", label="Focal"
            )
            (_h2,) = plt.plot(
                [], [], "o", markersize=8, color="#C5A3FF", label="Neutral"
            )
            plt.legend(
                handles=[_h0, _h1, _h2],
                loc="upper center",
                bbox_to_anchor=(0.48, 0.05),
                frameon=False,
                ncol=4,
                fontsize=18,
                columnspacing=0.2,
                handletextpad=-0.2,
            )
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_screen_mdcs():
    mo.md("""
    ## Screen Mutation-defined Clades

    MDCs are identified for each of the 16 trait sites by Fitch maximum parsimony over leaf states (root state assumed absent), as in `2026-10-15-mls-strong-fossils`.
    Because each population's trees share no nodes with other populations', one screen over the whole forest is equivalent to screening each population separately.

    The screen uses phyloframe's `alifestd_screen_trait_defined_clades_fitch_asexual`, which requires phyloframe >= 0.12.2 to run in linear time at this data size.

    MDC size is measured as `carriers`, the number of descendant fossil genomes carrying the mutation; `num_genomes` counts all descendant fossil genomes.
    `carrier_frac` is the carrier fraction among descendant genomes, and `is_terminal` marks MDCs arising on a terminal branch to a single genome (MDC node is an uncollapsed leaf).
    `clade_duration` spans from MDC origin to its latest descendant fossil.
    `num_pes` counts the distinct processing elements (PEs) among the MDC's carrier genomes, a measure of its spatial spread.
    """)
    return


@app.cell
def screen_mdcs(phylo):
    _work = phylo[["id", "ancestor_id"]].copy()
    assert (_work["id"] == np.arange(len(_work))).all()
    _is_leaf = phylo["is_leaf"].to_numpy()
    _clade_size = phylo["clade_size"].fillna(0).to_numpy(dtype=np.int64)
    _trait_present = phylo["trait_present"].to_numpy(dtype=np.uint64)
    _pe = phylo["pe"].fillna(-1).to_numpy(dtype=np.int64)
    assert (_pe[_is_leaf] >= 0).all()
    # each clade's nodes form a contiguous block in preorder, starting at the
    # clade root and spanning its descendants
    _preorder_pos = np.empty(len(_work), dtype=np.int64)
    _preorder_pos[pfl.alifestd_unfurl_traversal_preorder_asexual(_work)] = (
        np.arange(len(_work))
    )
    _num_descendants = pfl.alifestd_mark_num_descendants_asexual(_work)[
        "num_descendants"
    ].to_numpy()
    mdc_ids = {}
    mdc_carriers = {}
    mdc_num_pes = {}
    for _site in mo.status.progress_bar(range(NUM_SITES)):
        _has = ((_trait_present >> np.uint64(_site)) & np.uint64(1)) != 0
        _mask = _work.pipe(
            pfl.alifestd_screen_trait_defined_clades_fitch_asexual,
            mutate=True,
            mask_trait_absent=_is_leaf & ~_has,
            mask_trait_present=_is_leaf & _has,
        ).astype(bool)
        # passing counts rather than a mask sums carriers over descendants
        _carriers = _work.pipe(
            pfl.alifestd_calc_clade_trait_count_asexual,
            mutate=True,
            trait_mask=np.where(_is_leaf & _has, _clade_size, 0),
        )
        _ids = np.flatnonzero(_mask)
        mdc_ids[_site] = _ids
        mdc_carriers[_site] = _carriers[_mask]
        # count distinct PEs among each MDC's carrier tips, by pairing each
        # MDC with the carrier tips in its preorder block
        _tips = np.flatnonzero(_is_leaf & _has)
        _order = np.argsort(_preorder_pos[_tips])
        _tip_pos = _preorder_pos[_tips][_order]
        _tip_pe = _pe[_tips][_order]
        _lo = np.searchsorted(_tip_pos, _preorder_pos[_ids])
        _hi = np.searchsorted(
            _tip_pos, _preorder_pos[_ids] + _num_descendants[_ids], "right"
        )
        _counts = _hi - _lo
        _mdc_idx = np.repeat(np.arange(len(_ids)), _counts)
        _tip_idx = np.repeat(_lo - np.cumsum(_counts) + _counts, _counts)
        _tip_idx += np.arange(len(_tip_idx))
        _pairs = np.unique(
            np.stack([_mdc_idx, _tip_pe[_tip_idx]], axis=1), axis=0
        )
        mdc_num_pes[_site] = np.bincount(_pairs[:, 0], minlength=len(_ids))
    return mdc_carriers, mdc_ids, mdc_num_pes


@app.cell
def tabulate_mdcs(
    control_sites,
    focal_sites,
    mdc_carriers,
    mdc_ids,
    mdc_num_pes,
    phylo,
    populations,
    site_labels,
):
    _cols = [
        "id",
        "origin_time",
        "num_genomes",
        "clade_duration",
        "is_leaf",
        "clade_size",
        "pop_code",
    ]
    mdcs = pd.concat(
        [
            phylo.loc[ids, _cols].assign(
                site=site,
                is_focal=site in focal_sites,
                is_control=site in control_sites,
                carriers=mdc_carriers[site],
                num_pes=mdc_num_pes[site],
            )
            for site, ids in mdc_ids.items()
        ],
        ignore_index=True,
    )
    mdcs["is_terminal"] = mdcs["is_leaf"] & (mdcs["clade_size"] == 1)
    mdcs["carrier_frac"] = mdcs["carriers"] / mdcs["num_genomes"]
    mdcs["mutation"] = mdcs["site"].map(site_labels)
    mdcs = mdcs.drop(columns=["is_leaf", "clade_size"]).merge(
        populations[["pop_code", "pop_size", "pop_tiles"]], on="pop_code"
    )
    mdcs.groupby(["pop_size", "mutation"], sort=False).agg(
        num_mdcs=("id", "size"),
        carriers_mean=("carriers", "mean"),
        carriers_singleton_frac=("carriers", lambda x: (x == 1).mean()),
        carrier_frac_mean=("carrier_frac", "mean"),
        terminal_frac=("is_terminal", "mean"),
        clade_duration_mean=("clade_duration", "mean"),
        num_pes_mean=("num_pes", "mean"),
    ).unstack("pop_size")
    return (mdcs,)


@app.cell(hide_code=True)
def delimit_freq_screen():
    mo.md("""
    ## Mutation-frequency Screen

    Within each population, baseline rate $\lambda$ is the mean MDC count per background (neutral) site.
    Site significance is the Poisson survival probability of observing at least the site's MDC count under $\mathsf{Pois}(\lambda)$.
    Neutral control sites are screened the same way, against the mean over the other background sites.
    Populations with no background MDCs ($\lambda = 0$) are not screened.
    """)
    return


@app.cell
def calc_freq_screen(
    background_sites,
    control_sites,
    focal_sites,
    mdcs,
    populations,
    site_labels,
):
    _pops = populations.loc[populations["is_full_size"], "pop_code"]
    _counts = (
        mdcs.groupby(["pop_code", "site"])
        .size()
        .unstack("site")
        .reindex(index=_pops, columns=range(NUM_SITES))
        .fillna(0)
    )
    _bg_sum = _counts[background_sites].sum(axis=1)
    _num_bg = len(background_sites)
    _records = []
    for _site in [*focal_sites, *control_sites]:
        _is_control = _site in control_sites
        _lam = (
            (_bg_sum - _counts[_site]) / (_num_bg - 1)
            if _is_control
            else _bg_sum / _num_bg
        )
        _k = _counts[_site]
        _records.append(
            pd.DataFrame(
                {
                    "pop_code": _counts.index,
                    "site": _site,
                    "is_control": _is_control,
                    "mdc_count": _k.to_numpy(),
                    "lambda": _lam.to_numpy(),
                    "fold_enrichment": (_k / _lam).to_numpy(),
                    "poisson_sf_p": np.where(
                        _lam > 0, scipy_stats.poisson.sf(_k - 1, _lam), np.nan
                    ),
                },
            ),
        )
    freq_screen = pd.concat(_records, ignore_index=True).merge(
        populations[["pop_code", "pop_size", "pop_tiles"]], on="pop_code"
    )
    freq_screen["mutation"] = freq_screen["site"].map(site_labels)
    freq_screen["detected"] = freq_screen["poisson_sf_p"] < ALPHA
    freq_screen.groupby(["pop_size", "mutation"], sort=False).agg(
        num_populations=("pop_code", "nunique"),
        mdc_count_mean=("mdc_count", "mean"),
        lambda_mean=("lambda", "mean"),
        fold_enrichment_median=("fold_enrichment", "median"),
        detection_rate=("detected", "mean"),
    ).unstack("pop_size")
    return (freq_screen,)


@app.cell
def plot_freq_enrichment(freq_screen, mutation_order, pop_size_order):
    with tp.teed(
        sns.pointplot,
        data=freq_screen.replace([np.inf, -np.inf], np.nan),
        x="mutation",
        y="fold_enrichment",
        order=mutation_order,
        hue="pop_size",
        hue_order=pop_size_order,
        palette="viridis",
        estimator="median",
        errorbar=("pi", 50),
        dodge=0.4,
        teeplot_outexclude=["palette", "hue_order", "dodge", "order"],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _ax:
        gap_points(_ax, _ax.get_legend())
        _ax.axhline(1, color="gray", linestyle="--", zorder=0)
        _ax.set_yscale("log")
        _ax.set_xlabel("Mutation (site class and effect size)")
        _ax.set_ylabel("MDC Enrichment\n(median, IQR)")
        plt.setp(
            _ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        sns.move_legend(
            _ax,
            "upper left",
            bbox_to_anchor=(1, 1),
            title="Subgrid",
            frameon=False,
        )
        sns.despine(ax=_ax)
        _ax.figure.set_size_inches(8, 2.6)
    return


@app.cell
def plot_freq_detection(freq_screen, mutation_order, pop_size_order):
    with tp.teed(
        sns.pointplot,
        data=freq_screen.astype({"detected": float}),
        x="mutation",
        y="detected",
        order=mutation_order,
        hue="pop_size",
        hue_order=pop_size_order,
        palette="viridis",
        errorbar=("ci", 95),
        n_boot=1000,
        seed=SEED,
        dodge=0.4,
        teeplot_outexclude=["palette", "hue_order", "dodge", "order", "seed"],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _ax:
        gap_points(_ax, _ax.get_legend())
        _ax.axhline(ALPHA, color="gray", linestyle="--", zorder=0)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_xlabel("Mutation (site class and effect size)")
        _ax.set_ylabel("Detection Rate\n(frequency screen)")
        plt.setp(
            _ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        sns.move_legend(
            _ax,
            "upper left",
            bbox_to_anchor=(1, 1),
            title="Subgrid",
            frameon=False,
        )
        sns.despine(ax=_ax)
        _ax.figure.set_size_inches(8, 2.6)
    return


@app.cell(hide_code=True)
def delimit_fitness_screen():
    mo.md("""
    ## Clade-fitness Screen

    Within each population, each MDC's size (`carriers`), clade duration, and PE count (`num_pes`) are percentile-ranked (mean rank for ties) against comparator MDCs: the pooled MDCs of background (neutral) sites in the same population, drawn from the same MDC subset.
    Under the `epoch100` normalization, comparison is further restricted to comparators originating within the same 100-update `origin_time` bin, controlling for recency bias.

    For each population and site, a one-sided Mann-Whitney U test asks whether the site's MDC percentiles are lower than those of background MDCs (excluding the site itself, for neutral control sites).
    MLS predicts lower percentiles.

    MDC subsets are `all` MDCs, `inner` (excluding terminal-branch MDCs), and `inner_f75` (non-terminal MDCs with at least 75% carriers).
    """)
    return


@app.cell
def define_subsets():
    subsets = {
        "all": lambda df: np.ones(len(df), dtype=bool),
        "inner": lambda df: ~df["is_terminal"].to_numpy(),
        "inner_f75": lambda df: (
            ~df["is_terminal"].to_numpy()
            & (df["carrier_frac"].to_numpy() >= 0.75)
        ),
    }
    return (subsets,)


@app.cell
def calc_normed_stats(mdcs, subsets):
    normalizations = {"all": None, "epoch100": 100}
    stat_cols = ("carriers", "clade_duration", "num_pes")
    _parts = []
    for _subset, _subset_mask in subsets.items():
        _data = mdcs[_subset_mask(mdcs)]
        _comparator = ~_data["is_focal"].to_numpy()
        _keep = ~_comparator | _data["is_control"].to_numpy()
        for _norm, _binwidth in normalizations.items():
            _epoch = (
                np.zeros(len(_data), dtype=np.int64)
                if _binwidth is None
                else (_data["origin_time"].to_numpy() // _binwidth).astype(
                    np.int64
                )
            )
            _groups = _data["pop_code"].to_numpy() * 2**16 + _epoch
            for _stat in stat_cols:
                _parts.append(
                    _data[
                        ["pop_code", "site", "is_focal", "is_control"]
                    ].assign(
                        subset=_subset,
                        normalization=_norm,
                        stat=_stat,
                        percentile=percentileofscore_grouped(
                            _data[_stat].to_numpy(dtype=float),
                            _groups,
                            _comparator,
                        ),
                    ),
                )
    normed = pd.concat(_parts, ignore_index=True).dropna(subset=["percentile"])
    return (normed,)


@app.cell
def calc_fitness_screen(control_sites, focal_sites, normed):
    _records = []
    _tested_sites = [*focal_sites, *control_sites]
    for (_pop, _subset, _norm, _stat), _group in mo.status.progress_bar(
        normed.groupby(
            ["pop_code", "subset", "normalization", "stat"], sort=False
        ),
    ):
        _sites = _group["site"].to_numpy()
        _pct = _group["percentile"].to_numpy()
        _is_bg = ~_group["is_focal"].to_numpy()
        for _site in _tested_sites:
            _in_site = _sites == _site
            _x = _pct[_in_site]
            _y = _pct[_is_bg & ~_in_site]
            if not len(_x) or not len(_y):
                continue
            _mw_u, _mw_p = scipy_stats.mannwhitneyu(_x, _y, alternative="less")
            _records.append(
                {
                    "pop_code": _pop,
                    "subset": _subset,
                    "normalization": _norm,
                    "stat": _stat,
                    "site": _site,
                    "n_site": len(_x),
                    "n_background": len(_y),
                    "site_mean_pct": _x.mean(),
                    "p_mw_less": _mw_p,
                    "cliffs_delta": 2 * _mw_u / (len(_x) * len(_y)) - 1,
                },
            )
    fitness_screen = pd.DataFrame(_records)
    return (fitness_screen,)


@app.cell
def tabulate_fitness_screen(
    control_sites,
    fitness_screen,
    freq_screen,
    populations,
):
    # one row per screened (population, site), including those with no MDCs
    # in subset, which count as not detected
    _pop_sites = freq_screen[["pop_code", "site", "mutation"]]
    fitness_detection = (
        fitness_screen[["subset", "normalization", "stat"]]
        .drop_duplicates()
        .merge(_pop_sites, how="cross")
        .merge(
            fitness_screen,
            on=["pop_code", "site", "subset", "normalization", "stat"],
            how="left",
        )
        .merge(populations[["pop_code", "pop_size", "pop_tiles"]])
    )
    fitness_detection["is_control"] = fitness_detection["site"].isin(
        control_sites
    )
    fitness_detection["detected"] = fitness_detection["p_mw_less"] < ALPHA
    fitness_detection.groupby(
        ["subset", "normalization", "stat", "pop_size", "mutation"],
        sort=False,
    ).agg(
        num_populations=("pop_code", "nunique"),
        num_screened=("p_mw_less", "count"),
        n_site_mean=("n_site", "mean"),
        site_mean_pct_mean=("site_mean_pct", "mean"),
        cliffs_delta_mean=("cliffs_delta", "mean"),
        detection_rate=("detected", "mean"),
    ).unstack(
        "pop_size"
    )
    return (fitness_detection,)


@app.cell
def plot_fitness_percentiles(
    fitness_detection,
    mutation_order,
    pop_size_order,
):
    with tp.teed(
        sns.catplot,
        data=fitness_detection[
            fitness_detection["normalization"] == "epoch100"
        ].dropna(subset=["site_mean_pct"]),
        x="mutation",
        y="site_mean_pct",
        order=mutation_order,
        hue="pop_size",
        hue_order=pop_size_order,
        palette="viridis",
        col="stat",
        row="subset",
        kind="point",
        estimator="median",
        errorbar=("pi", 50),
        dodge=0.4,
        height=2.4,
        aspect=2.6,
        teeplot_outattrs={"normalization": "epoch100"},
        teeplot_outexclude=["palette", "hue_order", "dodge", "order"],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _g:
        gap_points(*_g.axes.flat, _g.legend)
        _g.refline(y=50, color="gray", linestyle="--")
        _g.set_axis_labels(
            "Mutation (site class and effect size)",
            "Mean MDC Percentile\n(median, IQR)",
        )
        for _a in _g.axes.flat:
            plt.setp(
                _a.get_xticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
        _g.legend.set_title("Subgrid")
    return


@app.cell
def plot_fitness_detection(fitness_detection, mutation_order, pop_size_order):
    for _norm in "all", "epoch100":
        with tp.teed(
            sns.catplot,
            data=fitness_detection[
                fitness_detection["normalization"] == _norm
            ].astype({"detected": float}),
            x="mutation",
            y="detected",
            order=mutation_order,
            hue="pop_size",
            hue_order=pop_size_order,
            palette="viridis",
            col="stat",
            row="subset",
            kind="point",
            errorbar=("ci", 95),
            n_boot=1000,
            seed=SEED,
            dodge=0.4,
            height=2.4,
            aspect=2.6,
            teeplot_outattrs={"normalization": _norm},
            teeplot_outexclude=[
                "palette",
                "hue_order",
                "dodge",
                "order",
                "seed",
            ],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            gap_points(*_g.axes.flat, _g.legend)
            _g.refline(y=ALPHA, color="gray", linestyle="--")
            _g.set(ylim=(-0.05, 1.05))
            _g.set_axis_labels(
                "Mutation (site class and effect size)",
                "Detection Rate\n(fitness screen)",
            )
            for _a in _g.axes.flat:
                plt.setp(
                    _a.get_xticklabels(),
                    rotation=30,
                    ha="right",
                    rotation_mode="anchor",
                )
            _g.legend.set_title("Subgrid")
            _g.figure.suptitle(f"normalization={_norm}", y=1.02)
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_detectability():
    mo.md(f"""
    ## Detectability Summary

    Fraction of populations in which each screen detects each site at $\\alpha = {ALPHA}$, by population size (rows) and site class and effect size (columns).
    The `NEU` column gives the false positive rate over neutral control sites, and should sit near {ALPHA}.
    Fitness screens shown use `epoch100` normalization.

    Note that sampled fossil genomes per population grow with subgrid size (fossils were sampled with an equal budget per fossil layer and subgrid), so population size and sample size are confounded here.
    The full grid is a single population, so its detection rates are each 0 or 1.
    """)
    return


@app.cell
def tabulate_detectability(fitness_detection, freq_screen):
    _screens = {
        "frequency": freq_screen,
        **{
            f"fitness {_stat} {_subset}": fitness_detection[
                (fitness_detection["normalization"] == "epoch100")
                & (fitness_detection["stat"] == _stat)
                & (fitness_detection["subset"] == _subset)
            ]
            for _stat in ("carriers", "clade_duration", "num_pes")
            for _subset in ("all", "inner_f75")
        },
    }
    detectability = pd.concat(
        [
            _df.groupby(["pop_size", "mutation"], as_index=False)
            .agg(
                detection_rate=("detected", "mean"),
                num_screened=("detected", "size"),
            )
            .assign(screen=_screen)
            for _screen, _df in _screens.items()
        ],
        ignore_index=True,
    )
    detectability.pivot_table(
        index=["screen", "pop_size"],
        columns="mutation",
        values="detection_rate",
        sort=False,
    )
    return (detectability,)


@app.cell
def plot_detectability(detectability, mutation_order, pop_size_order):
    for _screen, _df in detectability.groupby("screen", sort=False):
        _pivot = _df.pivot(
            index="pop_size", columns="mutation", values="detection_rate"
        ).reindex(index=pop_size_order[::-1], columns=mutation_order)
        with tp.teed(
            sns.heatmap,
            _pivot,
            annot=True,
            fmt=".2f",
            cmap="rocket_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Detection Rate"},
            teeplot_outattrs={"screen": _screen.replace(" ", "-")},
            teeplot_outexclude=["annot", "fmt", "cmap", "vmin", "vmax"],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _ax:
            _ax.set_xlabel("Mutation (site class and effect size)")
            _ax.set_ylabel("Subgrid")
            _ax.set_title(_screen)
            plt.setp(
                _ax.get_xticklabels(),
                rotation=40,
                ha="right",
                rotation_mode="anchor",
            )
            _ax.tick_params(axis="y", rotation=0)
            _ax.figure.set_size_inches(10, 2.4)
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_by_kind():
    mo.md(f"""
    ## Detectability by Site Class

    The same screen results, recast with effect size on the x axis, site class as color, and subgrid size as columns.
    Neutral control sites appear as their own x position.
    Error bars are 95% bootstrap confidence intervals over populations; the dashed line marks $\\alpha = {ALPHA}$.
    Fitness screens shown use `epoch100` normalization.
    """)
    return


@app.cell
def tabulate_by_kind(fitness_detection, freq_screen, kind_labels):
    def _split_mutation(df):
        _parts = df["mutation"].str.extract(r"^(.+) s=(\S+)$")
        return df.assign(
            kind=_parts[0].fillna(kind_labels["NEU"]),
            effect_size=_parts[1].fillna("neu").map(effect_pct),
        )

    by_kind = pd.concat(
        [
            freq_screen.assign(
                screen="frequency", p_value=freq_screen["poisson_sf_p"]
            ),
            *(
                fitness_detection[
                    (fitness_detection["normalization"] == "epoch100")
                    & (fitness_detection["stat"] == _stat)
                    & (fitness_detection["subset"] == _subset)
                ].assign(
                    screen=f"fitness {_stat} {_subset}",
                    p_value=lambda df: df["p_mw_less"],
                )
                for _stat in ("carriers", "clade_duration", "num_pes")
                for _subset in ("all", "inner_f75")
            ),
        ],
        ignore_index=True,
    )[
        [
            "screen",
            "pop_code",
            "pop_size",
            "site",
            "mutation",
            "detected",
            "site_mean_pct",
            "p_value",
        ]
    ].pipe(
        _split_mutation
    )
    kind_order = [
        kind_labels[_kind]
        for _kind in ("NEU", "BENwg", "DELbg", "MLS")
        if kind_labels[_kind] in set(by_kind["kind"])
    ]
    effect_size_order = ["neu"] + sorted(
        set(by_kind["effect_size"]) - {"neu"}, key=float
    )
    kind_palette = {
        kind_labels[_kind]: ETI_PALETTE[_key][1]
        for _kind, _key in (
            ("NEU", "neutral"),
            ("BENwg", "BENwg"),
            ("DELbg", "DELbg"),
            ("MLS", "MLS"),
        )
    }
    freq_by_kind = freq_screen.pipe(_split_mutation)
    by_kind.groupby(["screen", "pop_size", "kind", "effect_size"], sort=False)[
        "detected"
    ].mean().unstack("pop_size")
    return by_kind, effect_size_order, freq_by_kind, kind_order, kind_palette


@app.cell
def plot_detection_by_kind(
    by_kind,
    effect_size_order,
    kind_order,
    kind_palette,
    pop_size_order,
):
    for _screen, _df in by_kind.groupby("screen", sort=False):
        with tp.teed(
            sns.catplot,
            data=_df.astype({"detected": float}),
            x="effect_size",
            y="detected",
            order=effect_size_order,
            hue="kind",
            hue_order=kind_order,
            palette=kind_palette,
            col="pop_size",
            col_order=pop_size_order,
            col_wrap=2,
            kind="point",
            errorbar=("ci", 95),
            n_boot=1000,
            seed=SEED,
            dodge=0.3,
            height=1.5,
            aspect=1.1,
            teeplot_outattrs={"screen": _screen.replace(" ", "-")},
            teeplot_outexclude=[
                "palette",
                "hue_order",
                "col_order",
                "col_wrap",
                "dodge",
                "order",
                "seed",
            ],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            gap_points(*_g.axes.flat, _g.legend)
            _g.refline(y=ALPHA, color="gray", linestyle="--")
            _g.set(ylim=(-0.05, 1.05))
            _g.set_axis_labels("", f"Detection Rate\n({_screen})")
            _g.set_titles(col_template="subgrid {col_name}")
            for _a in _g.axes.flat:
                plt.setp(
                    _a.get_xticklabels(),
                    rotation=90,
                    ha="right",
                    va="center",
                    rotation_mode="anchor",
                )
            corner_xlabel(_g, right_legend(_g))
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_fitness_pct_by_kind():
    mo.md(r"""
    ## MDC Size and Duration by Site Class

    The fitness screen's underlying measurement: each site's mean MDC percentile for clade size (`carriers`) and clade duration, ranked against background (neutral) MDCs within the same population and 100-update origin epoch.
    All MDCs are included.
    The dashed line at 50 marks parity with background; lower values mean smaller or shorter-lived clades.
    Points are medians over populations, with error bars spanning the interquartile range.
    """)
    return


@app.cell
def plot_fitness_pct_by_kind(
    by_kind,
    effect_size_order,
    kind_order,
    kind_palette,
    pop_size_order,
):
    _stat_names = {
        "carriers": "Clade Size",
        "clade_duration": "Clade Duration",
        "num_pes": "Clade PE Count",
    }
    # all MDCs subset only
    _fitness = by_kind[
        by_kind["screen"].str.startswith("fitness ")
        & by_kind["screen"].str.endswith(" all")
    ]
    for _screen, _df in _fitness.groupby("screen", sort=False):
        _stat = _screen.split(" ")[1]
        with tp.teed(
            sns.catplot,
            data=_df.dropna(subset=["site_mean_pct"]),
            x="effect_size",
            y="site_mean_pct",
            order=effect_size_order,
            hue="kind",
            hue_order=kind_order,
            palette=kind_palette,
            col="pop_size",
            col_order=pop_size_order,
            col_wrap=2,
            kind="point",
            estimator="median",
            errorbar=("pi", 50),
            dodge=0.3,
            height=1.5,
            aspect=1.1,
            teeplot_outattrs={"screen": _screen.replace(" ", "-")},
            teeplot_outexclude=[
                "palette",
                "hue_order",
                "col_order",
                "col_wrap",
                "dodge",
                "order",
            ],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            gap_points(*_g.axes.flat, _g.legend)
            _g.refline(y=50, color="gray", linestyle="--")
            _g.set_axis_labels(
                "",
                f"{_stat_names[_stat]}\nPercentile\n(median, IQR)",
            )
            _g.set_titles(col_template="subgrid {col_name}")
            for _a in _g.axes.flat:
                plt.setp(
                    _a.get_xticklabels(),
                    rotation=90,
                    ha="right",
                    va="center",
                    rotation_mode="anchor",
                )
            corner_xlabel(_g, right_legend(_g))
        plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_pvalue_by_kind():
    mo.md(f"""
    ## Detection Strength by Site Class

    Screen p-values on a log scale, descending from $p = 1$ at top so stronger evidence plots lower.
    Points are medians over populations, with error bars spanning the interquartile range; the dashed line marks $\\alpha = {ALPHA}$.
    Populations where a site produced no MDCs (so the screen could not run) count as $p = 1$, and p-values below $10^{{-9}}$ are clipped to $10^{{-9}}$.
    Fitness screens shown use all MDCs and `epoch100` normalization.
    """)
    return


@app.cell
def plot_pvalue_by_kind(
    by_kind,
    effect_size_order,
    kind_order,
    kind_palette,
    pop_size_order,
):
    _screen_names = {
        "frequency": "MDC Frequency",
        "fitness carriers all": "Clade Size",
        "fitness clade_duration all": "Clade Duration",
        "fitness num_pes all": "Clade PE Count",
    }
    for _screen, _name in _screen_names.items():
        _df = by_kind[by_kind["screen"] == _screen].assign(
            p_value=lambda df: df["p_value"].fillna(1.0).clip(lower=1e-9),
        )
        with tp.teed(
            sns.catplot,
            data=_df,
            x="effect_size",
            y="p_value",
            order=effect_size_order,
            hue="kind",
            hue_order=kind_order,
            palette=kind_palette,
            col="pop_size",
            col_order=pop_size_order,
            col_wrap=2,
            kind="point",
            estimator="median",
            errorbar=("pi", 50),
            dodge=0.3,
            height=1.5,
            aspect=1.1,
            log_scale=(False, True),
            teeplot_outattrs={"screen": _screen.replace(" ", "-")},
            teeplot_outexclude=[
                "palette",
                "hue_order",
                "col_order",
                "col_wrap",
                "dodge",
                "order",
                "log_scale",
            ],
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as _g:
            gap_points(*_g.axes.flat, _g.legend)
            _g.refline(y=ALPHA, color="gray", linestyle="--")
            # p = 1 at top, descending to the 1e-9 clip floor, with a
            # little padding so edge markers aren't cut off
            _g.set(
                ylim=(10**-9.4, 10**0.2),
                yticks=[1, 1e-3, 1e-6, 1e-9],
            )
            _g.set_axis_labels("", f"p-value\n({_name})")
            _g.set_titles(col_template="subgrid {col_name}")
            for _a in _g.axes.flat:
                plt.setp(
                    _a.get_xticklabels(),
                    rotation=90,
                    ha="right",
                    va="center",
                    rotation_mode="anchor",
                )
            corner_xlabel(_g, right_legend(_g))
        plt.close("all")
    return


@app.cell
def plot_enrichment_by_kind(
    effect_size_order,
    freq_by_kind,
    kind_order,
    kind_palette,
    pop_size_order,
):
    with tp.teed(
        sns.catplot,
        data=freq_by_kind,
        x="effect_size",
        y="fold_enrichment",
        order=effect_size_order,
        hue="kind",
        hue_order=kind_order,
        palette=kind_palette,
        col="pop_size",
        col_order=pop_size_order,
        col_wrap=2,
        kind="point",
        estimator="median",
        errorbar=("pi", 50),
        dodge=0.3,
        height=1.5,
        aspect=1.1,
        teeplot_outexclude=[
            "palette",
            "hue_order",
            "col_order",
            "col_wrap",
            "dodge",
            "order",
        ],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _g:
        gap_points(*_g.axes.flat, _g.legend)
        _g.refline(y=1, color="gray", linestyle="--")
        _g.set_axis_labels("", "MDC Enrichment\n(median, IQR)")
        _g.set_titles(col_template="subgrid {col_name}")
        for _a in _g.axes.flat:
            plt.setp(
                _a.get_xticklabels(),
                rotation=90,
                ha="right",
                va="center",
                rotation_mode="anchor",
            )
        corner_xlabel(_g, right_legend(_g))
    plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_paired_freq_screen():
    mo.md(f"""
    ## Paired Mutation-frequency Screen: MLS vs. DELbg

    MLS and between-group deleterious (`DELbg`) sites at the same effect size share the same recolonization (clobber) cost, so any difference in their MDC counts comes from the MLS site's within-population advantage.
    Within each population, for each effect size, this screen tests whether the MLS site gives rise to more MDCs than the `DELbg` site.
    Under the null of equal rates, given $n = k_\\mathrm{{MLS}} + k_\\mathrm{{del}}$ MDCs in total, $k_\\mathrm{{MLS}} \\sim \\mathsf{{Binom}}(n, 1/2)$, so significance is the one-sided binomial survival probability of observing at least $k_\\mathrm{{MLS}}$ (the conditional exact test comparing two Poisson counts).
    Populations with no MDCs at either site count as not detected.
    As a null calibration, neutral sites are paired off and screened the same way.
    Error bars are 95% bootstrap confidence intervals over populations; the dashed line marks $\\alpha = {ALPHA}$.
    """)
    return


@app.cell
def calc_paired_freq_screen(
    background_sites,
    freq_screen,
    kind_labels,
    site_labels,
):
    _counts = freq_screen.pivot(
        index="pop_code", columns="site", values="mdc_count"
    )
    _site_of = {label: site for site, label in site_labels.items()}
    _pairs = [
        (
            label.split("=")[1],
            _site,
            _site_of[f"{kind_labels['DELbg']} s={label.split('=')[1]}"],
        )
        for _site, label in site_labels.items()
        if label.startswith(kind_labels["MLS"])
    ] + [
        ("neu", _a, _b)
        for _a, _b in zip(background_sites[::2], background_sites[1::2])
    ]
    paired_freq_screen = pd.concat(
        [
            pd.DataFrame(
                {
                    "pop_code": _counts.index,
                    "effect_size": effect_pct(_effect),
                    "kind": (
                        (
                            kind_labels["NEU"]
                            if _effect == "neu"
                            else kind_labels["MLS"]
                        )
                    ),
                    "site": _site,
                    "comparator_site": _comparator,
                    "k_site": _counts[_site].to_numpy(),
                    "k_comparator": _counts[_comparator].to_numpy(),
                },
            )
            for _effect, _site, _comparator in _pairs
        ],
        ignore_index=True,
    ).merge(
        freq_screen[["pop_code", "pop_size"]].drop_duplicates(),
        on="pop_code",
    )
    _n = paired_freq_screen["k_site"] + paired_freq_screen["k_comparator"]
    paired_freq_screen["binom_sf_p"] = np.where(
        _n > 0,
        scipy_stats.binom.sf(paired_freq_screen["k_site"] - 1, _n, 0.5),
        np.nan,
    )
    paired_freq_screen["detected"] = paired_freq_screen["binom_sf_p"] < ALPHA
    paired_freq_screen.groupby(
        ["pop_size", "kind", "effect_size"], sort=False
    ).agg(
        num_populations=("pop_code", "nunique"),
        k_site_mean=("k_site", "mean"),
        k_comparator_mean=("k_comparator", "mean"),
        detection_rate=("detected", "mean"),
    ).unstack(
        "pop_size"
    )
    return (paired_freq_screen,)


@app.cell
def plot_paired_freq_detection(
    effect_size_order,
    kind_labels,
    kind_palette,
    paired_freq_screen,
    pop_size_order,
):
    with tp.teed(
        sns.catplot,
        data=paired_freq_screen.astype({"detected": float}),
        x="effect_size",
        y="detected",
        order=effect_size_order,
        hue="kind",
        hue_order=[kind_labels["NEU"], kind_labels["MLS"]],
        palette=kind_palette,
        col="pop_size",
        col_order=pop_size_order,
        col_wrap=2,
        kind="point",
        errorbar=("ci", 95),
        n_boot=1000,
        seed=SEED,
        dodge=0.3,
        height=1.5,
        aspect=1.1,
        teeplot_outattrs={"screen": "paired-frequency"},
        teeplot_outexclude=[
            "palette",
            "hue_order",
            "col_order",
            "col_wrap",
            "dodge",
            "order",
            "seed",
        ],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _g:
        gap_points(*_g.axes.flat, _g.legend)
        _g.refline(y=ALPHA, color="gray", linestyle="--")
        _g.set(ylim=(-0.05, 1.05))
        _g.set_axis_labels(
            "",
            f"Detection Rate\nvs. {kind_labels['DELbg']}",
        )
        _g.set_titles(col_template="subgrid {col_name}")
        for _a in _g.axes.flat:
            plt.setp(
                _a.get_xticklabels(),
                rotation=90,
                ha="right",
                va="center",
                rotation_mode="anchor",
            )
        corner_xlabel(_g, right_legend(_g))
    plt.close("all")
    return


@app.cell(hide_code=True)
def delimit_del_enrichment():
    mo.md(r"""
    MDC fold enrichment normalized to the between-group deleterious (`DELbg`) site at the same effect size (rather than to neutral sites), i.e., $k_\mathrm{site} / k_\mathrm{DELbg}$ within each population.
    MLS sites above 1 reflect their within-population advantage over an otherwise-equal recolonization cost.
    As a null reference, neutral sites are paired off and normalized to each other.
    Populations with no MDCs at the normalizing site are omitted.
    Points give medians and error bars interquartile ranges over populations.
    """)
    return


@app.cell
def calc_del_enrichment(
    background_sites,
    freq_screen,
    kind_labels,
    kind_order,
    site_labels,
):
    _counts = freq_screen.pivot(
        index="pop_code", columns="site", values="mdc_count"
    )
    _site_of = {label: site for site, label in site_labels.items()}
    _pairs = [
        (
            _kind,
            _effect,
            _site,
            _site_of[f"{kind_labels['DELbg']} s={_effect}"],
        )
        for _site, _label in site_labels.items()
        if _label.startswith((kind_labels["MLS"], kind_labels["BENwg"]))
        for _kind, _effect in [_label.split(" s=")]
    ] + [
        (kind_labels["NEU"], "neu", _a, _b)
        for _a, _b in zip(background_sites[::2], background_sites[1::2])
    ]
    del_enrichment = pd.concat(
        [
            pd.DataFrame(
                {
                    "pop_code": _counts.index,
                    "kind": _kind,
                    "effect_size": effect_pct(_effect),
                    "site": _site,
                    "k_site": _counts[_site].to_numpy(),
                    "k_del": _counts[_norm].to_numpy(),
                },
            )
            for _kind, _effect, _site, _norm in _pairs
        ],
        ignore_index=True,
    ).merge(
        freq_screen[["pop_code", "pop_size"]].drop_duplicates(),
        on="pop_code",
    )
    del_enrichment = del_enrichment[del_enrichment["k_del"] > 0].copy()
    del_enrichment["fold_enrichment_del"] = (
        del_enrichment["k_site"] / del_enrichment["k_del"]
    )
    del_kind_order = [_k for _k in kind_order if _k != kind_labels["DELbg"]]
    del_enrichment.groupby(
        ["pop_size", "kind", "effect_size"], sort=False
    ).agg(
        num_populations=("pop_code", "nunique"),
        fold_enrichment_del_median=("fold_enrichment_del", "median"),
    ).unstack(
        "pop_size"
    )
    return del_enrichment, del_kind_order


@app.cell
def plot_del_enrichment(
    del_enrichment,
    del_kind_order,
    effect_size_order,
    kind_labels,
    kind_palette,
    pop_size_order,
):
    with tp.teed(
        sns.catplot,
        data=del_enrichment,
        x="effect_size",
        y="fold_enrichment_del",
        order=effect_size_order,
        hue="kind",
        hue_order=del_kind_order,
        palette=kind_palette,
        col="pop_size",
        col_order=pop_size_order,
        col_wrap=2,
        kind="point",
        estimator="median",
        errorbar=("pi", 50),
        dodge=0.3,
        height=1.5,
        aspect=1.1,
        teeplot_outattrs={"normalization": "DELbg"},
        teeplot_outexclude=[
            "palette",
            "hue_order",
            "col_order",
            "col_wrap",
            "dodge",
            "order",
        ],
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _g:
        gap_points(*_g.axes.flat, _g.legend)
        _g.refline(y=1, color="gray", linestyle="--")
        _g.set(yscale="log")
        _g.set_axis_labels(
            "",
            f"MDC Enrichment\nvs. {kind_labels['DELbg']}\n(median, IQR)",
        )
        _g.set_titles(col_template="subgrid {col_name}")
        for _a in _g.axes.flat:
            plt.setp(
                _a.get_xticklabels(),
                rotation=90,
                ha="right",
                va="center",
                rotation_mode="anchor",
            )
        corner_xlabel(_g, right_legend(_g))
    plt.close("all")
    return


if __name__ == "__main__":
    app.run()
