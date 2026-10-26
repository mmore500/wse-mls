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

    OSF_SLUG = "6pa9j"
    NUM_SITES = 64
    ALPHA = 0.05
    NUM_CONTROL_SITES = 8
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
        effect if effect in ("neu", "neutral") else f"{float(effect) * 100:g}"
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
    # MLS Spectrum Subgrid Fossils: Detectability by Population Size and Effect Size

    Applies the phylogeny-based multilevel selection (MLS) screen from *Extending a Phylogeny-based Method for Detecting Signatures of Multi-level Selection for Applications in Artificial Life* (ISAL 2025, <https://doi.org/10.1162/ISAL.a.916>; code at <https://github.com/mmore500/multilevel-selection-concept>) to wafer-scale evolution data, following `2026-10-15-mls-strong-fossils`.

    Data were generated by [`wse-async-ga` `wsclust/2026-10-15-mls-spectrum-subgrid-fossils`](https://github.com/mmore500/wse-async-ga/tree/mlsnk-genome/wsclust/2026-10-15-mls-spectrum-subgrid-fossils) and are hosted at <https://osf.io/{OSF_SLUG}>.
    That file is the minified hstrat-reconstructed phylogeny (`06-minify-phylo` output), holding one independently reconstructed tree per run.

    Runs use genome flavor `genome_mlsspectrum_2026_10_15`, which carries eight MLS sites with effect sizes $s_i \\in [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]$: carriers are clobbered as immigrants into an extinct tile population with probability $s_i$, but have within-population advantage $s'_i = (1 + s_i)^{{\\log_2(N) / 25}} - 1$.
    The other 56 genome sites are neutral.
    Site classes are labeled with ETI evaluation bar notation, giving each trait's between-group effect above and within-group effect below, and $\\varnothing$ for no effect: MLS $|^{{-}}_{{+}}$ and neutral sites NEU $|^{{\\varnothing}}_{{\\varnothing}}$.
    Runs sweep subgrid size (27, 81, 243, and 0, i.e., the full grid), where subgrids are blocks of tiles between which migration is disabled.

    Each subgrid is an isolated metapopulation, so populations are analyzed independently rather than pooled: within each population, the screen asks whether each MLS site

    1. gives rise to more mutation-defined clades (MDCs) than expected for a neutral site (mutation-frequency screen), and
    2. gives rise to MDCs that are smaller and/or shorter-lived than neutral-site MDCs (clade-fitness screen).

    Detectability is the fraction of populations in which a screen is significant at $\\alpha = {ALPHA}$, broken down by population size and effect size.
    As a null calibration, the same screens are applied to {NUM_CONTROL_SITES} randomly chosen neutral control sites.
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
    data_path = osf_fetch(OSF_SLUG)
    return (data_path,)


@app.cell
def show_schema(data_path):
    pl.scan_parquet(data_path).collect_schema()
    return


@app.cell
def load_config(data_path):
    _lazy = pl.scan_parquet(data_path)
    _schema = _lazy.collect_schema()
    _config_cols = [
        name
        for name in _schema.names()
        if name.startswith(("ASYNC_GA_", "CEREBRASLIB_", "dstream_algo"))
        or re.match(r"^mls_(clobber|selection)_p_byte\d+_bit\d+$", name)
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
    config_df = (
        _lazy.filter(pl.col("clade_size") > 0)
        .select(_config_cols)
        .unique()
        .sort("nRowSubgrid")
        .collect()
    )
    assert config_df["config"].is_unique().all(), "expected one row per run"
    assert (config_df["nRowSubgrid"] == config_df["nColSubgrid"]).all()
    config_df.transpose(include_header=True)
    return (config_df,)


@app.cell
def tabulate_effect_sizes(config_df):
    # MLS site i is flag byte B, bit b (trait site 8B + b); its clobber
    # probability is effect size s_i (CEREBRASLIB_CLOBBER_IMMIGRANT_P = 1.0)
    _site_cols = [
        (int(B) * 8 + int(b), what, col)
        for col in config_df.columns
        for what, B, b in re.findall(
            r"^mls_(clobber|selection)_p_byte(\d+)_bit(\d+)$", col
        )
    ]
    effect_sizes = (
        pd.DataFrame(
            [
                {
                    "config": row["config"],
                    "site": site,
                    f"{what}_p": row[col],
                }
                for row in config_df.iter_rows(named=True)
                for site, what, col in _site_cols
            ],
        )
        .groupby(["config", "site"], as_index=False)
        .first()
        .merge(config_df.select("config", "tilePopSize").to_pandas())
    )
    # effect sizes are the same across runs
    assert (effect_sizes.groupby("site")["clobber_p"].nunique() == 1).all()
    effect_sizes = effect_sizes.rename(columns={"clobber_p": "effect_size"})
    focal_sites = sorted(effect_sizes["site"].unique().tolist())
    _focal_mask = sum(1 << site for site in focal_sites)
    assert (config_df["trait_focal"] == _focal_mask).all(), "focal mismatch"
    background_sites = [
        site for site in range(NUM_SITES) if site not in focal_sites
    ]
    control_sites = sorted(
        np.random.default_rng(SEED)
        .choice(background_sites, NUM_CONTROL_SITES, replace=False)
        .tolist()
    )
    site_effect_sizes = effect_sizes.groupby("site")["effect_size"].first()
    # ETI evaluation bar notation: between-group effect above, within-group
    # effect below, with null (crossed circle) for no effect
    kind_labels = {
        "MLS": r"MLS$|^{-}_{+}$",
        "DELbg": r"DEL$\mathit{bg}|^{-}_{\varnothing}$",
        "BENwg": r"BEN$\mathit{wg}|^{\varnothing}_{+}$",
        "NEU": r"NEU$|^{\varnothing}_{\varnothing}$",
    }
    site_labels = pd.Series(
        {
            site: (
                f"{kind_labels['MLS']} s={site_effect_sizes[site]:g}"
                if site in focal_sites
                else kind_labels["NEU"]
            )
            for site in range(NUM_SITES)
        },
    )
    mutation_order = [
        f"{kind_labels['MLS']} s={_s:g}"
        for _s in sorted(site_effect_sizes.unique(), reverse=True)
    ] + [kind_labels["NEU"]]
    effect_sizes
    return (
        background_sites,
        control_sites,
        focal_sites,
        kind_labels,
        mutation_order,
        site_effect_sizes,
        site_labels,
    )


@app.cell
def load_phylogeny(data_path):
    _df = pl.read_parquet(
        data_path,
        columns=[
            "ancestor_offset",
            "origin_time",
            "last_origin_time",
            "clade_size",
            "trait_present",
            "config",
            "layer",
            "subgrid",
        ],
    )
    # unminify: rows are in depth-first order, with id = row index
    phylo_raw = (
        _df.with_row_index("id")
        .with_columns(
            pl.col("id").cast(pl.Int64),
            ancestor_id=(
                pl.col("id").cast(pl.Int64)
                - pl.col("ancestor_offset").cast(pl.Int64)
            ),
            config=pl.col("config").cast(pl.String),
        )
        .drop("ancestor_offset")
        .to_pandas()
    )
    del _df
    # some roots lack origin_time; roots can be no later than the run start
    _no_time = phylo_raw["origin_time"].isna()
    assert (
        phylo_raw.loc[_no_time, "ancestor_id"] == phylo_raw.loc[_no_time, "id"]
    ).all()
    phylo_raw["origin_time"] = phylo_raw["origin_time"].fillna(-1)
    phylo_raw["last_origin_time"] = phylo_raw["last_origin_time"].fillna(
        phylo_raw["origin_time"]
    )
    phylo_raw.info()
    return (phylo_raw,)


@app.cell
def check_phylogeny(phylo_raw):
    _is_genome = phylo_raw["clade_size"] > 0
    assert (_is_genome == phylo_raw["trait_present"].notna()).all()
    assert (_is_genome == phylo_raw["subgrid"].notna()).all()
    assert phylo_raw["config"].notna().all()
    assert (phylo_raw["ancestor_id"] <= phylo_raw["id"]).all()
    _has_child = np.zeros(len(phylo_raw), dtype=bool)
    _has_child[phylo_raw["ancestor_id"]] = True
    _has_child[phylo_raw["ancestor_id"] == phylo_raw["id"]] = False
    # collapsed monomorphic clades and fossil genomes are leaves
    assert not (_has_child & _is_genome).any()
    "ok"
    return


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
def tabulate_tips(phylo_raw):
    _tips = phylo_raw[phylo_raw["clade_size"] > 0]
    _tips.groupby("config").agg(
        num_leaves=("id", "size"),
        num_genomes=("clade_size", "sum"),
        clade_size_mean=("clade_size", "mean"),
        clade_size_max=("clade_size", "max"),
        num_layers=("layer", "nunique"),
        origin_time_min=("origin_time", "min"),
        last_origin_time_max=("last_origin_time", "max"),
        negative_origin_frac=("last_origin_time", lambda x: (x < 0).mean()),
    ).T
    return


@app.cell
def describe_times(phylo_raw):
    phylo_raw.assign(is_genome=phylo_raw["clade_size"] > 0).groupby(
        "is_genome"
    )[["origin_time", "last_origin_time"]].describe().T.assign(
        dtype=str(phylo_raw["origin_time"].dtype),
    )
    return


@app.cell
def prune_phylogeny(phylo_raw):
    phylo_pruned = (
        phylo_raw.assign(
            extant=(phylo_raw["clade_size"] > 0)
            & (phylo_raw["last_origin_time"] >= 0),
        )
        .pipe(pfl.alifestd_to_working_format)
        .pipe(pfl.alifestd_prune_extinct_lineages_asexual, mutate=True)
        .pipe(pfl.alifestd_collapse_unifurcations, mutate=True)
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_mark_leaves, mutate=True)
        .reset_index(drop=True)
    )
    assert (phylo_pruned["is_leaf"] == phylo_pruned["extant"]).all()
    return (phylo_pruned,)


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


@app.cell
def split_populations(config_df, phylo_pruned):
    _config_codes = {
        config: i for i, config in enumerate(config_df["config"].to_list())
    }
    _is_leaf = phylo_pruned["is_leaf"].to_numpy()
    _pop_code = np.where(
        _is_leaf,
        phylo_pruned["config"].map(_config_codes).to_numpy(dtype=np.int64)
        * 2**32
        + phylo_pruned["subgrid"].fillna(0).to_numpy(dtype=np.int64),
        -1,
    )
    _pure = pfl.alifestd_mask_monomorphic_clades_asexual(
        phylo_pruned[["id", "ancestor_id"]],
        trait_mask=_is_leaf,
        trait_values=_pop_code,
    )["alifestd_mask_monomorphic_clades_asexual"].to_numpy()
    _ancestor_ids = phylo_pruned["ancestor_id"].to_numpy()
    _is_root = _ancestor_ids == phylo_pruned["id"].to_numpy()
    split_summary = pd.Series(
        {
            "num_nodes": len(phylo_pruned),
            "num_multipopulation_nodes": int((~_pure).sum()),
            "num_roots_before": int(_is_root.sum()),
            "num_roots_after": int(
                (_pure & (_is_root | ~_pure[_ancestor_ids])).sum()
            ),
        },
    ).to_frame("value")

    phylo_split = (
        phylo_pruned.assign(
            ancestor_id=np.where(
                _pure[_ancestor_ids], _ancestor_ids, phylo_pruned["id"]
            ),
            pop_code=_pop_code,
        )[_pure]
        .pipe(pfl.alifestd_to_working_format, mutate=True)
        .pipe(pfl.alifestd_mark_roots, mutate=True)
        .pipe(pfl.alifestd_mark_root_id, mutate=True)
        .reset_index(drop=True)
    )
    _root_pops = (
        phylo_split.loc[phylo_split["is_leaf"]]
        .groupby("root_id")["pop_code"]
        .agg(["first", "nunique"])
    )
    assert (_root_pops["nunique"] == 1).all()
    phylo_split["pop_code"] = phylo_split["root_id"].map(_root_pops["first"])
    assert (phylo_split["pop_code"] >= 0).all()
    phylo_split["config"] = phylo_split.groupby("pop_code")[
        "config"
    ].transform("first")
    phylo_split["subgrid"] = (phylo_split["pop_code"] % 2**32).astype(int)
    split_summary
    return (phylo_split,)


@app.cell
def tabulate_populations(config_df, phylo_split):
    _config = config_df.select(
        "config", "nRow", "nCol", "nRowSubgrid", "nColSubgrid", "tilePopSize"
    ).to_pandas()
    _leaves = phylo_split[phylo_split["is_leaf"]]
    populations = (
        _leaves.groupby("pop_code")
        .agg(
            config=("config", "first"),
            subgrid=("subgrid", "first"),
            num_leaves=("id", "size"),
            num_genomes=("clade_size", "sum"),
            num_roots=("root_id", "nunique"),
        )
        .reset_index()
        .merge(_config, on="config")
    )
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
    pop_size_order = (
        populations.sort_values("pop_tiles")["pop_size"].unique().tolist()
    )
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
    return pop_size_order, populations


@app.cell
def prep_phylogeny(phylo_split, populations):
    _full_pops = populations.loc[populations["is_full_size"], "pop_code"]
    phylo = (
        phylo_split[phylo_split["pop_code"].isin(_full_pops)]
        .pipe(pfl.alifestd_to_working_format)
        .pipe(pfl.alifestd_mark_roots, mutate=True)
        .pipe(pfl.alifestd_mark_root_id, mutate=True)
        .pipe(pfl.alifestd_mark_num_leaves_asexual, mutate=True)
        .reset_index(drop=True)
    )
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
    phylo.info()
    return (phylo,)


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
    Diamonds mark tips carrying any MLS (focal) mutation; circles mark tips carrying any neutral (background) mutation.
    """)
    return


@app.cell
def plot_trees(focal_sites, phylo, pop_size_order, populations):
    _focal_mask = np.uint64(sum(1 << site for site in focal_sites))
    _trait_present = phylo["trait_present"].to_numpy(dtype=np.uint64)
    _is_leaf = phylo["is_leaf"].to_numpy()
    _phylo = phylo.assign(
        has_mls=np.where(
            _is_leaf, (_trait_present & _focal_mask) != 0, np.nan
        ),
        has_neutral=np.where(
            _is_leaf, (_trait_present & ~_focal_mask) != 0, np.nan
        ),
    )
    _roots = _phylo.loc[_phylo["is_root"], ["id", "num_leaves", "pop_code"]]
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

        _subtree = (
            _phylo[_phylo["root_id"] == _root_id]
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
                [],
                [],
                "d",
                markersize=8,
                color="#6AB81C",
                label=r"MLS$|^{-}_{+}$",
            )
            (_h2,) = plt.plot(
                [],
                [],
                "o",
                markersize=8,
                color="#C5A3FF",
                label=r"NEU$|^{\varnothing}_{\varnothing}$",
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

    MDCs are identified for each of the 64 trait sites by Fitch maximum parsimony over leaf states (root state assumed absent), as in `2026-10-15-mls-strong-fossils`.
    Because each population's trees share no nodes with other populations', one screen over the whole forest is equivalent to screening each population separately.

    The screen uses phyloframe's `alifestd_screen_trait_defined_clades_fitch_asexual`, which requires phyloframe >= 0.12.2 to run in linear time at this data size.

    MDC size is measured as `carriers`, the number of descendant fossil genomes carrying the mutation; `num_genomes` counts all descendant fossil genomes.
    `carrier_frac` is the carrier fraction among descendant genomes, and `is_terminal` marks MDCs arising on a terminal branch to a single genome (MDC node is an uncollapsed leaf).
    `clade_duration` spans from MDC origin to its latest descendant fossil.
    """)
    return


@app.cell
def screen_mdcs(phylo):
    _work = phylo[["id", "ancestor_id"]].copy()
    _is_leaf = phylo["is_leaf"].to_numpy()
    _clade_size = phylo["clade_size"].fillna(0).to_numpy(dtype=np.int64)
    _trait_present = phylo["trait_present"].to_numpy(dtype=np.uint64)
    mdc_ids = {}
    mdc_carriers = {}
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
        mdc_ids[_site] = np.flatnonzero(_mask)
        mdc_carriers[_site] = _carriers[_mask]
    return mdc_carriers, mdc_ids


@app.cell
def tabulate_mdcs(
    control_sites,
    focal_sites,
    mdc_carriers,
    mdc_ids,
    phylo,
    populations,
    site_effect_sizes,
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
            )
            for site, ids in mdc_ids.items()
        ],
        ignore_index=True,
    )
    mdcs["is_terminal"] = mdcs["is_leaf"] & (mdcs["clade_size"] == 1)
    mdcs["carrier_frac"] = mdcs["carriers"] / mdcs["num_genomes"]
    mdcs["effect_size"] = mdcs["site"].map(site_effect_sizes)
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
        _ax.figure.set_size_inches(5, 2.6)
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
        _ax.figure.set_size_inches(5, 2.6)
    return


@app.cell(hide_code=True)
def delimit_fitness_screen():
    mo.md("""
    ## Clade-fitness Screen

    Within each population, each MDC's size (`carriers`) and clade duration are percentile-ranked (mean rank for ties) against comparator MDCs: the pooled MDCs of background (neutral) sites in the same population, drawn from the same MDC subset.
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
    stat_cols = ("carriers", "clade_duration")
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
        aspect=1.6,
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
            aspect=1.6,
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

    Fraction of populations in which each screen detects each site at $\\alpha = {ALPHA}$, by population size (rows) and effect size (columns).
    The NEU column gives the false positive rate over neutral control sites, and should sit near {ALPHA}.
    Fitness screens shown use `epoch100` normalization.

    Note that sampled fossil genomes per population grow with subgrid size (fossils were sampled with an equal budget per subgrid), so population size and sample size are confounded here.
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
            for _stat in ("carriers", "clade_duration")
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
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            _ax.tick_params(axis="y", rotation=0)
            _ax.figure.set_size_inches(6, 2.4)
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
            freq_screen.assign(screen="frequency"),
            *(
                fitness_detection[
                    (fitness_detection["normalization"] == "epoch100")
                    & (fitness_detection["stat"] == _stat)
                    & (fitness_detection["subset"] == _subset)
                ].assign(screen=f"fitness {_stat} {_subset}")
                for _stat in ("carriers", "clade_duration")
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
            height=1.8,
            aspect=1.6,
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
                    rotation=30,
                    ha="right",
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
            height=1.8,
            aspect=1.6,
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
                    rotation=30,
                    ha="right",
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
        height=1.8,
        aspect=1.6,
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
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
        corner_xlabel(_g, right_legend(_g))
    plt.close("all")
    return


if __name__ == "__main__":
    app.run()
