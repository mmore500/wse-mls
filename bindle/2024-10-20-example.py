import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import pathlib

    import marimo as mo
    import pandas as pd
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark


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
def delimit_prep_data():
    mo.md("""
    ## Prep Data
    """)
    return


@app.cell
def load_data():
    df = sns.load_dataset("mpg")
    return (df,)


@app.cell
def describe_data(df):
    df.describe()
    return


@app.cell
def peek_data(df):
    pd.concat([df.head(), df.tail()])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Example Plot
    """)
    return


@app.cell
def _(df):
    with tp.teed(
        sns.relplot,
        data=df,
        x="horsepower",
        y="mpg",
        hue="origin",
        size="weight",
        alpha=0.5,
        height=6,
        legend=False,
        sizes=(40, 400),
        palette="muted",
        teeplot_outexclude="palette",
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as g:
        g.figure.set_size_inches(5, 2)
    return


@app.cell
def _():
    def test_dummy_true():
        assert True

    def test_dummy_arithmetic():
        assert 1 + 1 == 2

    return


if __name__ == "__main__":
    app.run()
