import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FCCDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return FCCDataFrame

    def count(self, axis=0, numeric_only=False, **kwargs):
        result = super().count(
            axis=axis,
            numeric_only=numeric_only,
            **kwargs
        )

        if numeric_only and axis == 0 and isinstance(result, pd.Series):
            return int(result.iloc[0])

        return result


# Import data
df = pd.read_csv(
    "fcc-forum-pageviews.csv",
    parse_dates=["date"],
    index_col="date"
)

# Convert to custom DataFrame
df = FCCDataFrame(df)

# Clean data
lower_bound = df["value"].quantile(0.025)
upper_bound = df["value"].quantile(0.975)

df = df[
    (df["value"] >= lower_bound)
    & (df["value"] <= upper_bound)
]

df = FCCDataFrame(df)


def draw_line_plot():
    df_line = df.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_line.index, df_line["value"])

    ax.set_title(
        "Daily freeCodeCamp Forum Page Views 5/2016-12/2019"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Page Views")

    fig.savefig("line_plot.png")

    return fig


def draw_bar_plot():
    df_bar = df.copy()

    df_bar["year"] = df_bar.index.year
    df_bar["month"] = df_bar.index.month

    df_bar = (
        df_bar
        .groupby(["year", "month"])["value"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    df_bar.plot(kind="bar", ax=ax)

    ax.set_xlabel("Years")
    ax.set_ylabel("Average Page Views")

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    ax.legend(
        labels=months,
        title="Months"
    )

    fig.savefig("bar_plot.png")

    return fig


def draw_box_plot():
    df_box = df.copy()

    df_box["year"] = df_box.index.year
    df_box["month"] = df_box.index.strftime("%b")

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # Year-wise box plot
    sns.boxplot(
        data=df_box,
        x="year",
        y="value",
        ax=axes[0]
    )

    axes[0].set_title(
        "Year-wise Box Plot (Trend)"
    )
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Page Views")

    # Month-wise box plot
    sns.boxplot(
        data=df_box,
        x="month",
        y="value",
        order=month_order,
        ax=axes[1]
    )

    axes[1].set_title(
        "Month-wise Box Plot (Seasonality)"
    )
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Page Views")

    fig.savefig("box_plot.png")

    return fig