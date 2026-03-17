from Utils import Fred_MD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


fred_stationary = Fred_MD(
    start="1980-01-01", factors=1, factor_verbose=True, quarterly=True
)

# NBER recession dates
recessions = [
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# Extract first factor
scores = fred_stationary["PC1"]  # adjust key if different
factor1 = scores  # first column = PC1
dates = factor1.index

fig, ax = plt.subplots(figsize=(12, 5))

# Recession bands
for start, end in recessions:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e >= dates.min() and s <= dates.max():
        ax.axvspan(
            max(s, dates.min()),
            min(e, dates.max()),
            color="grey",
            alpha=0.3,
            linewidth=0,
        )

# Zero line
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

# Factor line
ax.plot(dates, factor1, color="#1F77B4", linewidth=1.2)

# Labels
ax.set_title("Latent Factor: FRED_MD — PC1", fontweight="bold", fontsize=13)
ax.set_subtitle = None  # not native in matplotlib, use text instead
ax.text(
    0.01,
    0.97,
    f"PC1 | start: 1980-01-01 | factors = 1",
    transform=ax.transAxes,
    fontsize=9,
    color="grey",
    va="top",
)
ax.set_ylabel("Factor Score (Z-scored)")
ax.set_xlabel("")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()

factor1.to_csv("FRED_QD_1.csv", header=["PC1"])
