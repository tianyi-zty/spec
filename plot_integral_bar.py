import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

csv_path = r"../res/03232026_col1+4/CAF2/org/spectrum_fitting_results/subspectrum_fitting_results.csv"

# =========================================================
# Read CSV robustly
# Case 1: file already has header
# Case 2: file is tab-separated / mixed separator without header
# =========================================================
try:
    df = pd.read_csv(csv_path)
    if "File Name" not in df.columns:
        raise ValueError("Header not found, try no-header mode.")
except:
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|\t+|,",
        engine="python",
        header=None
    )

    # rename columns based on your pasted data structure
    # File Name | Amplitude | Center Value | Sigma | Integral Value
    if df.shape[1] >= 5:
        df = df.iloc[:, :5].copy()
        df.columns = ["File Name", "Amplitude", "Center Value", "Sigma", "Integral Value"]
    else:
        raise ValueError("CSV format not recognized. Need at least 5 columns.")

# =========================================================
# Parse filename
# Example:
# caf2_col1_4_8020_LMT_1_mean_spectrum.npy
# caf2_col1_LMT_3_mean_spectrum.npy
# =========================================================
def parse(fname):
    fname = str(fname).strip()
    m = re.match(r"(.+)_LMT_(\d+)_mean_spectrum(?:\.npy)?$", fname)
    if not m:
        return None, None, None
    group = m.group(1)
    replicate = m.group(2)
    lmt = f"{group}_LMT_{replicate}"
    return group, replicate, lmt

df[["Group", "Replicate", "LMT"]] = df["File Name"].apply(
    lambda x: pd.Series(parse(x))
)

df = df.dropna(subset=["Group", "Replicate", "LMT"]).copy()

# =========================================================
# Numeric columns
# =========================================================
for col in ["Amplitude", "Center Value", "Sigma", "Integral Value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Integral Value"]).copy()

# =========================================================
# Keep original within-spectrum peak order
# Each LMT spectrum has multiple fitted peaks; define Component by row order
# =========================================================
df["Component"] = df.groupby("LMT").cumcount() + 1

# =========================================================
# Use ALL groups found in the csv
# =========================================================
groups = sorted(df["Group"].dropna().unique().tolist())

if len(groups) < 2:
    raise ValueError("Need at least 2 groups in the CSV for one-way ANOVA.")

print("\nDetected groups:")
for g in groups:
    n_lmt = df.loc[df["Group"] == g, "LMT"].nunique()
    print(f"  {g}: {n_lmt} replicate spectra")

# =========================================================
# Keep only components that appear in ALL groups
# =========================================================
components_by_group = {
    g: set(df.loc[df["Group"] == g, "Component"].unique())
    for g in groups
}
common_components = sorted(set.intersection(*components_by_group.values()))

if len(common_components) == 0:
    raise ValueError("No common components found across all groups.")

components = np.array(common_components)

# =========================================================
# Mean / SD table
# rows = Component, columns = Group
# =========================================================
means = df.groupby(["Component", "Group"])["Integral Value"].mean().unstack("Group")
sds   = df.groupby(["Component", "Group"])["Integral Value"].std().unstack("Group")

# reorder columns
means = means.reindex(index=components, columns=groups)
sds   = sds.reindex(index=components, columns=groups)

# =========================================================
# One-way ANOVA per component across ALL groups
# =========================================================
pvals = []
fstats = []

for comp in components:
    arrays = []
    valid_group_names = []

    for g in groups:
        vals = df[
            (df["Group"] == g) &
            (df["Component"] == comp)
        ]["Integral Value"].dropna().values

        # ANOVA needs at least 2 observations in each included group
        if len(vals) >= 2:
            arrays.append(vals)
            valid_group_names.append(g)

    # require at least 2 groups with enough values
    if len(arrays) < 2:
        f_stat, p = np.nan, np.nan
    else:
        f_stat, p = stats.f_oneway(*arrays)

    fstats.append(f_stat)
    pvals.append(p)

def stars(p):
    if pd.isna(p):
        return "NA"
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# =========================================================
# Plot grouped bar chart for ALL groups
# =========================================================
n_comp = len(components)
n_groups = len(groups)

x = np.arange(n_comp)

# total width occupied by bars within each component
total_width = 0.82
bar_width = total_width / n_groups

# use tab20 automatically
cmap = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(n_groups)]

plt.figure(figsize=(max(12, n_comp * 1.3), 6))

all_bar_tops = []

for i, g in enumerate(groups):
    xpos = x - total_width/2 + i * bar_width + bar_width/2
    y = means[g].to_numpy()
    e = sds[g].fillna(0).to_numpy()

    plt.bar(
        xpos, y, bar_width,
        yerr=e, capsize=3,
        label=g,
        color=colors[i],
        alpha=0.9
    )

    all_bar_tops.append(y + e)

all_bar_tops = np.vstack(all_bar_tops)
ymax_each_comp = np.nanmax(all_bar_tops, axis=0)

# =========================================================
# ANOVA p-value annotation above each component
# =========================================================
y_global_max = np.nanmax(ymax_each_comp)
pad = max(y_global_max * 0.05, 0.1)

for i, (comp, p) in enumerate(zip(components, pvals)):
    y = ymax_each_comp[i] + pad
    label = f"{stars(p)}\np={p:.2e}" if not pd.isna(p) else "NA"
    plt.text(
        x[i], y, label,
        ha="center", va="bottom",
        fontsize=9, rotation=0
    )

plt.xticks(x, components)
plt.xlabel("Component")
plt.ylabel("Integral Value (mean ± SD)")
plt.title("Average integral per component across all groups\n(one-way ANOVA for each component)")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)

legend_text = (
    "One-way ANOVA per component\n"
    "ns   p ≥ 0.05\n"
    "*    p < 0.05\n"
    "**   p < 0.01\n"
    "***  p < 0.001\n"
    "**** p < 0.0001"
)

plt.gca().text(
    0.02, 0.98, legend_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.show()

# =========================================================
# Summary table
# =========================================================
summary_rows = []
for comp, f_stat, p in zip(components, fstats, pvals):
    row = {
        "Component": comp,
        "F_stat": f_stat,
        "p_value": p,
        "Significance": stars(p)
    }
    for g in groups:
        row[f"{g}_mean"] = means.loc[comp, g] if g in means.columns else np.nan
        row[f"{g}_sd"]   = sds.loc[comp, g] if g in sds.columns else np.nan
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("\nParsed groups / LMT:")
print(df[["File Name", "Group", "Replicate", "LMT"]].drop_duplicates().head(20))

print("\nANOVA summary:")
print(summary_df.to_string(index=False))

# optional: save results
summary_df.to_csv("anova_summary_all_groups.csv", index=False)
print("\nSaved: anova_summary_all_groups.csv")