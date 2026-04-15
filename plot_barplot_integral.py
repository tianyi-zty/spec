import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

csv_path = r"../res/04082026_col1+4/CAF2/org/spectrum_fitting_results/subspectrum_fitting_results.csv"

# =========================================================
# Read CSV robustly
# Case 1: file already has header
# Case 2: file is tab-separated without header (like your pasted content)
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
    # File Name | Amplitude | Center Value| Sigma | Integral Value
    if df.shape[1] >= 5:
        df = df.iloc[:, :5].copy()
        df.columns = ["File Name", "Amplitude", "Center Value", "Sigma", "Integral Value"]
    else:
        raise ValueError("CSV format not recognized. Need at least 5 columns.")

# =========================================================
# Parse filename correctly
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

# numeric columns
for col in ["Amplitude", "Center Value", "Sigma", "Integral Value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Integral Value"]).copy()

# =========================================================
# Keep original within-spectrum order
# Each LMT spectrum has multiple fitted peaks; define Component by row order
# =========================================================
df["Component"] = df.groupby("LMT").cumcount() + 1

# =========================================================
# Keep only target groups
# =========================================================
g1, g2 = "9109", "9505"
df = df[df["Group"].isin([g1, g2])].copy()

# check available components in both groups
comp_counts = df.groupby(["Group", "Component"]).size().unstack("Group")
common_components = sorted(
    set(df.loc[df["Group"] == g1, "Component"]) &
    set(df.loc[df["Group"] == g2, "Component"])
)

components = np.array(common_components)

if len(components) == 0:
    raise ValueError("No common components found between the two groups.")

# =========================================================
# Average per component
# =========================================================
means = df.groupby(["Group", "Component"])["Integral Value"].mean().unstack("Group")
sds   = df.groupby(["Group", "Component"])["Integral Value"].std().unstack("Group")

y1 = means.loc[components, g1].to_numpy()
y2 = means.loc[components, g2].to_numpy()
e1 = sds.loc[components, g1].fillna(0).to_numpy()
e2 = sds.loc[components, g2].fillna(0).to_numpy()

# =========================================================
# p-values per component
# =========================================================
pvals = []

for comp in components:
    a = df[(df["Group"] == g1) & (df["Component"] == comp)]["Integral Value"].dropna()
    b = df[(df["Group"] == g2) & (df["Component"] == comp)]["Integral Value"].dropna()

    if len(a) < 2 or len(b) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(a, b, equal_var=False)
    pvals.append(1 if np.isnan(p) else p)

def stars(p):
    if p < 0.01:
        return "****"
    if p < 0.05:
        return "***"
    if p < 0.1:
        return "**"
    if p < 0.2:
        return "*"
    return "ns"

# =========================================================
# Plot
# =========================================================
x = np.arange(len(components))
w = 0.38

plt.figure(figsize=(12, 5))
plt.bar(x - w/2, y1, w, yerr=e1, capsize=4, label=g1)
plt.bar(x + w/2, y2, w, yerr=e2, capsize=4, label=g2)

plt.xticks(x, components)
plt.xlabel("Component")
plt.ylabel("Integral Value (mean ± SD)")
plt.title(f"Average integral per component: collagen1:4 91:09 vs collagen1:4 95:05")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)

# significance annotations
ymax = np.maximum(y1 + e1, y2 + e2)
pad = max(ymax.max() * 0.05, 0.1)

for i, p in enumerate(pvals):
    y = ymax[i] + pad
    x1, x2 = x[i] - w/2, x[i] + w/2
    plt.plot([x1, x1, x2, x2], [y, y + pad, y + pad, y], lw=1, c="black")
    plt.text((x1 + x2) / 2, y + pad, stars(p), ha="center", va="bottom")

legend_text = (
    "Significance:\n"
    "ns   p ≥ 0.2\n"
    "*    p < 0.2\n"
    "**   p < 0.1\n"
    "***  p < 0.05\n"
    "**** p < 0.01"
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
# Print summary
# =========================================================
print("\nParsed groups:")
print(df[["File Name", "Group", "Replicate", "LMT"]].drop_duplicates().head(10))

print("\nP-values:")
for comp, p in zip(components, pvals):
    print(f"Component {comp:2d}: {p:.4g} ({stars(p)})")