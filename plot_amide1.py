import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Recreate amide I schematic
# -----------------------------
x = np.linspace(1700, 1600, 1000)

def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Broad Amide I envelope
amide = (
    gaussian(x, 1655, 32, 1.00) +
    gaussian(x, 1685, 18, 0.18)
)

# Secondary structure components
beta_turn  = gaussian(x, 1672, 12, 0.3)
alpha_helix = gaussian(x, 1654, 14, 0.95)
beta_sheet = gaussian(x, 1625, 6, 0.40)
# beta_sheet_left = gaussian(x, 1692, 7, 0.10)

# Plot
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

# Fill component regions
ax.fill_between(x, beta_turn, color="#E993D2", alpha=0.75, edgecolor="#E993D2", linewidth=1.2)
ax.fill_between(x, alpha_helix, color="#E7D7D9", alpha=0.75, edgecolor="#E7D7D9", linewidth=1.2)
ax.fill_between(x, beta_sheet, color="#B98F8A", alpha=0.85, edgecolor="#B98F8A", linewidth=1.2)
# ax.fill_between(x, beta_sheet_left, color="#b7d8ff", alpha=0.85, edgecolor="#6da3d6", linewidth=1.2)

# Broad amide I line
ax.plot(x, amide, color="gray", linewidth=2)

# Labels
ax.text(1655, amide.max() + 0.03, "Amide I", ha="center", va="bottom", fontsize=11)
ax.text(1672, 0.20, "TURNS", ha="center", va="center", fontsize=9)
ax.text(1654, 0.67, r"$\alpha$-HELIX", ha="center", va="center", fontsize=9)
ax.text(1625, 0.3, r"$\beta$-SHEET", ha="center", va="center", fontsize=9)
# ax.text(1692, 0.035, r"$\beta$-SHEET", ha="center", va="center", fontsize=8)

# Axes formatting
ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=10)
ax.set_xlim(1600, 1700)   # IR style: decreasing to the right
ax.set_ylim(0, 1.15)
ax.set_xticks([1600, 1620, 1640, 1660, 1680, 1700])
ax.set_yticks([])

# Style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("gray")
ax.tick_params(axis='x', colors='gray', labelsize=9)

plt.tight_layout()
plt.show()