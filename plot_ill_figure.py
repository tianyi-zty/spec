import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Wavenumber range from 950 to 1800 cm⁻¹ (left to right)
wavenumbers = np.linspace(950, 1800, 1000)[::-1]  # flip axis direction

# Create peaks for each biomolecule
abs_lipids = gaussian(wavenumbers, 1740, 20, 0.2) 
abs_proteins = gaussian(wavenumbers, 1650, 25, 1.3) + gaussian(wavenumbers, 1550, 25, 1.0)
abs_nucleic = gaussian(wavenumbers, 1230, 10, 0.5) + gaussian(wavenumbers, 1080, 20, 0.4)
abs_carb = gaussian(wavenumbers, 1050, 10, 0.3)

# Random baseline (low-frequency sine + random noise)
np.random.seed(42)  # for reproducibility
baseline = 0.1 * np.sin(np.linspace(0, 2*np.pi, len(wavenumbers))) + 0.01 * np.random.randn(len(wavenumbers))

# Final absorbance spectrum (sum + baseline)
total_abs = abs_lipids + abs_proteins + abs_nucleic + abs_carb + baseline

# Plotting
plt.figure(figsize=(12, 6))

# Colored background for each biomolecule region
plt.axvspan(1720, 1755, color='purple', alpha=0.1, label='Lipids')
plt.axvspan(1610, 1690, color='green', alpha=0.1, label='Proteins')
plt.axvspan(1500, 1600, color='green', alpha=0.1)
plt.axvspan(1215, 1245, color='blue', alpha=0.1, label='Nucleic Acids')
plt.axvspan(1065, 1095, color='blue', alpha=0.1)
plt.axvspan(1020, 1120, color='orange', alpha=0.1, label='Carbohydrates')

# Plot final spectrum
plt.plot(wavenumbers, total_abs, color='black', linewidth=2, label='Total Spectrum')

# Labels and formatting
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorbance (a.u.)')
plt.title('Illustrative Absorbance Spectrum')
plt.legend(loc='upper left', fontsize=14)
# plt.grid(True)
plt.tight_layout()
plt.show()
