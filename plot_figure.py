import matplotlib.pyplot as plt
import numpy as np

# Define wavenumber range
wavenumber = np.linspace(800, 4000, 500)

# Create a base spectrum with some random noise
absorbance = 0.1 + 0.03 * np.random.rand(len(wavenumber))


# Add peaks for different biomolecules (using Gaussian functions for simplicity)
def gaussian(x, center, amplitude, width):
    return amplitude * np.exp(-((x - center) / width) ** 2)

# Lipids (e.g., C-H stretching)
absorbance += gaussian(wavenumber, 2920, 0.3, 50)
absorbance += gaussian(wavenumber, 2850, 0.2, 40)
absorbance += gaussian(wavenumber, 1740, 0.4, 30) # C=O stretching

# Proteins (e.g., Amide I and II)
absorbance += gaussian(wavenumber, 1650, 0.5, 40) # Amide I
absorbance += gaussian(wavenumber, 1540, 0.3, 30) # Amide II

# DNA (e.g., Phosphate backbone, bases)
absorbance += gaussian(wavenumber, 1240, 0.25, 35) # Asymmetric PO2- stretching
absorbance += gaussian(wavenumber, 1080, 0.2, 30)  # Symmetric PO2- stretching

# Carbohydrates (e.g., C-O-C stretching)
absorbance += gaussian(wavenumber, 1030, 0.4, 45)
absorbance += gaussian(wavenumber, 1150, 0.3, 35)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(wavenumber, absorbance, color='black', linewidth=2)

# Annotate specific bands with shaded regions
plt.axvspan(2800, 3000, color='thistle', alpha=0.3, label='Lipids (C-H stretching)')
plt.axvspan(1700, 1780, color='thistle', alpha=0.3)
plt.axvspan(1600, 1700, color='lightgreen', alpha=0.3, label='Proteins (Amide I)')
plt.axvspan(1500, 1580, color='lightgreen', alpha=0.3, label='Proteins (Amide II)')
plt.axvspan(1200, 1280, color='lightblue', alpha=0.3, label='DNA (PO2- asym)')
plt.axvspan(1050, 1100, color='lightblue', alpha=0.3, label='DNA (PO2- sym)')
plt.axvspan(1000, 1180, color='lightsalmon', alpha=0.3, label='Carbohydrates (C-O-C)')

# Add labels and title
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
plt.ylabel('Absorbance (a.u.)', fontsize=12)
plt.title('Illustrative Absorbance Spectrum of Biomolecules', fontsize=14)
plt.legend(fontsize='large')
# plt.grid(True)
plt.xlim(800, 3000)
plt.ylim(0, np.max(absorbance) * 1.1)
plt.gca().invert_xaxis() # Typically wavenumber is plotted with decreasing values

plt.tight_layout()
plt.show()