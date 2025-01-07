__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from lmfit import models
from scipy.integrate import simpson


def rubberband_baseline_correction(x, y):
    """
    Perform rubberband baseline correction for a given spectrum.
    
    Parameters:
        x (array-like): The x-axis values (e.g., wavelength or wavenumber).
        y (array-like): The y-axis values (e.g., intensity).

    Returns:
        corrected_y (array-like): The baseline-corrected y values.
        baseline (array-like): The estimated baseline.
    """
    # Find the convex hull
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    
    # Extract the vertices of the convex hull
    v = hull.vertices
    
    # Rotate the convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    
    # Leave only the ascending part of the convex hull
    v = v[:v.argmax() + 1]
    
    # Create the baseline using linear interpolation between the vertices
    lower_baseline = np.interp(x, x[v], y[v])
    
    # Subtract the baseline from the original spectrum
    corrected_y = y - lower_baseline
    # st()
    return corrected_y, lower_baseline

def local_max_baseline_with_x_ranges(x, y, x_ranges, min_x_distance=10):
    """
    Generate a baseline using local maxima within specified x-ranges by:
    - Extracting vertices of local maxima within the specified ranges.
    - Rotating to start from the lowest x-value.
    - Keeping only the ascending part of the maxima.
    - Interpolating a baseline using these vertices, stopping at the last local maximum.

    Parameters:
        x (array-like): The x-axis values (e.g., wavelength or wavenumber).
        y (array-like): The y-axis values (e.g., intensity).
        x_ranges (list of tuples): List of x-ranges (min_x, max_x) to restrict the search for local maxima.
        min_x_distance (int): Minimum x distance between peaks.

    Returns:
        baseline (array-like): The estimated baseline, stopping at the last local maximum.
    """
    # Calculate the minimum index distance based on min_x_distance
    min_index_distance = np.searchsorted(x, x[0] + min_x_distance) - 1

    # Initialize lists for x and y of local maxima within specified ranges
    x_peaks_list = []
    y_peaks_list = []
    
    for x_min, x_max in x_ranges:
        # Find indices within the current x-range
        range_indices = np.where((x >= x_min) & (x <= x_max))[0]
        if len(range_indices) == 0:
            continue  # Skip empty ranges
        
        # Find local maxima within the range with minimum x-distance constraint
        peaks, _ = find_peaks(y[range_indices], distance=min_index_distance)
        x_peaks = x[range_indices][peaks]
        y_peaks = y[range_indices][peaks]
        
        # Append these peaks to the lists
        x_peaks_list.extend(x_peaks)
        y_peaks_list.extend(y_peaks)
    
    # Sort the peaks by x value
    x_peaks = np.array(x_peaks_list)
    y_peaks = np.array(y_peaks_list)
    sorted_indices = np.argsort(x_peaks)
    x_peaks = x_peaks[sorted_indices]
    y_peaks = y_peaks[sorted_indices]
    # st()
    
    # Rotate the peaks to start from the lowest x-value
    min_index = np.argmin(x_peaks)
    x_peaks = np.roll(x_peaks, -min_index)
    y_peaks = np.roll(y_peaks, -min_index)
    print('local maximum find at:', x_peaks)
    
    # Stop the baseline at the last local maximum by setting the x limit
    x_max_limit = x_peaks[-1]

    # Interpolate the baseline between these vertices, stopping at the last local maximum
    # x_interp = x[x <= x_max_limit]
    upper_baseline = np.interp(x, x_peaks, y_peaks)
    
    # Find the index where x exceeds the last maximum
    beyond_max_index = np.where(x > x_max_limit)[0]
    
    # Replace baseline values beyond the last maximum with original y values
    if beyond_max_index.size > 0:
        upper_baseline[beyond_max_index] = y[beyond_max_index]
    return upper_baseline, x_peaks

def flip_between_local_maxima(x, y, x_peaks, baseline):
    """
    Flip sections of the spectrum according to the baseline between each pair of local maxima.
    """
    y_flipped = np.copy(y)
    final_flipped = np.copy(y)
    for i in range(len(x_peaks) - 1):
        x_min = x_peaks[i]
        x_max = x_peaks[i + 1]
        flip_indices = np.where((x >= x_min) & (x <= x_max))[0]
        y_flipped[flip_indices] = 2 * baseline[flip_indices] - y[flip_indices]
        final_flipped[flip_indices] = baseline[flip_indices] - y[flip_indices]
    # st()
    # Set `final_flipped` to 0 beyond the last maximum
    if len(x_peaks) > 0:
        last_max_index = np.where(x > x_peaks[-1])[0]
        final_flipped[last_max_index] = 0

    return y_flipped, final_flipped

# Function to generate model and parameters without fitting
def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            
            default_params = {
                prefix+'center': x_min + x_range * np.random.random(),
                prefix+'height': y_max * np.random.random(),
                prefix+'sigma': x_range * np.random.random()
            }
        else:
            raise NotImplemented(f'Model {basis_func["type"]} not implemented yet')
        
        if 'help' in basis_func:
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)           
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    print('center    model   amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')


def main():
    after_collagen = r'W:/3. Students/TianYi/Caf2_11152024/aftercollagen/LMT_1.mat'
    save_path = '../res/Caf2_11152024/'

    wavelength_start = 950
    wavelength_end = 1800
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))
    # Define the region of interest on the x and y axes
    x_start, x_end = 100, 400 #30, 230 #250, 350 # #  # Replace with your desired x range
    y_start, y_end = 100, 400  # Replace with your desired y range

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    # Extract the subregion and wavelength range
    extracted_region_after = spectra_after[x_start:x_end, y_start:y_end, z_indices]
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))
    # Calculate 10^(-mean_spectrum_after)
    transformed_spectrum = 10 ** (-mean_spectrum_after)
    # Apply rubberband baseline correction to the transformed spectrum
    corrected_spectrum, lower_baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum)
    # Define your x ranges for maxima
    x_ranges = [(1190, 1215), (1224,1249), (1450,1475), (1542,1567), (1650,1675)]

    upper_baseline, x_peaks = local_max_baseline_with_x_ranges(wavelengths[z_indices], corrected_spectrum, x_ranges, min_x_distance=10)
    flipped_spectrum, final_flipped = flip_between_local_maxima(wavelengths[z_indices], corrected_spectrum, x_peaks, upper_baseline)
    st()
    # Define the specification of the models
    spec = {
        'x': wavelengths[z_indices],
        'y': final_flipped,
        'model': [
            {
                'type': 'GaussianModel',
                'params': {'center': 1022, 'height': 0.1, 'sigma': 1},
            },
            {   'type': 'VoigtModel',
                'params': {'center': 1058, 'height': 0.1, 'sigma': 1, 'gamma': 0.1},
            },
            {
                'type': 'GaussianModel',
                'params': {'center': 1137, 'height': 0.1, 'sigma': 1},
            },
            {   'type': 'GaussianModel', 
                'params': {'center': 1188, 'height': 0.1, 'sigma': 1}
            },
            {   'type': 'GaussianModel', 
                'params': {'center': 1358, 'height': 0.1, 'sigma': 1}
            },
            {
                'type': 'GaussianModel',
                'params': {'center': 1466, 'height': 0.1, 'sigma': 1},
            },
            {
                'type': 'GaussianModel',
                'params': {'center': 1480, 'height': 0.1, 'sigma': 1},
            },
            {   'type': 'VoigtModel', 
                'params': {'center': 1540, 'height': 0.1, 'sigma': 1, 'gamma': 0.1}},
            {
                'type': 'GaussianModel',
                'params': {'center': 1624, 'height': 0.1, 'sigma': 1},
            },
            {   'type': 'GaussianModel', 
                'params': {'center': 1660, 'height': 0.1, 'sigma': 1},
            }
        ]
    }
    # st()
    # Generate the model and fit the data
    model, params = generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['x'])

    # Print best fit values
    print_best_values(spec, output)

    # Define the component names
    component_names = ['glucose','C-O','Oligosaccharide C-OH 2-Methylmannoside',  'deoxyribose', 'Stretching C-O, deformation C-H, deformation N-H','CH2','Polyethylene methylene deformation modes ', 'amide 2', 'amide 1', 'amide 2']
    component_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFB3E6', '#FFD700', '#B0E0E6', '#C71585', '#8A2BE2', '#20B2AA']  # Define colors
    # component_names = ['CH2', 'amide 2', 'amide 1', 'amide 2']
    # component_colors = ['#FFCC99', '#FFB3E6', '#FFD700', '#B0E0E6']

    # Plot the results
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(spec['x'], spec['y'], 'o', label='Data', markersize=4)
    components = output.eval_components(x=spec['x'])
    # st()
    
    # Integrate and plot each component
    for i, model in enumerate(spec['model']):
        if i < len(component_names):
            # Extract the component data
            component_y = components[f'm{i}_']
            
            # Compute the numerical integral
            integral_value = simpson(component_y, x = spec['x'])
            
            # Plot the component with a label including the integral
            ax.plot(spec['x'], component_y, label=f'{component_names[i]} (∫={integral_value:.0f})', color=component_colors[i])
            ax.fill_between(spec['x'], component_y, color=component_colors[i], alpha=0.3)  # Fill under the curve

    # Add dashed vertical lines for each model center
    for i, model in enumerate(spec['model']):
        center = output.best_values[f'm{i}_center']
        ax.axvline(x=center, linestyle='--', color='gray', alpha=0.5)
        
        # Get the maximum y value of the data for annotation positioning
        max_y = ax.get_ylim()[1]  # Get the upper limit of y-axis
        ax.annotate(f'{center:.0f}', xy=(center, max_y), 
                    xytext=(5, 5), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=16,
                    ha='left', va='center')  # Align to the left and centered vertically

    ax.legend(loc="upper right")
    plt.show()
    st()

    plt.savefig(os.path.join(save_path, f'peak_fit_integration.png'))


if __name__ == '__main__':
    main()