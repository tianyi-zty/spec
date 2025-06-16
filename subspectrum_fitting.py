__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from lmfit import models
from scipy.integrate import simpson
import json


def generate_model_from_specification(json_file, spec, threshold=10):
    """
    Generate and fit models from JSON specifications to the provided spectrum.
    
    Args:
        json_file (str): Path to the JSON file containing model specifications.
        spec (dict): Dictionary containing 'x' and 'y' keys for spectrum data.
        threshold (float): Reduced chi-squared threshold to decide fit quality.
    
    Returns:
        composite_model (lmfit.Model): Final composite model with well-fitted components.
        params (lmfit.Parameters): Parameters for the composite model.
    """
    # Load JSON
    with open(json_file, 'r') as file:
        model_spec = json.load(file)

    x = spec['x']
    y = spec['y']
    composite_model = None
    params = None
    good_components = []

    for i, basis_func in enumerate(model_spec['models']):
        if "_comment" in basis_func:  # Optional: Skip commented-out components
            print(f"Processing: {basis_func['_comment']}")

        model_type = basis_func['type']
        params_dict = basis_func['params']
        prefix = f'm{i}_'
        
        # Create model
        model = getattr(models, model_type)(prefix=prefix)
        
        # Enforce parameter ranges from model_spec
        for param, options in params_dict.items():
            # Check if the parameter has a range defined in the model_spec and apply it
            if 'min' in options:
                model.set_param_hint(param, min=options['min'])
            if 'max' in options:
                model.set_param_hint(param, max=options['max'])
            if 'value' in options:
                model.set_param_hint(param, value=options['value'])
       
        # Ensure amplitude is positive
        if 'amplitude' in params_dict:
            model.set_param_hint('amplitude', min=0)

        # Generate initial parameters
        init_params = model.make_params()
        for param, options in params_dict.items():
            if 'value' in options:
                init_params[prefix + param].value = options['value']
        
        # Fit this individual model
        result = model.fit(y, init_params, x=x)
        reduced_chi_squared = result.chisqr / result.nfree
        
        print(f"Model {model_type} Reduced Chi^2: {reduced_chi_squared:.3f}")
        if reduced_chi_squared < threshold:  # Add good fits to composite model
            print(f"Accepted {model_type} (Reduced Chi^2 < {threshold})")
            good_components.append(model_type)
            if composite_model is None:
                composite_model = model
            else:
                composite_model += model
            if params is None:
                params = result.params
            else:
                params.update(result.params)
        else:
            print(f"Rejected {model_type} (Reduced Chi^2 >= {threshold})")

    print(f"Final Composite Model includes: {', '.join(good_components)}")
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
        # values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        values = ', '.join(f'{best_values.get(prefix+param, float("nan")):8.3f}' for param in model_params[model["type"]])

        # print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')
        center_value = best_values.get(prefix + "center", None)
        if center_value is None:
            print(f"Warning: Key {prefix+'center'} not found in best_values.")
            continue  # Skip this model if the key is missing
        print(f'[{center_value:3.3f}] {model["type"]:16}: {values}')


def main():
    filename = '99-1'
    spectrum_path = '../res/AuPillars_Al2O3_12102024/ALS/3/new_way/'+f'{filename}'+'/' +f'{filename}'+'ROI Spectrum1199-1600.mat'
    # spectrum_path = '../res/AuPillars_Al2O3_12102024/ALS/2/new_way/99-1/99-1ROI Spectrum1299-1600.mat'
   
    save_path = '../res/AuPillars_Al2O3_12102024/ALS/3/new_way'
    # Wavelength range (cm⁻¹)
    wavelength_start = 1200
    wavelength_end = 1600

    # Load the .mat file
    data = loadmat(spectrum_path)
    spectra = data['corrected_spectrum'].flatten()
    # st()
    # spectra = spectra[:100]
    # st()
    # # Find the indices corresponding to the desired wavelength range
    # z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    
    with open('model_specification.json', 'r') as f:
        model_spec = json.load(f)

    # Define the specification for models
    spec = {
        'x': np.linspace(wavelength_start, wavelength_end, spectra.shape[0]),
        'y': spectra,
        'model': model_spec['models']
    }
    # st()
    # Generate the model and fit the data
    model, params = generate_model_from_specification("model_specification.json", spec, threshold=0.5)
    output = model.fit(spec['y']-min(spec['y']), params, x=spec['x'])
    # st()
    # Print best fit values
    print_best_values(spec, output)

    # Define the component names
    component_names = ['Amide III',
                       "weak peaks of DNA",
                       'CH3 of collagen',
                       "Amide II "
                       ]
    component_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']    
    # '#99FF99', '#7FFF00', '#FFA500', '#20B2AA', '#8A2BE2', '#FFB3E6', '#C71585', '#B0E0E6', '#FFD700'

    # Plot the results
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(spec['x'], spec['y']-min(spec['y']), 'o', label='Data', markersize=4)
    components = output.eval_components(x=spec['x'])
    # st()
    
    # Integrate and plot each component
    for i, model in enumerate(spec['model']):
        if i < len(component_names):
            # Extract the component data
            # component_y = components[f'm{i}_']
            component_y = components.get(f'm{i}_', None)
            if component_y is None:
                print(f"Warning: Missing component for model with prefix m{i}_")
                continue  # Skip this model if the component is missing
            
            # Compute the numerical integral
            integral_value = simpson(component_y, x = spec['x'])
            
            # Plot the component with a label including the integral
            ax.plot(spec['x'], component_y, label=f'{component_names[i]} (∫={integral_value:.0f})', color=component_colors[i])
            ax.fill_between(spec['x'], component_y, color=component_colors[i], alpha=0.3)  # Fill under the curve

    # Add dashed vertical lines for each model center
    for i, model in enumerate(spec['model']):
        # center = output.best_values[f'm{i}_center']
        center = output.best_values.get(f'm{i}_center', None)
        if center is None:
            print(f"Warning: No center value found for model with prefix m{i}_")
            continue  # Skip this model if the center value is missing
        
        ax.axvline(x=center, linestyle='--', color='gray', alpha=0.5)
        
        # Get the maximum y value of the data for annotation positioning
        max_y = ax.get_ylim()[1]  # Get the upper limit of y-axis
        ax.annotate(f'{center:.0f}', xy=(center, max_y), 
                    xytext=(5, 5), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=16,
                    ha='left', va='center')  # Align to the left and centered vertically

    ax.legend(loc="upper right")
    ax.set_xlabel('Wavelength (cm⁻¹)', fontsize=16)
    ax.set_ylabel('Intensity', fontsize=16)
    # ax.set_title('Spectral Fitting and Component Integration', fontsize=18)
    # plt.show()
    # st()

    plt.savefig(os.path.join(save_path, f'subspectrum_fit_integration_ALS_corrected_flipped_spectrum_'+f'{filename}''.png'))


if __name__ == '__main__':
    main()