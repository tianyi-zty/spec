from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from lmfit import models
import numpy as np
from scipy.io import loadmat
import json
import os
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from pdb import set_trace as st


################not done yet, cannot run#################
# Define the objective function to evaluate the model's performance
def objective_function(params, spec, model_spec, threshold=0.95):
    """
    The objective function for Bayesian optimization. It evaluates the reduced chi-squared of a model
    with the given parameters and returns it for minimization.
    
    Args:
        params (list): The parameters to optimize.
        spec (dict): The spectrum data (x and y).
        model_spec (dict): Model specifications loaded from a JSON file.
        threshold (float): Reduced chi-squared threshold to decide fit quality.
        
    Returns:
        reduced_chi_squared (float): The reduced chi-squared value of the fit.
    """
    # Print received parameters to check correctness
    print(f"Received parameters: {params}")

    # Extract parameters from the list
    param_dict = {}
    i = 0
    for basis_func in model_spec['models']:
        for param, options in basis_func['params'].items():
            param_dict[param] = params[i]
            i += 1
    
    # Generate the model using the parameters
    model, params_init = generate_model_from_specification_with_params(spec, model_spec, param_dict)
    
    # Fit the model
    result = model.fit(spec['y'], params_init, x=spec['x'])
    reduced_chi_squared = result.chisqr / result.nfree
    
    return reduced_chi_squared



# Generate the model using the specified parameters
def generate_model_from_specification_with_params(spec, model_spec, param_dict):
    x = spec['x']
    y = spec['y']
    composite_model = None
    params = None
    
    for i, basis_func in enumerate(model_spec['models']):
        model_type = basis_func['type']
        params_dict = basis_func['params']
        prefix = f'm{i}_'
        
        model = getattr(models, model_type)(prefix=prefix)
        
        for param, options in params_dict.items():
            if param in param_dict:
                model.set_param_hint(param, value=param_dict[param])

        init_params = model.make_params()
        for param, options in params_dict.items():
            if param in param_dict:
                init_params[prefix + param].value = param_dict[param]
        
        if composite_model is None:
            composite_model = model
        else:
            composite_model += model
        
        if params is None:
            params = init_params
        else:
            params.update(init_params)
            
    return composite_model, params

# Define the search space for parameters (dynamic space generation)
space = [
    Real(0, 10, name='param1'),
    Categorical([1, 2, 3], name='param2')
]

@use_named_args(space)
def fit_model_with_optimization(params):
    """
    Fit the model using Bayesian Optimization and return the reduced chi-squared value.
    
    Args:
        params (list): The parameters to optimize (will be passed by `gp_minimize`).
        
    Returns:
        reduced_chi_squared (float): The reduced chi-squared value for the current model fit.
    """
    # Load data and model specification
    spectrum_path = '../res/Caf2_11152024/rubberband/Rubberband_flipped_spectrum_after.mat'
    with open('model_specification.json', 'r') as f:
        model_spec = json.load(f)

    data = loadmat(spectrum_path)
    spectra = data['corrected_spectrum'].flatten()
    wavelengths = np.linspace(950, 1800, 426)

    spec = {
        'x': wavelengths,
        'y': spectra,
        'model': model_spec['models']
    }

    # Print the number of parameters received by the function
    print(f"Number of parameters received: {len(params)}")
    print(f"Input x to fit_model_with_optimization: {params} (len={len(params)})")  # Debug line
    # Evaluate the reduced chi-squared with the given parameters
    reduced_chi_squared = objective_function(params, spec, model_spec)
    return reduced_chi_squared

def optimize_parameters():
    """
    Perform Bayesian optimization to find the best parameters for model fitting.
    """
    # Load the model specification
    model_spec = json.load(open('model_specification.json'))

    # Define the search space for the parameters
    space = []
    for i, basis_func in enumerate(model_spec['models']):
        for param, options in basis_func['params'].items():
            if 'min' in options and 'max' in options:
                if isinstance(options['min'], (int, float)) and isinstance(options['max'], (int, float)):
                    space.append(Real(options['min'], options['max'], name=f'{param}_{i}'))
                else:
                    space.append(Real(0.01, 10.0, name=f'{param}_{i}'))
            else:
                space.append(Real(0.01, 10.0, name=f'{param}_{i}'))  # Default range

    # Print the length of the space for debugging
    print(f"Search space dimensions: {len(space)}")
    print(f"Parameter names: {[dim.name for dim in space]}")

    print(f"Final space passed to gp_minimize: {space}")
    # Perform Bayesian optimization
    res = gp_minimize(func=fit_model_with_optimization, dimensions=space, n_calls=50, random_state=42)

    print(f"Best parameters: {res.x}")
    print(f"Best reduced chi-squared: {res.fun}")

    return res



def main():
    optimize_parameters()

if __name__ == "__main__":
    main()
