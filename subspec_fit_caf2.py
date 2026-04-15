import os
import csv
import numpy as np
from scipy.io import loadmat
from scipy.integrate import simpson
from lmfit import models
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter


def generate_model_from_specification(json_file, spec, threshold=6):
    """
    Generate and fit models from JSON specifications to the provided spectrum.
    """
    with open(json_file, 'r') as file:
        model_spec = json.load(file)

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
            if 'min' in options:
                model.set_param_hint(param, min=options['min'])
            if 'max' in options:
                model.set_param_hint(param, max=options['max'])
            if 'value' in options:
                model.set_param_hint(param, value=options['value'])

        if 'amplitude' in params_dict:
            model.set_param_hint('amplitude', min=0)

        init_params = model.make_params()
        for param, options in params_dict.items():
            if 'value' in options:
                init_params[prefix + param].value = options['value']

        result = model.fit(y, init_params, x=x)
        reduced_chi_squared = result.chisqr / result.nfree

        if reduced_chi_squared < threshold:
            composite_model = model if composite_model is None else composite_model + model
            params = result.params if params is None else params.update(result.params)

    return composite_model, params

def clean_axes(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def plot_results(spec, components, component_names, component_colors, output, file_name, save_path):
    """
    Plot the spectrum fitting results and save the plot.
    """
    fig, ax = plt.subplots(figsize=(36, 16))

    ax.plot(
        spec['x'],
        spec['y'] - min(spec['y']),
        'o',
        label='Data',
        markersize=10
    )

    # Plot each component
    for i, name in enumerate(component_names):
        component_y = components.get(f'm{i}_', None)
        if component_y is not None:
            integral_value = simpson(component_y, x=spec['x'])
            ax.plot(
                spec['x'],
                component_y,
                label=f'{name}',
                color=component_colors[i]
            )
            ax.fill_between(
                spec['x'],
                component_y,
                color=component_colors[i],
                alpha=0.3
            )

    # Axes formatting
    ax.set_ylim([-0.02, 0.7])
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=28)
    ax.set_ylabel('Intensity', fontsize=28)

    formatter = FuncFormatter(lambda val, pos: f'{val:.1f}')
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    # Legend on the right side (outside)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=24,
        frameon=False
    )

    # Adjust layout to make room for legend
    plt.tight_layout() #rect=[0, 0, 0.82, 1]
    clean_axes()
    # Save plot
    plot_file = os.path.join(save_path, f"{file_name}_fitting.png")
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)


def process_folder(
    input_folder,
    output_csv,
    summary_csv,
    json_file,
    save_plots_folder,
    wavelength_start=950,
    wavelength_end=1800
):
    """
    Process all .npy files in a folder, perform subspectrum fitting,
    plot results, and save statistics to a CSV.
    """
    os.makedirs(save_plots_folder, exist_ok=True)
    component_integrals = []

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Amplitude', 'Center Value', 'Sigma', 'Integral Value'])

        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if file_name.endswith('.npy'):
                    file_path = os.path.join(root, file_name)
                    print(f"Processing: {file_path}")
                    spectra = np.load(file_path)
                    spectra = savgol_filter(spectra, window_length=11, polyorder=3)

                    spec = {
                        'x': np.linspace(wavelength_start, wavelength_end, spectra.shape[0]),
                        'y': spectra
                    }

                    try:
                        model, params = generate_model_from_specification(
                            json_file, spec, threshold=0.5
                        )

                        output = model.fit(
                            spec['y'] - min(spec['y']),
                            params,
                            x=spec['x']
                        )

                        components = output.eval_components(x=spec['x'])
                        best_values = output.best_values

                        for i, basis_func in enumerate(json.load(open(json_file))['models']):
                            prefix = f'm{i}_'
                            center_value = best_values.get(prefix + 'center', None)
                            amplitude = best_values.get(prefix + 'amplitude', None)
                            sigma = best_values.get(prefix + 'sigma', None)

                            component_y = components.get(prefix, None)
                            integral_value = simpson(component_y, x=spec['x']) if component_y is not None else 0

                            if len(component_integrals) <= i:
                                component_integrals.append([])
                            component_integrals[i].append(integral_value)

                            writer.writerow([
                                file_name,
                                center_value,
                                amplitude,
                                sigma,
                                integral_value
                            ])

                        component_names = [
                            "Symmetric PO2",
                            "Phosphate band/Collagen",
                            "CO stretching",
                            "Amide III",
                            "Phosphate I/Amide III",
                            "Amide III",
                            "CH2 wagging/collagen",
                            "Symmetric CH3 bending",
                            "Asymmetric CH3 bending",
                            "Amide II",
                            "Amide I beta-sheet ",
                            "Amide I alpha-helix",
                            "Amide I beta-turn"
                        ]

                        component_colors = plt.cm.tab20.colors[:len(component_names)]

                        plot_results(
                            spec,
                            components,
                            component_names,
                            component_colors,
                            output,
                            file_name,
                            save_plots_folder
                        )

                        print(f'Done processing: {file_name}')

                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

    averages = [np.mean(vals) for vals in component_integrals]
    std_devs = [np.std(vals) for vals in component_integrals]

    with open(summary_csv, mode='w', newline='') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Component', 'Average Integral', 'Standard Deviation'])
        for i, (avg, std) in enumerate(zip(averages, std_devs)):
            writer.writerow([f'Component {i + 1}', avg, std])

    print(f"Summary statistics saved to {summary_csv}")


if __name__ == '__main__':
    path = '../res/03232026_col1+4/CAF2/org/mean_spec/'
    input_folder = path
    output_folder = '../res/03232026_col1+4/CAF2/org/spectrum_fitting_results'

    os.makedirs(output_folder, exist_ok=True)

    output_csv = os.path.join(output_folder, 'subspectrum_fitting_results.csv')
    summary_csv = os.path.join(output_folder, 'summary_statistics.csv')
    save_plots_folder = os.path.join(output_folder, 'plots_col')
    json_file = 'model_specification_caf2_col.json'

    os.makedirs(save_plots_folder, exist_ok=True)

    process_folder(
        input_folder,
        output_csv,
        summary_csv,
        json_file,
        save_plots_folder
    )
