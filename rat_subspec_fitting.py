import os
import csv
import numpy as np
from scipy.io import loadmat
from scipy.integrate import simpson
from lmfit import models
import json
import matplotlib.pyplot as plt
from pdb import set_trace as st
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter
import random



def generate_model_from_specification(json_file, spec, threshold=10):
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
        # print(reduced_chi_squared)
        
        if reduced_chi_squared < threshold:
            if composite_model is None:
                composite_model = model
            else:
                composite_model += model
            if params is None:
                params = result.params
            else:
                params.update(result.params)

    return composite_model, params


def plot_results(spec, components, component_names, component_colors, output, file_name, save_path):
    """
    Plot the spectrum fitting results and save the plot.
    """
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.plot(spec['x'], spec['y'] - min(spec['y']), 'o', label='Data', markersize=10)
    
    # Plot each component
    for i, name in enumerate(component_names):
        component_y = components.get(f'm{i}_', None)
        if component_y is not None:
            integral_value = simpson(component_y, x=spec['x'])
            ax.plot(spec['x'], component_y, label=f'{name} (∫={integral_value:.0f})', color=component_colors[i])
            # ax.plot(spec['x'], component_y, label=f'{name}', color=component_colors[i])
            ax.fill_between(spec['x'], component_y, color=component_colors[i], alpha=0.3)
    
    # Add dashed vertical lines for each model center
    for i, name in enumerate(component_names):
        center = output.best_values.get(f'm{i}_center', None)
        if center is not None:
            ax.axvline(x=center, linestyle='--', color='gray', alpha=0.7)
            max_y = ax.get_ylim()[1]
            # ax.annotate(f'{center:.0f}', xy=(center, max_y), 
            #             xytext=(5, 5), textcoords='offset points',
            #             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=20,
            #             ha='left', va='center')
    ax.set_ylim([-0.02,1.5])
    ax.legend(loc="upper left", fontsize=34)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=28)
    ax.set_ylabel('Intensity', fontsize=28)
    formatter = FuncFormatter(lambda val,pos:f'{val:.1f}')
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticklabels(ax.get_xticks(),fontsize=34)
    ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()],fontsize=34)
    # ax.set_title(f'Spectrum Fitting Results - {file_name}', fontsize=18)

    # Save the plot
    plt.tight_layout()
    # plt.legend(loc='upper left', fontsize=24)
    plot_file = os.path.join(save_path, f"{file_name}_fitting.png")
    # st()
    plt.savefig(plot_file)
    plt.close(fig)

############################## change here wavelength range#####################################
def process_folder(input_folder, output_csv, summary_csv, json_file, save_plots_folder,
                   wavelength_start=950, wavelength_end=1800, n_random=20):
    """
    Process random subset of .npy files in a folder, perform subspectrum fitting,
    plot results, and save statistics to a CSV.
    """
    os.makedirs(save_plots_folder, exist_ok=True)

    component_integrals = []

    # --- collect all npy files ---
    all_files = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".npy"):
                all_files.append(os.path.join(root, file_name))

    # --- randomly select up to n_random files ---
    selected_files = random.sample(all_files, min(n_random, len(all_files)))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Amplitude', 'Center Value', 'Sigma', 'Integral Value'])

        for file_path in selected_files:
            file_name = os.path.basename(file_path)

            # Load the .mat file
            # data = loadmat(file_path)
            # spectra = data['spectrum'].flatten()
            data = np.load(file_path)
            spectra = data

            # smoothing spectrum
            spectra = savgol_filter(spectra, window_length=11, polyorder=3)

            # Define the spectrum
            spec = {
                'x': np.linspace(wavelength_start, wavelength_end, spectra.shape[0]),
                'y': spectra
            }
            
            try:
                # Perform model fitting
                model, params = generate_model_from_specification(json_file, spec, threshold=0.5)
                output = model.fit(spec['y'] - min(spec['y']), params, x=spec['x'])
                components = output.eval_components(x=spec['x'])

                best_values = output.best_values
                integrals_for_file = []
                for i, basis_func in enumerate(json.load(open(json_file))['models']):
                    prefix = f'm{i}_'
                    center_value = best_values.get(prefix + 'center', None)
                    amplitude = best_values.get(prefix + 'amplitude', None)
                    sigma = best_values.get(prefix + 'sigma', None)
                    component_y = components.get(f'm{i}_', None)
                    integral_value = simpson(component_y, x=spec['x']) if component_y is not None else 0

                    if len(component_integrals) <= i:
                        component_integrals.append([])
                    component_integrals[i].append(integral_value)

                    writer.writerow([file_name, center_value, amplitude, sigma, integral_value])
                    integrals_for_file.append(integral_value)
                print(f'Done processing: {file_name}.')
                # st()
                # # # Plot and save the results
                
                # component_names = ["PO4?","C-O bending","Phosphate band/Collagen","C-O of carbonhydrates","C-O stretching band of collagen (type I)",
                #                    "phosphate I/Amide III","PO2 /Amide III",
                #                    "Amide III band components of protein","CH2 wagging/collagen","Symmetric CH3 bending modes of the methyl groups of proteins",
                #                    "Asymmetric CH3 bending modes of the methyl groups of proteins","In-plane CH bending vibration from the phenyl rings/Amide II",
                #                    "Amide II","Ring C-C stretch of phenyl (2)/Amide II","b-sheet Amide I","α-helix Amide I","β-turn Amide I"] #collagen
                # component_colors = plt.cm.tab20.colors[:len(components)]

                # component_names = ["Proteins","Lipids","Amide II","Amide II","Amide II","Amide II","Amide I","Amide I"]#kidney ffpe
                # component_colors = ((1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                #                     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313))
                
                component_names = ["Amide III","Proteins","Proteins","Lipids","Amide II","Amide II","Amide I"]#liver ffpe
                component_colors = ((0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                                    (1.0, 0.4980392156862745, 0.054901960784313725),
                                    (1.0, 0.4980392156862745, 0.054901960784313725),
                                    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                                    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                                    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                                    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313))
                # component_names = ["Symmetric PO2 stretching","Phosphate band/Collagen","Amide III","phosphate I/Amide III",
                #                     "Amide III band components of protein","CH2 wagging/collagen",
                #                     "Symmetric CH3 bending modes of the methyl groups of proteins",
                #                     "Asymmetric CH3 bending modes of the methyl groups of proteins",
                #                     "Amide II beta-sheet","Amide II","beta-sheet Amide I","alpha-helix Amide I","Coils/turn Amide I"] #collagen
                # component_colors = plt.cm.tab20.colors[:len(components)]

                # component_names = ["DNA","Glycogen","Collagen","Glycogen","Amide III","Amide III","Proteins","Proteins","Lipids","Amide II","Amide II","Amide I","Lipids"] #liver oct
                # component_colors = ((0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                #                     (1.0, 0.596078431372549, 0.5882352941176471),
                #                     (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                #                     (1.0, 0.596078431372549, 0.5882352941176471),
                #                     (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                #                     (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                #                     (1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
                #                     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392))
               
                # st()
                # component_names = ["DNA","DNA","Collagen","Proteins","Amide III","Amide III","Proteins","Proteins","Lipids","Amide II","Amide II","Amide I","Lipids"] #kidney oct
                # component_colors = ((0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                #                     (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                #                     (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                #                     (1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                #                     (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                #                     (1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (1.0, 0.4980392156862745, 0.054901960784313725),
                #                     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                #                     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
                #                     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392))

                plot_results(spec, components, component_names, component_colors, output, file_name, save_plots_folder)
                # st()

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Calculate averages and standard deviations
    averages = [np.mean(integrals) for integrals in component_integrals]
    std_devs = [np.std(integrals) for integrals in component_integrals]

    # Save summary statistics
    with open(summary_csv, mode='w', newline='') as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(['Component', 'Average Integral', 'Standard Deviation'])
        for i, (avg, std) in enumerate(zip(averages, std_devs)):
            summary_writer.writerow([f'Component {i + 1}', avg, std])

    print(f"Summary statistics saved to {summary_csv}")


if __name__ == '__main__':
    # Define paths
    hmt_list = ['HMT_1','HMT_2','HMT_3','HMT_4','HMT_5','HMT_6'] #,'HMT_2','HMT_3','HMT_4','HMT_5','HMT_6'
    for hmt in hmt_list:
        print(hmt)
        path = f'D:/spec_res/rat/liver_ffpe/{hmt}'
        input_folder = path 
        output_folder = f'D:/res/rat_integral/liver_ffpe/{hmt}'
        os.makedirs(output_folder, exist_ok=True)
        output_csv = output_folder +'/subspectrum_fitting_results.csv'
        summary_csv = output_folder +'/summary_statistics.csv'
        save_plots_folder = output_folder + '/plots'
        json_file = 'model_specification_liverffpe.json'
        # os.makedirs(input_folder, exist_ok=True)
        # os.makedirs(save_plots_folder, exist_ok=True)

        # Run the processing
        process_folder(input_folder, output_csv, summary_csv, json_file, save_plots_folder)
