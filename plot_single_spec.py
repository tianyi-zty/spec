import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
import glob
from scipy.spatial import ConvexHull


def rubberband_baseline_correction(x, y):

    x = np.array(x)
    y = np.array(y)

    # Get points forming the convex hull
    v = np.vstack((x, y)).T
    hull = ConvexHull(v)

    # Extract lower convex hull indices (start and end inclusive)
    hull_indices = sorted(hull.vertices)
    lower_indices = [idx for idx in hull_indices if idx == 0 or idx == len(x) - 1 or (y[idx] < y[idx-1] and y[idx] < y[idx+1])]
    lower_indices = np.array(sorted(lower_indices))

    # Interpolate baseline across those points
    baseline = np.interp(x, x[lower_indices], y[lower_indices])

    corrected_y = y - baseline

    return baseline, corrected_y


def main():
        # # Plot all spectrum 
        foldername_list = ['1000/','8020SER/','8020PSER/','6040SER/','6040PSER/'] #['1000/','9109/','9505/'] # '1000','9010', '8020','7030' 
        sub_folder_list = {'LMT_1','LMT_2'} #'LMT_1',

        for fl in foldername_list:
            for sub in sub_folder_list:
                folders = f'../res/Caf2_10142025/org/{fl}/{sub}'
                label = f'{fl}_{sub}'
                wavelengths = np.linspace(950, 1800, 426)

                plt.figure(figsize=(10, 7))
                spectra_all = []
                for i, file in enumerate(os.listdir(folders)):
                    if file.endswith('.npy'):
                        # print(f"{folder} - {i}")
                        spectrum = np.load(os.path.join(folders, file))
                        _, corrected = rubberband_baseline_correction(wavelengths, spectrum)
                        plt.plot(wavelengths, corrected)
                        spectra_all.append(corrected)

                spectra_all = np.array(spectra_all)
                mean_spectrum = np.mean(spectra_all, axis=0)
                std_spectrum = np.std(spectra_all, axis=0)
                plt.xlabel("Wavelength (cm⁻¹)", fontsize=14)
                plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
                plt.title(f"{fl}/{sub} Spectra", fontsize=16)
                # plt.legend(loc='upper left', fontsize=12)
                # plt.grid(True)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(f'{folders}/spec_vis.png'))
                # plt.show()
                # st()
                # # Save mean spectrum as .npy
                # save_subfolder = '../spec_res/Caf2_02092026_amide1/org/mean_spec'
                # os.makedirs(save_subfolder, exist_ok=True)
                # np.save(os.path.join(save_subfolder, f'{fl}_{sub}_mean_spectrum.npy'), mean_spectrum)
            
                # # Plot the average spectrum with standard deviation
                plt.figure(figsize=(10, 7))
                plt.plot(wavelengths, mean_spectrum, label=label)
                plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum +std_spectrum, alpha=0.2)
                plt.xlabel("Wavelength (cm⁻¹)", fontsize=14)
                plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
                plt.title(f"{fl}/{sub} Average Spectra", fontsize=16)
                # plt.legend(loc='upper left', fontsize=12)
                # plt.grid(True)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(f'{folders}/avg_spec.png'))
                # plt.show()

        # # Colored background for each biomolecule region
        # plt.axvspan(1720, 1755, color='purple', alpha=0.1, label='Lipids')
        # plt.axvspan(1610, 1690, color='green', alpha=0.1, label='Proteins')
        # plt.axvspan(1500, 1600, color='green', alpha=0.1)
        # plt.axvspan(1215, 1245, color='blue', alpha=0.1, label='Nucleic Acids')
        # plt.axvspan(1065, 1095, color='blue', alpha=0.1)
        # plt.axvspan(1020, 1120, color='orange', alpha=0.1, label='Carbohydrates')
        # plt.plot(wavelengths, spectra_col_als_1, label='liver_ffpe',color='orange',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_2, label='kidney_ffpe',color='blue',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_3, label='liver_ff',color='green',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_4, label='kidney_ff',color='purple',linewidth=3) 
        # plt.xlabel('Wavenumber (cm⁻¹)', fontsize=14)
        # plt.ylabel('Intensity', fontsize=14)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.legend(loc='upper left', fontsize=14)
        # plt.title(f"Absorption Spectrum")
        # plt.grid(True)

        # plt.show()
        # # plt.savefig(os.path.join(path, 'spectrum visualization.png'))


if __name__ == '__main__':
    main()