
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import yaml
import umap
import glob
import numba
import joblib
import argparse
import matplotlib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from matplotlib import pyplot as plt

# rc('axes', linewidth = 1)

def convert_code_to_7b(code):
    bits = list(code)
    converted_code = ''.join([bits[i] for i in [0,2,3,4,7,8,9]])
    return(converted_code)

def cm_to_inches(x):
    return(x/2.54)

def general_plot(xlabel='', ylabel='', ft=12, dims=(5,3), col='k', lw=1, pad=0):
    fig, ax = plt.subplots(figsize=(dims[0], dims[1]),  tight_layout={'pad': pad})
    for i in ax.spines:
        ax.spines[i].set_linewidth(lw)
    ax.spines['top'].set_color(col)
    ax.spines['bottom'].set_color(col)
    ax.spines['left'].set_color(col)
    ax.spines['right'].set_color(col)
    ax.tick_params(direction='in', labelsize=ft, color=col, labelcolor=col)
    ax.set_xlabel(xlabel, fontsize=ft, color=col)
    ax.set_ylabel(ylabel, fontsize=ft, color=col)
    ax.patch.set_alpha(0)
    return(fig, ax)

def plot_umap(umap_transform, training_data, dims=(5,5), marker='o', alpha=0.5, markersize=1, ft=8, line_col='k'):
    embedding_df = pd.DataFrame(umap_transform.embedding_)
    embedding_df['numeric_barcode'] = training_data.code.apply(int, args = (2,))
    fig, ax = general_plot(dims=dims, col=line_col)
    barcodes = embedding_df.numeric_barcode.unique()
    n_barcodes = barcodes.shape[0]
    cmap = matplotlib.cm.get_cmap('jet')
    delta = 1/n_barcodes
    color_list = [cmap(i*delta) for i in range(n_barcodes)]
    col_df = pd.DataFrame(columns=['numeric_barcode','color'])
    col_df['numeric_barcode'] = barcodes
    col_df['numeric_barcode'] = col_df['numeric_barcode'].astype(int)
    col_df['color'] = color_list
    embedding_df = embedding_df.merge(col_df, how='left', on='numeric_barcode')
    x = embedding_df.iloc[:,0].values
    y = embedding_df.iloc[:,1].values
    colors_plot = embedding_df['color'].values.tolist()
    # ax.plot(x, y, marker, alpha=alpha, ms=markersize, rasterized=True)
    ax.scatter(x, y, marker=marker, c=colors_plot, alpha=alpha, s=markersize, rasterized=True)
    # for i in range(n_barcodes):
    #     enc = i+1
    #     emd = embedding_df.loc[embedding_df.numeric_barcode.values == enc]
    #     ax.plot(emd.iloc[:,0], emd.iloc[:,1], 'o', alpha = 0.5, color = color_list[i], markersize = 1, rasterized = True)
    ax.set_aspect('equal')
    ax.set_xlabel('UMAP 1', fontsize = 8, color = line_col)
    ax.set_ylabel('UMAP 2', fontsize = 8, color = line_col, labelpad = -1)
    return fig, ax

@numba.njit()
def channel_cosine_intensity_7b_v2(x, y):
    check = np.sum(np.abs(x[63:67] - y[63:67]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[63] == 0:
            cos_weight_1 = 0.0
            cos_dist_1 = 0.0
        else:
            cos_weight_1 = 1.0
            for i in range(0,23):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_1 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_1 = 1.0
            else:
                cos_dist_1 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[64] == 0:
            cos_weight_2 = 0.0
            cos_dist_2 = 0.0
        else:
            cos_weight_2 = 1.0
            for i in range(23,43):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_2 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_2 = 1.0
            else:
                cos_dist_2 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[65] == 0:
            cos_weight_3 = 0.0
            cos_dist_3 = 0.0
        else:
            cos_weight_3 = 1.0
            for i in range(43,57):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_3 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_3 = 1.0
            else:
                cos_dist_3 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[66] == 0:
            cos_weight_4 = 0.0
            cos_dist_4 = 0.0
        else:
            cos_weight_4 = 1.0
            for i in range(57,63):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist_4 = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist_4 = 1.0
            else:
                cos_dist_4 = 1.0 - (result / np.sqrt(norm_x * norm_y))
        cos_dist = 0.5*(cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/4
    else:
        cos_dist = 1
    return(cos_dist)


def calculate_fret_efficiency(data_folder, distance):
    files = glob.glob(data_folder + '/*_excitation.csv')
    samples = [re.sub('_excitation.csv', '', os.path.basename(file)) for file in files]
    forster_distance = np.zeros((7,7))
    fret_transfer_matrix = np.eye(7)
    kappa_squared = 2/3
    ior = 1.4
    NA = 6.022e23
    Qd = 1
    prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
    molar_extinction_coefficient = [73000, 112000, 120000, 144000, 270000, 50000, 81000]
    fluorescence_quantum_yield = [0.92, 0.79, 1, 0.33, 0.33, 1, 0.61]
    fluorophores = [10,8,7,6,3,2,1]
    for i in range(7):
        for j in range(7):
            if i != j:
                fi = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[i])))
                fj = pd.read_csv('{}/R{}_excitation.csv'.format(data_folder, str(fluorophores[j])))
                emission_max_i = np.argmax(fi.Emission.values)
                emission_max_j = np.argmax(fj.Emission.values)
                if emission_max_i < emission_max_j:
                    fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                    fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                    j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                    forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[j]*fluorescence_quantum_yield[i], 1/6)
                else:
                    fi_norm = np.clip(fi.Excitation.values/fi.Excitation.max(), 0, 1)
                    fj_norm = np.clip(fj.Emission.values/fj.Emission.sum(), 0, 1)
                    j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                    forster_distance[i,j] = np.power(prefactor*j_overlap*molar_extinction_coefficient[i]*fluorescence_quantum_yield[j], 1/6)
                fret_transfer_matrix[i,j] = np.sign(emission_max_i - emission_max_j)*1/(1+(distance/forster_distance[i,j])**6)
    return(fret_transfer_matrix)


def load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_limited(reference_folder, fret_folder, probe_design_filename, spc):
    # reference_folder = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/utilities/hiprfish_reference_2022_04_07'
    # fret_folder = '/fs/cbsuvlaminck2/workdir/bmg224/archive_hs673/hiprfish_fret'
    # probe_design_filename = '/fs/cbsuvlaminck2/workdir/Data/HIPRFISH/Simulations/DSGN0672-0690/DSGN0673/DSGN0673_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
    # probe_design_filename = '/fs/cbsuvlaminck2/workdir/hs673/Runs/V1/Samples/HIPRFISH_1/simulation/DSGN0567/DSGN0567_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'

    barcode_list = [512, 128, 64, 32, 4, 2, 1]
    files = ['{}/08_18_2018_enc_{}_avgint.csv'.format(reference_folder, b) for b in barcode_list]
    spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
    spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
    nbit = 7
    nchannels = 63
    simulation_per_code = spc
    training_data = pd.DataFrame()
    training_data_negative = pd.DataFrame()
    fret_transfer_matrix = np.zeros((simulation_per_code, 7, 7))
    probes = pd.read_csv(probe_design_filename, dtype = {'code': str})
    code_set = np.unique(probes.code.values)
    print('Calculating FRET efficiency...')
    for i in range(simulation_per_code):
        fret_transfer_matrix[i,:,:] = calculate_fret_efficiency(fret_folder, 6 + 4*np.random.random())  # 20bp minimum between fluors, 0.3 nm/bp so min 6nm max FRET at 10nm

    excitation_matrix = np.array([[1, 1, 0, 0, 1, 1, 1],
                                  [1, 1, 0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    print('Building training data...')
    for enc in range(1, 128):
        code = re.sub('0b', '', format(enc, '#0' + str(nbit+2) + 'b'))
        if code in code_set:
            numeric_code_list = np.array([int(a) for a in list(code)])
            simulated_spectra = np.zeros((simulation_per_code, nchannels))
            indices = [0,23,43,57,63]
            if numeric_code_list[6] == 1:
                error_scale = [0.25, 0.25, 0.35, 0.45]
            else:
                error_scale = [0.1, 0.25, 0.35, 0.45]
            for exc in range(4):
                relevant_fluorophores = numeric_code_list*excitation_matrix[exc, :]
                coefficients = np.zeros((simulation_per_code, 7))
                for i in range(simulation_per_code):
                    coefficients[i,:] = np.dot(fret_transfer_matrix[i,:,:], relevant_fluorophores)*relevant_fluorophores
                simulated_spectra_list = [coefficients[:,k][:,None]*np.random.multivariate_normal(spec_avg[k], spec_cov[k], simulation_per_code)[:,32:95] for k in range(nbit)]
                simulated_spectra[:,indices[exc]:indices[exc+1]] = np.sum(np.stack(simulated_spectra_list, axis = 2), axis = 2)[:,indices[exc]:indices[exc+1]]
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            for k in range(0,4):
                error_coefficient = error_scale[k] + (1-error_scale[k])*np.random.random(simulated_spectra_norm.shape[0])
                max_intensity = np.max(simulated_spectra_norm[:,indices[k]:indices[k+1]], axis = 1)
                max_intensity_error_simulation = error_coefficient*max_intensity
                error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = error_coefficient[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm/np.max(simulated_spectra_norm, axis = 1)[:,None]
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = numeric_code_list[6] or numeric_code_list[1] or numeric_code_list[0]
            ss_norm['c2'] = numeric_code_list[6] or numeric_code_list[0] or numeric_code_list[1] or numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c3'] = numeric_code_list[4] or numeric_code_list[5]
            ss_norm['c4'] = numeric_code_list[2] or numeric_code_list[3]
            ss_norm['code'] = code
            training_data = training_data.append(ss_norm, ignore_index = True)
            simulated_spectra_norm = simulated_spectra/np.max(simulated_spectra, axis = 1)[:,None]
            indices = [0,23,43,57,63]
            for k in range(0,4):
                simulated_spectra_norm[:,indices[k]:indices[k+1]] = (error_scale[k]*np.random.random(simulated_spectra_norm.shape[0]))[:,None]*simulated_spectra_norm[:,indices[k]:indices[k+1]]
            simulated_spectra_adjusted_norm = simulated_spectra_norm
            ss_norm = pd.DataFrame(simulated_spectra_adjusted_norm)
            ss_norm['c1'] = 0
            ss_norm['c2'] = 0
            ss_norm['c3'] = 0
            ss_norm['c4'] = 0
            ss_norm['code'] = '{}_error'.format(code)
            training_data_negative = training_data_negative.append(ss_norm, ignore_index = True)
    print('Training classifier...')
    training_data_full = pd.concat([training_data, training_data_negative])
    scaler_full = preprocessing.StandardScaler().fit(training_data_full.values[:,0:63])
    training_data_full_scaled = scaler_full.transform(training_data_full.values[:,0:63])
    umap_transform = umap.UMAP(n_neighbors = 25, metric = channel_cosine_intensity_7b_v2).fit(training_data.iloc[:,0:67], y = training_data.code.values)
    clf = [svm.SVC(C = 10, gamma = 1) for i in range(4)]
    clf[0].fit(training_data_full_scaled[:,0:23], training_data_full.c1.values)
    clf[1].fit(training_data_full_scaled[:,23:43], training_data_full.c2.values)
    clf[2].fit(training_data_full_scaled[:,43:57], training_data_full.c3.values)
    clf[3].fit(training_data_full_scaled[:,57:63], training_data_full.c4.values)
    clf_umap = svm.SVC(C = 10, gamma = 1, probability = True)
    clf_umap.fit(umap_transform.embedding_, training_data.code.values)
    # joblib.dump(clf, '{}/reference_simulate_{}_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_check_svc.pkl'.format(output_folder, str(spc)))
    # joblib.dump(clf_umap, '{}/reference_simulate_{}_{}_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_7b_svc.pkl'.format(output_folder, str(spc)))
    # joblib.dump(umap_transform, '{}/reference_simulate_{}_{}_interaction_simulated_excitation_adjusted_normalized_umap_transform_biofilm_7b.pkl'.format(output_folder, str(spc)))
    return(scaler_full, clf, clf_umap, umap_transform, training_data)


def main():
    parser = argparse.ArgumentParser('Mesure environmental microbial community spectral images')
    parser.add_argument('-c', '--config_filename', dest = 'config_filename', type = str, default='config.yaml', help = 'Input image filenames')
    args = parser.parse_args()

    # Load config file
    with open(args.config_filename, 'r') as f:
        config = yaml.safe_load(f)

    probe_design_filename = (config['probe_design_dir'] +
                                '/' + config['probe_design_filename']
                                )
    probe_design_basename = os.path.splitext(os.path.basename(probe_design_filename))[0]
    spc = config['ref_train_simulations']
    scaler_full, clf, clf_umap, umap_transform, training_data = load_training_data_simulate_reabsorption_excitation_adjusted_umap_transformed_with_fret_biofilm_7b_limited(
                    reference_folder=config['hipr_ref_dir'],
                    fret_folder=config['fret_dir'],
                    probe_design_filename=probe_design_filename,
                    spc=spc
                    )

    print('Saving trained objects...')
    output_folder = (
            config['output_dir']
            + '/' + config['reference_training']['out_dir']
            )
    pkl_fmt = (
                '{}'
                + '/NSIMS_{}'
                + '_PROBEDESIGN_{}'
                + '_FUNCTION_interaction_simulated_excitation_adjusted'
                    + '_normalized_umap_transformed_biofilm_7b'
                + '_OBJ_{}.pkl'
                )
    joblib.dump(
                scaler_full,
                pkl_fmt.format(
                    output_folder, str(spc), probe_design_basename, 'scaler'
                    ))
    joblib.dump(
                clf,
                pkl_fmt.format(
                    output_folder, str(spc), probe_design_basename, 'check_svc'
                    ))
    joblib.dump(
                clf_umap,
                pkl_fmt.format(
                    output_folder, str(spc), probe_design_basename, 'svc'
                    ))
    joblib.dump(
                umap_transform,
                pkl_fmt.format(
                    output_folder, str(spc), probe_design_basename, 'umap_transform'
                    ))
    training_data_fmt = re.sub('.pkl', '.csv', pkl_fmt)
    training_data.to_csv(
            training_data_fmt.format(
                    output_folder, str(spc), probe_design_basename, 'umap_transform'
                    ), index=False)

    print('Plotting UMAP...')
    fig, ax = plot_umap(umap_transform, training_data)
    plt.savefig(output_folder + '/{}_umap.png'.format(probe_design_basename))
    plt.close()

if __name__ == '__main__':
    main()
