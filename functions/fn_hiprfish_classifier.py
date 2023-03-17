import numpy as np
import pandas as pd
import numba
import os
import re
import matplotlib.pyplot as plt

# =============================================================================
# ## Functions
# =============================================================================

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


def plot_umap(embedding, labels, dims=(5,5), marker='o', alpha=0.5,
        markersize=1, ft=8, line_col='k', xlims=[], ylims=[]):
    embedding_df = pd.DataFrame(embedding)
    embedding_df['numeric_barcode'] = labels
    fig, ax = general_plot(dims=dims, col=line_col)
    barcodes = np.unique(labels)
    n_barcodes = barcodes.shape[0]
    cmap = plt.get_cmap('jet')
    delta = 1/n_barcodes
    color_list = [cmap(i*delta) for i in range(n_barcodes)]
    col_df = pd.DataFrame(columns=['numeric_barcode','color'])
    col_df['numeric_barcode'] = barcodes
    # col_df['numeric_barcode'] = col_df['numeric_barcode'].astype(int)
    col_df['color'] = color_list
    embedding_df = embedding_df.merge(col_df, how='left', on='numeric_barcode')
    x = embedding_df.iloc[:,0].values
    y = embedding_df.iloc[:,1].values
    colors_plot = embedding_df['color'].values.tolist()
    # ax.plot(x, y, marker, alpha=alpha, ms=markersize, rasterized=True)
    ax.scatter(x, y, marker=marker, c=colors_plot, alpha=alpha, s=markersize, rasterized=True)
    lab_dict = {}
    for x_, y_, bc in zip(x,y, labels):
        try:
            _ = lab_dict[bc]
            pass
        except:
            if any(xlims) or any(ylims):
                if (xlims[0] < x_ < xlims[1]) and (ylims[0] < y_ < ylims[1]):
                    ax.text(x_, y_, str(bc), fontsize=ft)
                    lab_dict[bc] = 1
            else:
                ax.text(x_, y_, str(bc), fontsize=ft)
                lab_dict[bc] = 1

    # for i in range(n_barcodes):
    #     enc = i+1
    #     emd = embedding_df.loc[embedding_df.numeric_barcode.values == enc]
    #     ax.plot(emd.iloc[:,0], emd.iloc[:,1], 'o', alpha = 0.5, color = color_list[i], markersize = 1, rasterized = True)
    ax.set_aspect('equal')
    ax.set_xlabel('UMAP 1', fontsize=ft, color = line_col)
    ax.set_ylabel('UMAP 2', fontsize=ft, color = line_col, labelpad = -1)
    if any(xlims) or any(ylims):
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
    return fig, ax, col_df

@numba.njit()
def channel_cosine_intensity_5b_v2(x, y):
    check = np.sum(np.abs(x[57:60] - y[57:60]))
    if check < 0.01:
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        if x[57] == 0:
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
        if x[58] == 0:
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
        if x[59] == 0:
            cos_weight_3 = 0.0
            cos_dist = 0.0
        else:
            cos_weight_3 = 1.0
            for i in range(43,57):
                result += x[i] * y[i]
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                cos_dist = 0.0
            elif norm_x == 0.0 or norm_y == 0.0:
                cos_dist = 1.0
            else:
                cos_dist = 1.0 - (result / np.sqrt(norm_x * norm_y))
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
    #     if x[60] == 0:
    #         cos_weight_4 = 0.0
    #         cos_dist_4 = 0.0
    #     else:
    #         cos_weight_4 = 1.0
    #         for i in range(57,63):
    #             result += x[i] * y[i]
    #             norm_x += x[i] ** 2
    #             norm_y += y[i] ** 2
    #         if norm_x == 0.0 and norm_y == 0.0:
    #             cos_dist_4 = 0.0
    #         elif norm_x == 0.0 or norm_y == 0.0:
    #             cos_dist_4 = 1.0
    #         else:
    #             cos_dist_4 = 1.0 - (result / np.sqrt(norm_x * norm_y))
    #     cos_dist = 0.5*(cos_dist_1 + cos_dist_2 + cos_dist_3 + cos_dist_4)/4
    else:
        cos_dist = 1
    return(cos_dist)


@numba.njit()
def euclid_dist_cumul_spec(s0, s1):
    d_ecs = 0
    for i in range(s0.shape[0]-1):
        a0 = 0.5*(s0[i]+s0[i+1])  # Trapezoid area 0.5*(a+b)*h, here we just set h as 1
        a1 = 0.5*(s1[i]+s1[i+1])
        d_ecs += (a0 - a1)**2
    return (d_ecs)**(1/2)

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



def get_7b_params():
    '''
    Parameters for hiprfish barcodes using Merfish R1, R2, R3, R6, R7, R8, R10
    With lasers 488, 514, 561, 633
    '''
    excitation_matrix = np.array([[1, 1, 0, 0, 1, 1, 1],
                                  [1, 1, 0, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    error_scale = {
            'special':[6,[0.25, 0.25, 0.35, 0.45]],
            'standard':[0.1, 0.25, 0.35, 0.45]
            }
    molar_extinction_coefficient = [
            73000, 112000, 120000, 144000, 270000, 50000, 81000
            ]
    fluorescence_quantum_yield = [0.92, 0.79, 1, 0.33, 0.33, 1, 0.61]
    rough_classifier_matrix =   [[1,1,0,0,0,0,1],
                                [1,1,0,0,1,1,1],
                                [0,0,0,0,1,1,0],
                                [0,0,1,1,0,0,0]]
    return {
            'nbit': 7,
            'n_las': 4,
            'barcode_list': [512, 128, 64, 32, 4, 2, 1],
            'merfish_fluors': [10,8,7,6,3,2,1],
            'channel_indices': [0,23,43,57,63],
            'ref_spec_indices': [32,95],
            'excitation_matrix': excitation_matrix,
            'error_scale': error_scale,
            'molar_extinction_coefficient': molar_extinction_coefficient,
            'fluorescence_quantum_yield': fluorescence_quantum_yield,
            'rough_classifier_matrix': rough_classifier_matrix
            }


# =============================================================================
# ## Classes
# =============================================================================

class Training_Data:
    '''
    Load training data: simulate reabsorption, excitation adjusted, with fret,
    limited to only barcodes in probe design. Requires excitation information
    on the fluors in the "fret folder". Also requires single fluor reference
    spectra for in the "reference folder"

    config: hiprfish configuration filename used in pipeline
    params: hardcoded parameters for a given number of bits and lasers
            (functions for extracting these parameters are defined above)
    '''
    def __init__(self, config, params):
        self.spc = config['ref_train_simulations']
        self.nbit = params['nbit']
        self.probe_design_filename = (
                config['__default__']['PROBE_DESIGN_DIR']
                        + '/' + config['probe_design_filename']
                        )
        self.code_col = config['probe_design_barcode_column']
        self.excitation_matrix = params['excitation_matrix']
        self.ind = params['channel_indices']
        self.n_las = self.excitation_matrix.shape[0]
        self.n_chan = self.ind[-1] - self.ind[0]
        self.es_dict = params['error_scale']
        self.barcode_list = params['barcode_list']
        self.ref_dir = config['hipr_ref_dir']
        self.ref_files_fmt = config['ref_files_fmt']
        self.rsi = params['ref_spec_indices']
        self.fret_dir = config['fret_dir']
        self.exc_fmt = config['excitation_files_fmt']
        self.molar_extinction_coefficient = params['molar_extinction_coefficient']
        self.fluorescence_quantum_yield = params['fluorescence_quantum_yield']
        self.fluorophores = params['merfish_fluors']
        self.dmin = config['fret_dist_min']
        self.drange = config['fret_dist_max'] - self.dmin
        self.rc_matrix = params['rough_classifier_matrix']


    # ==================
    # Internal functions

    def _add_rough_class_columns(self, numeric_code_list, ss_norm, neg=False):
        nc_arr = np.array(numeric_code_list)
        shp = ss_norm.shape[0]
        for ch in range(self.n_las):
            # If the code should not show up in this laser, or you are simulating negative data, give a zero
            bool = not any(nc_arr * np.array(self.rc_matrix[ch]))
            if neg or bool:
                col = np.zeros((shp ,1), dtype=int)
            # If the code should show up in this laser, give a one
            else:
                col = np.ones((shp ,1), dtype=int)
            ss_norm = np.hstack([ss_norm, col])
        return ss_norm


    def _add_error_to_spectra(self, numeric_code_list, simulated_spectra_norm, pos=True):
        simulated_spectra_err = np.zeros(simulated_spectra_norm.shape)
        if numeric_code_list[self.es_dict['special'][0]] == 1:
            error_scale = self.es_dict['special'][1]
        else:
            error_scale = self.es_dict['standard']
        for k in range(0,self.n_las):
            ec_rand = np.random.random(simulated_spectra_norm.shape[0])
            spec_slice = simulated_spectra_norm[:,self.ind[k]:self.ind[k+1]]
            if pos:
                error_coefficient = error_scale[k] + (1-error_scale[k])*ec_rand
                max_intensity = np.max(spec_slice, axis = 1)
                max_intensity_error_simulation = error_coefficient*max_intensity
                error_coefficient[max_intensity_error_simulation < error_scale[k]] = 1
                ss_err = error_coefficient[:,None]*spec_slice
            else:
                ss_err = error_scale[k]*ec_rand[:,None]*spec_slice
            simulated_spectra_err[:,self.ind[k]:self.ind[k+1]] = ss_err
        return simulated_spectra_err


    def _simulate_spectra(self, numeric_code_list):
        # params
        simulated_spectra = np.zeros((self.spc, self.n_chan))
        for exc in range(self.n_las):
            relevant_fluorophores = numeric_code_list*self.excitation_matrix[exc, :]
            coefficients = np.zeros((self.spc, self.nbit))
            for i in range(self.spc):
                coefficients[i,:] = np.dot(
                        self.fret_transfer_matrix[i,:,:],
                        relevant_fluorophores
                        )*relevant_fluorophores
            spec_avg_sub = [s[self.ind[exc]:self.ind[exc+1]] for s in self.spec_avg]
            spec_cov_sub = [
                    s[self.ind[exc]:self.ind[exc+1], self.ind[exc]:self.ind[exc+1]]
                    for s in self.spec_cov
                    ]
            simulated_spectra_list = []
            for k in range(self.nbit):
                coeff = coefficients[:,k]
                if any(coeff):
                    rand_spec = np.random.multivariate_normal(
                            spec_avg_sub[k], spec_cov_sub[k], self.spc
                            )
                    sim_spec = coeff[:,None] * rand_spec
                else:
                    sim_spec = np.zeros((self.spc, spec_avg_sub[k].shape[0]))
                simulated_spectra_list.append(sim_spec)
            sim_spec_las_stack = np.stack(simulated_spectra_list, axis=2)
            sim_spec_sum = np.sum(sim_spec_las_stack, axis=2)
            simulated_spectra[:,self.ind[exc]:self.ind[exc+1]] = sim_spec_sum
        return simulated_spectra


    def _get_single_fluor_reference_spectra(self):
        ref_files_fmt = self.ref_dir + '/' + self.ref_files_fmt
        files = [ref_files_fmt.format(b) for b in self.barcode_list]
        spec_avg = [np.average(pd.read_csv(f, header = None), axis = 0) for f in files]
        spec_cov = [np.cov(pd.read_csv(f, header = None).transpose()) for f in files]
        self.spec_avg = [s[self.rsi[0]:self.rsi[1]] for s in spec_avg]
        self.spec_cov = [
                s[self.rsi[0]:self.rsi[1], self.rsi[0]:self.rsi[1]]
                for s in spec_cov
                ]


    def _calculate_fret_efficiency(self, distance):
        # files = glob.glob(data_folder + '/' + exc_fmt.format('*'))
        # samples = [re.sub('_excitation.csv', '', os.path.basename(file)) for file in files]
        exc_fmt = self.fret_dir + '/' + self.exc_fmt
        forster_distance = np.zeros((self.nbit,self.nbit))
        fret_transfer_matrix = np.eye(self.nbit)
        kappa_squared = 2/3
        ior = 1.4
        NA = 6.022e23
        Qd = 1
        prefactor = 2.07*kappa_squared*Qd/(128*(np.pi**5)*(ior**4)*NA)*(1e17)
        for i in range(self.nbit):
            for j in range(self.nbit):
                if i != j:
                    fi = pd.read_csv(exc_fmt.format(str(self.fluorophores[i])))
                    fj = pd.read_csv(exc_fmt.format(str(self.fluorophores[i])))
                    emission_max_i = np.argmax(fi.Emission.values)
                    emission_max_j = np.argmax(fj.Emission.values)
                    if emission_max_i < emission_max_j:
                        fi_norm = np.clip(fi.Emission.values/fi.Emission.sum(), 0, 1)
                        fj_norm = np.clip(fj.Excitation.values/fj.Excitation.max(), 0, 1)
                        j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                        C = self.molar_extinction_coefficient[j]
                        Y = self.fluorescence_quantum_yield[i]
                        forster_distance[i,j] = np.power(prefactor*j_overlap*C*Y, 1/6)
                    else:
                        fi_norm = np.clip(fi.Excitation.values/fi.Excitation.max(), 0, 1)
                        fj_norm = np.clip(fj.Emission.values/fj.Emission.sum(), 0, 1)
                        j_overlap = np.sum(fi_norm*fj_norm*fi.Wavelength.values**4)
                        C = self.molar_extinction_coefficient[i]
                        Y = self.fluorescence_quantum_yield[j]
                        forster_distance[i,j] = np.power(prefactor*j_overlap*C*Y, 1/6)
                    sgn = np.sign(emission_max_i - emission_max_j)
                    fret_transfer_matrix[i,j] = sgn*1/(1+(distance/forster_distance[i,j])**6)
        return(fret_transfer_matrix)


    def _get_fret_transfer_matrix(self):
        fret_transfer_matrix = np.zeros((self.spc, self.nbit, self.nbit))
        for i in range(self.spc):
            dist = self.dmin + self.drange*np.random.random()
            fret_transfer_matrix[i,:,:] = self._calculate_fret_efficiency(dist)
        self.fret_transfer_matrix = fret_transfer_matrix


    # =================
    # Wrapper function

    def get_training_data(self):
        # Get fret transfer matrix
        print('Calculating FRET efficiency...')
        self._get_fret_transfer_matrix()

        # Construct training data
        print('Building training data...')
        # Load reference spectra
        self._get_single_fluor_reference_spectra()
        # Which barcodes are in the probe design
        probes = pd.read_csv(self.probe_design_filename, dtype={self.code_col: str})
        code_set = np.unique(probes[self.code_col].values)
        # Construct table
        ncols = self.n_chan + self.n_las + 1
        training_data = np.empty([0, ncols])
        training_data_negative = np.empty([0, ncols])
        for enc in range(1, 2**self.nbit): # Iterate over all possible barcodes
            code = re.sub('0b', '', format(enc, '#0' + str(self.nbit+2) + 'b'))
            if code in code_set:  # Limit to only those barcodes in the probe design
                numeric_code_list = np.array([int(a) for a in list(code)])

                ### Simulate many spectra with random fret and random intensity
                simulated_spectra = self._simulate_spectra(numeric_code_list)
                ss_max = np.max(simulated_spectra, axis = 1)[:,None]
                simulated_spectra_norm = simulated_spectra / ss_max
                # Add error to the spectra
                simulated_spectra_err = self._add_error_to_spectra(
                        numeric_code_list, simulated_spectra_norm
                        )
                sse_max = np.max(simulated_spectra_err, axis = 1)[:,None]
                ss_norm = simulated_spectra_err / sse_max
                # Add rough classifier ground truth columns: 1 or 0 for each laser
                ss_norm = self._add_rough_class_columns(numeric_code_list, ss_norm)
                # Add code ground truth column
                code_col = np.repeat(code, ss_norm.shape[0])[:,None]
                ss_norm = np.hstack([ss_norm, code_col])
                # Concatenate new spectra to the training data
                training_data = np.vstack([training_data, ss_norm])

                ### Get the negative training spectra
                simulated_spectra_neg = self._add_error_to_spectra(
                        numeric_code_list, simulated_spectra_norm, pos=False
                        )
                # Get the negative rough classifier columns
                simulated_spectra_neg = self._add_rough_class_columns(
                        numeric_code_list, simulated_spectra_neg, neg=True
                        )
                # Add negative code column
                code_col_neg = np.repeat(
                        '{}_error'.format(code), simulated_spectra_neg.shape[0]
                        )[:,None]
                simulated_spectra_neg = np.hstack([simulated_spectra_neg, code_col_neg])
                # Concatenate new spectra to the training data
                training_data_negative = np.vstack(
                        [training_data_negative, simulated_spectra_neg]
                        )
        return(training_data, training_data_negative)
