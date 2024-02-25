# Python script for use in hiprfish processing pipeline

import sys
import yaml
import argparse
import numpy as np
import pandas as pd

def main():
    """
    Construct training data for 7bit hiprfish by simulating random fret
    interactions
    """
    parser = argparse.ArgumentParser('')
    parser.add_argument('-c', '--config_filename', dest = 'config_filename', type = str, default='config.yaml', help = 'Configuration file name')
    args = parser.parse_args()

    # Load config file
    with open(args.config_filename, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_hiprfish_classifier as fhc

    # Get parameters for training data
    params = fhc.get_7b_params()

    # Get training data
    print('Get training data')
    trn = fhc.Training_Data(config, params)
    training_data, training_data_negative = trn.get_training_data()

    # Save training data
    print('Saving training data...')
    out_dir = config['reference_training']['out_dir']
    pid = config['probe_design_id']
    spc = config['ref_train_simulations']
    out_fmt = config['clf_fmt'].format(pid, spc, '{}')
    names = [config['training_data_name'], config['training_data_neg_name']]
    data_list = [training_data, training_data_negative]
    for name, data in zip(names, data_list):
        out_fn = out_dir + '/' + out_fmt.format(name) + '.csv'
        pd.DataFrame(data).to_csv(out_fn, index=False)

if __name__ == '__main__':
    main()
