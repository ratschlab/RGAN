import itertools
import numpy as np
from os import system
from time import sleep


def submit_job(params, use_gpu=False):

    if use_gpu:
        cmd_line = 'bsub -W 2:00 -n 1 -R "rusage[mem=1000,ngpus_excl_p=1]"'
    else:
        cmd_line = 'bsub -W 2:00 -n 1 -R "rusage[mem=1000]"'

    job_name = "_".join(map(lambda x: str(x), params.values()))

    cmd_line += ' -J %s -o %s.txt'%(job_name, job_name)
        

    cmd_line += ' python sine_generation_RVAE_new.py '

    for key, val in params.items():
        cmd_line += ' -%s %s'%(key, val)
    
    print(cmd_line)    
    system(cmd_line)

if __name__ == '__main__':

    learning_rate = [0.001, 0.01]
    optimizer_str = ["adam"]
    hidden_units_dec = [10, 50, 100, 300]
    hidden_units_enc = [10, 50, 100, 300]
    emb_dim = [10, 50, 100, 300]
    mult = [0.1, 0.001, 0.01]

    configs = itertools.product(learning_rate, optimizer_str, hidden_units_dec, hidden_units_enc, emb_dim, mult)
    config_keys = ['learning_rate', 'optimizer_str', 'hidden_units_dec', 'hidden_units_enc', 'emb_dim', 'mult']

    for config in configs:
        params = {}
        params['learning_rate'] = config[0]
        params['optimizer_str'] = config[1]
        params['hidden_units_dec'] = config[2]
        params['hidden_units_enc'] = config[3]
        params['emb_dim'] = config[4]
        params['mult'] = config[5]
        params['experiment_id'] = 'experiments_test_RVAE_sine_SRNN_new_RVAE_HS_short'

        #if((params['hidden_units_dec'] == 300) or (params['hidden_units_enc'] == 300) or (params['emb_dim'] == 300) or (params['mult'] == 0.01)):
        submit_job(params, use_gpu=True)