import os
import argparse
import gr4h

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

###############################################################################
# CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--basin_id", type=str, help="basin_id: 01436500")
args = parser.parse_args()

###############################################################################
# GLOBAL PARAMETERS
## parsed
basin_id = args.basin_id

## defined
## Workflow-dependent
PATH_FORCING_DATA = "../forcing/"

YEAR_START_CALIBRATION = 1996 # 1995 as a warm-up
YEAR_END_CALIBRATION = 2009

YEAR_START_VALIDATION = 2011 # 2010 as a warm-up
YEAR_END_VALIDATION = 2019

CALIBRATION_DATA_LENGTHS = [1, 2, 3, 5, 7, 10, 14]

###############################################################################
# Reproducibility
np.random.seed(42)

###############################################################################
# Helpers
def nse(y_true, y_pred):
    return 1 - np.nansum((y_true-y_pred)**2)/np.nansum((y_true-np.nanmean(y_true))**2)

# Preparation of the periods
def periods_constructor(duration, 
                        year_start=YEAR_START_CALIBRATION, 
                        year_end=YEAR_END_CALIBRATION, 
                        stride=1):
    """
    Construction of individual calibration periods
    Input:
    duration: the required duration in calender years, int
    year_start: the first year considered, int
    year_end: the last year considered, int
    stride: default=1
    Output:
    the list of considered calender years
    """
    
    duration = duration - 1
    
    periods=[]
    
    while year_end - duration >= year_start:
        
        period = [year_end-duration, year_end]
        
        periods.append(period)
        
        year_end = year_end - stride
    
    return periods

###############################################################################
# Definition of a single experiment
def single_experiment(basin_id,
                      year_start,
                      year_end, 
                      path_forcing=PATH_FORCING_DATA):
    
    # creare folder for results
    results_folder = f'../results/{basin_id}/GR4H'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # read data for a particular basin
    data = pd.read_pickle(os.path.join(path_forcing, f'{basin_id}.pkl'))
    
    df_calib = data[str(year_start):str(year_end)].copy(deep=True)
    
    # add a full preceeding year as a warm-up period
    # add only meteorology, not runoff (--> Nan)
    df_calib_warmup = data[str(year_start-1)].copy(deep=True)
    df_calib_warmup["Q"] = np.nan
    
    df = pd.concat([df_calib_warmup, df_calib], axis=0)
        
    Q = df["Q"].to_numpy()
    T = df["T"].to_numpy()
    P = df["P"].to_numpy()
    PE= df["PET"].to_numpy()
    
    # len of warm-up that should be cropped
    warm_up_len = len(df_calib_warmup)
    
    # loss definition (MSE)
    def loss_gr4h(params, warm_up_len=warm_up_len):
        # calculate runoff
        Qsim = gr4h.run(T, P, PE, params)
        # MSE
        return np.nanmean((Q[warm_up_len:] - Qsim[warm_up_len:]) ** 2, axis=0)
    
    # optimization
    opt_par = differential_evolution(loss_gr4h, 
                                     bounds=gr4h.bounds(), 
                                     maxiter=100, 
                                     polish=True, 
                                     disp=False, 
                                     seed=42).x
    
    # save optimal parameters
    np.save(os.path.join(results_folder,f'{year_start}_{year_end}.npy'), opt_par)

    # calculate runoff with optimal parameters for the calibration period
    qsim_calibration = gr4h.run(T, P, PE, opt_par)
    
    # cut the warmup period
    qsim_calibration = qsim_calibration[warm_up_len:]
    
    eval_calibration = pd.DataFrame()
    eval_calibration["Qobs"] = data["Q"][f'{year_start}':f'{year_end}']
    eval_calibration["Qsim"] = qsim_calibration
    
    #########################################################################################
    # runoff simulation for the validation period
    df_val = data[f'{YEAR_START_VALIDATION}':f'{YEAR_END_VALIDATION}'].copy(deep=True) 
    
    # add a full preceeding year as a warm-up period
    # add only meteorology, not runoff (--> Nan)
    df_val_warmup = data[f'{YEAR_START_VALIDATION - 1}'].copy(deep=True) 
    df_val_warmup["Q"] = np.nan
    
    df = pd.concat([df_val_warmup, df_val], axis=0)
        
    Q = df["Q"].to_numpy()
    T = df["T"].to_numpy()
    P = df["P"].to_numpy()
    PE= df["PET"].to_numpy()
    
    # len of warm-up that should be cropped
    warm_up_len = len(df_val_warmup)
    
    # calculate runoff with optimal parameters for the calibration period
    qsim_validation = gr4h.run(T, P, PE, opt_par)
    
    # cut the warmup period
    qsim_validation = qsim_validation[warm_up_len:]
    
    eval_validation = pd.DataFrame()
    eval_validation["Qobs"] = data["Q"][f'{YEAR_START_VALIDATION}':f'{YEAR_END_VALIDATION}']
    eval_validation["Qsim"] = qsim_validation
    
    # save the results
    eval_calibration.to_pickle(os.path.join(results_folder,
                                            f'calibration_{year_start}_{year_end}.pkl'))
    
    eval_validation.to_pickle(os.path.join(results_folder,
                                           f'validation_{year_start}_{year_end}.pkl'))

    # briefly calculate and report the statistics
    print(f'Results for GR4H, {basin_id}, calibration period from {year_start} to {year_end}')
    print(f'NSE on calibration: {np.round(nse(eval_calibration["Qobs"], eval_calibration["Qsim"]), 2)}')
    print(f'NSE on validation: {np.round(nse(eval_validation["Qobs"], eval_validation["Qsim"]), 2)}')


###############################################################################
# MAIN EXPERIMENT
###############################################################################

# loop over the predefined list of calibration data lengths
for period_length in CALIBRATION_DATA_LENGTHS:
    
    # create a list of available calibration periods
    periods = periods_constructor(period_length)
    
    # loop over all available periods
    for period in periods:
        
        year_start, year_end = period
        
        single_experiment(basin_id=basin_id, 
                          year_start=year_start, 
                          year_end=year_end)
