import os
import argparse
import resource

import math
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras


###############################################################################
# CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--basin_id", type=str, help="basin_id: 01487000")
parser.add_argument("--model_name", type=str, help="model name: GRU")
parser.add_argument("--gpu", type=str, help="gpu number to use: 0...3")
parser.add_argument("--batch_size", type=int, default=256, help="size of minibatch: 256 by default")
parser.add_argument("--history", type=int, default=720*6, help="length of lookback period: 6*720 by default")
args = parser.parse_args()

###############################################################################
# GLOBAL PARAMETERS
## parsed
basin_id       = args.basin_id
model_name     = args.model_name
gpu            = args.gpu
batch_size     = args.batch_size
history        = args.history

## defined
## ANN-dependent
LOSS = "mse"
BATCH_SIZE = batch_size
EPOCHS = 100
PATIENCE = 5
HISTORY = history # 6 months by default
HIDDEN_STATES = [20] # [5, 10, 20]

## Workflow-dependent
PATH_FORCING_DATA = "../forcing/"

YEAR_START_CALIBRATION = 1996 # 1995 as a warm-up
YEAR_END_CALIBRATION = 2009

YEAR_START_VALIDATION = 2011 # 2010 as a warm-up
YEAR_END_VALIDATION = 2019

CALIBRATION_DATA_LENGTHS = [1, 2, 3, 5, 7, 10, 14]

###############################################################################
# GPU environment preparation
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


###############################################################################
# Setting up the reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

###############################################################################
# Constructor for model definition
def constructor(model_name="GRU", 
                hidden_state=20,
                history=720, 
                hindcast=1,
                input_vars=3, 
                loss=LOSS, 
                optimizer=keras.optimizers.Adam()):
    
    # model instance initialization
    model = keras.models.Sequential()
    
    # add a core layer
    if model_name == "GRU":
        model.add(keras.layers.GRU(hidden_state, return_sequences=False, input_shape=(history, input_vars)))
    elif model_name == "LSTM":
        model.add(keras.layers.LSTM(hidden_state, return_sequences=False, input_shape=(history, input_vars)))
    
    # add the Dense layer on top
    model.add(keras.layers.Dense(hindcast))
    
    # compilation
    model.compile(loss=loss, optimizer=optimizer)

    return model

###############################################################################
# Data generator for calibration
class Generator_calibration(keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, 
                 data_instance, 
                 year_start, 
                 year_end, 
                 history=720, 
                 batch_size=256, 
                 validation_split=0.25, 
                 mode="train"):
        
        self.data_instance = data_instance
        self.year_start, self.year_end = year_start, year_end
        self.history = history
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.mode = mode

        df_calib = self.data_instance[str(self.year_start):str(self.year_end)].copy(deep=True)
    
        # add a full preceeding year as a warm-up period
        # add only meteorology, not runoff (--> Nan)
        df_warmup = self.data_instance[str(self.year_start-1)].copy(deep=True)
        df_warmup["Q"] = np.nan

        df = pd.concat([df_warmup, df_calib], axis=0)
        
        self.X_matrix = df[["T", "P", "PET"]].to_numpy()
        self.y_matrix = df[["Q"]].to_numpy()

        self.X_mean = np.nanmean(self.X_matrix)
        self.X_std = np.nanstd(self.X_matrix)
        self.y_mean = np.nanmean(self.y_matrix)
        self.y_std = np.nanstd(self.y_matrix)

        self.indices = np.array([i for i in np.arange(self.y_matrix.shape[0]) if not np.isnan(self.y_matrix[i])])
        np.random.shuffle(self.indices)
        
        self.test_samples_number = math.ceil(len(self.indices)*self.validation_split)

        self.indices_train = self.indices[self.test_samples_number:]
        self.indices_test = self.indices[:self.test_samples_number]


    def __len__(self):
        if self.mode == "train": 
            return math.ceil( (len(self.indices) - self.test_samples_number) / self.batch_size) #number of batches in train
        elif self.mode == "test":
            return math.ceil( self.test_samples_number / self.batch_size) # number of batches in test

    def __getitem__(self, idx):   

        inds_train = self.indices_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        inds_test = self.indices_test[idx * self.batch_size:(idx + 1) * self.batch_size]   

        # form batches from raw data
        batch_x_train = np.array([self.X_matrix[i-self.history:i,::] for i in inds_train])
        batch_y_train = np.array([self.y_matrix[i,::] for i in inds_train]).reshape(-1)

        batch_x_test = np.array([self.X_matrix[i-self.history:i,::] for i in inds_test])
        batch_y_test = np.array([self.y_matrix[i,::] for i in inds_test]).reshape(-1)

        # normalization: substract mean and divide by std
        batch_x_train -= self.X_mean
        batch_x_train /= self.X_std

        batch_x_test -= self.X_mean
        batch_x_test /= self.X_std

        batch_y_train -= self.y_mean
        batch_y_train /= self.y_std

        batch_y_test -= self.y_mean
        batch_y_test /= self.y_std
        
        if self.mode == "train":
            return batch_x_train, batch_y_train
        elif self.mode == "test":
            return batch_x_test, batch_y_test
    
    def on_epoch_end(self):
        # shuffle train indices after each epoch
        np.random.shuffle(self.indices_train)

###############################################################################
# Data generator for validation
class Generator_validation(keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, 
                 data_instance, 
                 year_start, 
                 year_end,
                 norms, 
                 history=720, 
                 batch_size=256):
        
        self.data_instance = data_instance
        self.year_start, self.year_end = year_start, year_end
        self.history = history
        self.batch_size = batch_size
        self.X_mean, self.X_std, self.y_mean, self.y_std = norms

        df_calib = self.data_instance[str(self.year_start):str(self.year_end)].copy(deep=True)
    
        # add a full preceeding year as a warm-up period
        # add only meteorology, not runoff (--> Nan)
        df_warmup = self.data_instance[str(self.year_start-1)].copy(deep=True)
        df_warmup["Q"] = np.nan

        df = pd.concat([df_warmup, df_calib], axis=0)
        
        self.X_matrix = df[["T", "P", "PET"]].to_numpy()
        self.y_matrix = df[["Q"]].to_numpy()

        self.indices = np.arange(len(df_warmup), len(df))
        

    def __len__(self):
        return math.ceil( len(self.indices) / self.batch_size) #number of batches
        
    def __getitem__(self, idx):   

        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # form batches from raw data
        batch_x_val = np.array([self.X_matrix[i-self.history:i,::] for i in inds])
        batch_y_val = np.array([self.y_matrix[i,::] for i in inds]).reshape(-1)

        # normalization: substract mean and divide by std --> derived from calibration generator
        batch_x_val -= self.X_mean
        batch_x_val /= self.X_std

        batch_y_val -= self.y_mean
        batch_y_val /= self.y_std
        
        return batch_x_val
    
    def on_epoch_end(self):
        pass # to nothing as it is only a single epoch

###############################################################################
# HELPERS
# NSE
def nse(y_true, y_pred):
    return 1 - np.nansum((y_true-y_pred)**2)/np.nansum((y_true-np.nanmean(y_true))**2)


# Memory leakage detection
class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


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
                      model_name="GRU",
                      hidden_state=20,
                      history=720,
                      epochs=300,
                      patience=5,
                      batch_size=256, 
                      path_forcing='/content'):
    
    # creare folder for results
    results_folder = f'../results/{basin_id}/{model_name}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # read data for a particular basin
    data = pd.read_pickle(os.path.join(path_forcing, f'{basin_id}.pkl'))
    
    # initialize model
    model = constructor(model_name=model_name,
                        hidden_state=hidden_state,
                        history=history)
    
    # prepare generators for the calibration phase
    gen_train = Generator_calibration(data_instance=data,
                                      year_start=year_start,
                                      year_end=year_end,
                                      history=history,
                                      batch_size=batch_size,
                                      validation_split=0.25,
                                      mode="train")
    
    gen_test  = Generator_calibration(data_instance=data,
                                      year_start=year_start,
                                      year_end=year_end,
                                      history=history,
                                      batch_size=batch_size,
                                      validation_split=0.25,
                                      mode="test")
    
    # prepare generator for the evaluation on the calibration data
    gen_eval_calibration= Generator_validation(data_instance=data,
                                               year_start=year_start,
                                               year_end=year_end,
                                               norms=[gen_train.X_mean, gen_train.X_std, gen_train.y_mean, gen_train.y_std],
                                               history=history,
                                               batch_size=batch_size)
    
    # prepare generator for the evaluation on the validation data
    gen_eval_validation = Generator_validation(data_instance=data,
                                               year_start=YEAR_START_VALIDATION,
                                               year_end=YEAR_END_VALIDATION, 
                                               norms=[gen_train.X_mean, gen_train.X_std, gen_train.y_mean, gen_train.y_std],
                                               history=history,
                                               batch_size=batch_size)
    
    # create a filepath to save the model
    path_for_model = os.path.join(results_folder,
                                  f'{year_start}_{year_end}_{hidden_state}_{history}.h5')
    
    # model calibration on training data
    # with performance control on test data
    model_log = model.fit(gen_train,
                          validation_data=gen_test,
                          epochs=epochs,
                          callbacks=[keras.callbacks.EarlyStopping(patience=patience),
                                     keras.callbacks.ModelCheckpoint(filepath=path_for_model,
                                                                     save_best_only=True,
                                                                     save_weights_only=True)], 
                          verbose=2)
    
    # save model log
    np.save(os.path.join(results_folder,
                         f'{year_start}_{year_end}_{hidden_state}_{history}_log.npy'), model_log.history)

    # find an epoch number with the lowest loss on test
    # to use this number to finetune the model on test data
    # That secures an utilization of FULL calibration period 
    epochs_for_finetuning = np.argmin(model_log.history["val_loss"]) + 1

    # model finetuning on test data
    # for the same number of epochs used to train the model
    model.load_weights(path_for_model)
    model.fit(gen_test,
              epochs=epochs_for_finetuning, 
              verbose=2)
    
    # save weights of finetuned model
    model.save_weights(path_for_model)

    # evaluation of model performance on calibration period
    qsim_calibration = model.predict(gen_eval_calibration, 
                                     batch_size=batch_size)
    
    # post-processing of predictions
    qsim_calibration *= gen_train.y_std
    qsim_calibration += gen_train.y_mean

    eval_calibration = pd.DataFrame()
    eval_calibration["Qobs"] = data["Q"][f'{year_start}':f'{year_end}']
    eval_calibration["Qsim"] = qsim_calibration

    # evaluation of model performance on validation period
    qsim_validation = model.predict(gen_eval_validation,
                                    batch_size=batch_size)
    
    # post-processing of predictions
    qsim_validation *= gen_train.y_std
    qsim_validation += gen_train.y_mean

    eval_validation = pd.DataFrame()
    eval_validation["Qobs"] = data["Q"][f'{YEAR_START_VALIDATION}':f'{YEAR_END_VALIDATION}'] 
    eval_validation["Qsim"] = qsim_validation

    # save the results
    eval_calibration.to_pickle(os.path.join(results_folder,
                                            f'calibration_{year_start}_{year_end}_{hidden_state}_{history}.pkl'))
    
    eval_validation.to_pickle(os.path.join(results_folder,
                                           f'validation_{year_start}_{year_end}_{hidden_state}_{history}.pkl'))
    
    # briefly calculate and report the statistics
    print(f'Results for {model_name} trained with {hidden_state} states and {history} lookback steps for {epochs_for_finetuning} epochs')
    print(f'NSE on calibration: {np.round(nse(eval_calibration["Qobs"], eval_calibration["Qsim"]), 2)}')
    print(f'NSE on validation: {np.round(nse(eval_validation["Qobs"], eval_validation["Qsim"]), 2)}')

    # clear session for keras
    del model
    keras.backend.clear_session()


###############################################################################
# MAIN EXPERIMENT
# loop over the predefined list of calibration data lengths
for period_length in CALIBRATION_DATA_LENGTHS:
    
    # create a list of available calibration periods
    periods = periods_constructor(period_length)
    
    # loop over all available periods
    for period in periods:
        
        # define the first and the last years
        year_start, year_end = period
        
        # loop over the predefined list of hidden states to tune to
        for hidden_state in HIDDEN_STATES:
            
            print(f'{basin_id}, {year_start}, {year_end}')
        
            single_experiment(basin_id=basin_id, 
                              year_start=year_start, 
                              year_end=year_end, 
                              model_name=model_name,  
                              hidden_state=hidden_state,
                              history=HISTORY,
                              epochs=EPOCHS,
                              patience=PATIENCE,
                              batch_size=BATCH_SIZE,
                              path_forcing=PATH_FORCING_DATA)

