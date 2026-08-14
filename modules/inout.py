"""!
@file inout.py
@brief Functions to manage input and output

This file contains the functions to manage inputs and outputs
from Pandas dataframes and text files. Its definitions are
organized in 5 groups:

1) Directory Management

2) Input from dataframe

3) Output to dataframe

4) Output to text files

5) Outputs relative to training
"""

import os
import numpy as np
import pandas as pd

from modules.iec import IECparams
from modules.control import mae, mse, rmse, obs_matrix, system_poles

import train_params as par



# ########### #
# Directories #
# ########### #

def mkdir(dir_name: str) -> None:
    """! @brief Creates the folder if not existing
    @param dir_name: string, name of the folder
    @return None
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return



# ################ #
# Dataframe Inputs #
# ################ #

def df_to_inputs(df_path:str) -> (
        np.ndarray[:], np.ndarray[:], np.ndarray[:], np.ndarray[:]
):
    """! @brief Extracts the PT inputs from a given dataframe
    @param df_path: string, path of the dataframe
    @return t: np.ndarray[T], time index of the sample
    @return delta_t: np.ndarray[T], time between current and previous sample
    @return K: np.ndarray[T], load factor's time series
    @return theta_a: np.ndarray[T], ambient temperature's time series

    This function reads a Pandas dataframe and extracts the time series
    relative to the inputs of a Power Transformer Model, in the form of NumPy arrays:
    time index, time delta, load factor and ambient temperature.
    In case the dataframe contains more fields those are ignored.
    """

    df = pd.read_csv(
        df_path,
        delimiter=";",
        #skiprows=[0,2],
        parse_dates=["t_d"],
        date_format="%d/%m/%Y %H:%M"
    )

    df["delta_t"] = df["t_d"].diff()
    delta_t_seconds = df["delta_t"].dt.total_seconds().to_numpy()
    delta_t_seconds[0] = 0. # Otherwise it is Nan
    delta_t = np.int_( np.round( delta_t_seconds / 60 ))

    t = np.cumsum(delta_t)
    K = df["K"].to_numpy()
    theta_a = df["theta_a"].to_numpy()

    return t, delta_t, K, theta_a



def df_to_outputs(df_path:str) -> (np.ndarray[:], np.ndarray[:]):
    """! @brief Extracts the PT reference outputs from a given dataframe
    @param df_path: string, path of the dataframe
    @return theta_o: np.ndarray[T], top-oil temperature's time series
    @return theta_h: np.ndarray[T], hotspot temperature's time series

    This function reads a Pandas dataframe and extracts the time series
    relative to the outputs of a Power Transformer Model, in the form of NumPy arrays:
    top-oil temperature and hotspot temperature.
    In case the dataframe contains more fields those are ignored.
    """

    df = pd.read_csv(
        df_path,
        delimiter=";",
        #skiprows=[0,2],
    )

    theta_o = df["theta_o"].to_numpy()
    theta_h = df["theta_h"].to_numpy()

    return theta_o, theta_h



def df_to_params(df_path:str) -> IECparams:
    """! @brief Extracts the PT parameters from a given dataframe
    @param df_path: string, path of the dataframe
    @return p: IECparams, parameter object

    This function reads a Pandas dataframe and extracts parameters of a
    Power Transformer Model, and stores them in the form of an
    <code>IECparams</code> object.
    In case the dataframe contains more fields those are ignored.
    """

    df = pd.read_csv(
        df_path,
        delimiter=";",
        #skiprows=[0,2],
    )

    p = IECparams(
        delta_theta_or=df["delta_theta_or"][0],
        delta_theta_hr=df["delta_theta_hr"][0],
        tau_o=df["tau_o"][0],
        tau_w=df["tau_w"][0],
        R=df["R"][0],
        x=df["x"][0],
        y=df["y"][0],
        k_11=df["k_11"][0],
        k_21=df["k_21"][0],
        k_22=df["k_22"][0]
    )

    return p



# ################# #
# Dataframe Outputs #
# ################# #

def save_simulation(
        t: np.ndarray[:],
        K: np.ndarray[:],
        theta_a: np.ndarray[:],
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        ll: np.ndarray[:],
        out_path: str
) -> None:
    """! @brief Saves the PT simulation into a given dataframe
    @param t: np.ndarray[T], time index of the sample
    @param K: np.ndarray[T], load factor's time series
    @param theta_a: np.ndarray[T], ambient temperature's time series
    @param theta_o: np.ndarray[T], top-oil temperature's time series
    @param theta_h: np.ndarray[T], hotspot temperature's time series
    @param ll: np.ndarray[T], loss of life's time series
    @param out_path: string, path of the output dataframe
    @return None

    This function saves the outcomes of a Power Transformer simulation into
    a Pandas dataframe. It requires the NumPy arrays of both the Model
    inputs and Model outputs.
    """

    df = pd.DataFrame()

    df["t"] = t
    df["K"] = K
    df["theta_a"] = theta_a
    df["theta_o"] = np.round(theta_o, decimals=1)
    df["theta_h"] = np.round(theta_h, decimals=1)
    df["ll"] = ll.astype(int)

    df.to_csv(path_or_buf=out_path, sep=";", index=False)

    return



def save_params(p:IECparams, out_path:str) -> None:
    """! @brief Saves the PT parameters into a given dataframe
    @param p: IECparams, parameter object
    @param out_path: string, path of the output dataframe
    @return None

    This function saves the parameters of a Power Transformer into
    a Pandas dataframe. It requires such parameters to be given in the
    form of a <code>IECparams</code> object.
    """

    df = pd.DataFrame()
    df["delta_theta_or"] = [ p.delta_theta_or ]
    df["delta_theta_hr"] = [ p.delta_theta_hr ]
    df["tau_o"] = [ p.tau_o ]
    df["tau_w"] = [ p.tau_w ]
    df["R"] = [ p.R ]
    df["x"] = [ p.x ]
    df["y"] = [ p.y ]
    df["k_11"] = [ p.k_11 ]
    df["k_21"] = [ p.k_21 ]
    df["k_22"] = [ p.k_22 ]

    df.to_csv(path_or_buf=out_path, sep=";", index=False)

    return



# ############ #
# Text Outputs #
# ############ #

def print_system_info(
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        L: np.ndarray[3,1],
        Q: np.ndarray[3,3],
        R: np.ndarray[1,1],
        out_path: str
) -> None:
    """! @brief Saves the DCT matrices into a text file
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param L: np.ndarray[3,1], Luemberger gain
    @param Q: np.ndarray[3,3], prediction covariance matrix
    @param R: np.ndarray[1,1], measurement covariance matrix
    @param out_path: string, path of the output file
    @return None

    The parameters of a Power Transformer are converted into
    Control Theory's matrices for the simulation to happen.
    This function saves such matrices into a text file, for debugging
    purposes as well as system analysis.

    Some system properties are reported as well: the Observability matrix
    together with its rank, and the <code>A-LC</code> error dynamic matrix
    together with its poles.
    """

    with open(out_path, "w") as f:
        f.write("--- System Matrices:\n")
        f.write(f"\nA:\n{A}\n")
        f.write(f"\nB:\n{B}\n")
        f.write(f"\nC:\n{C}\n")
        f.write(f"\nM:\n{M}\n")

        f.write("\n--- Observer and Kalman Matrices:\n")
        f.write(f"\nL:\n{L}\n")
        f.write(f"\nQ:\n{Q}\n")
        f.write(f"\nR:\n{R}\n")

        O, rank_O = obs_matrix(A=A, C=C)
        ALC, poles = system_poles(A=A, C=C, L=L)

        f.write("\n--- System Observability:\n")
        f.write(f"\nObservability Matrix:\n{O}\n")
        f.write(f"\nObservability Rank: {rank_O}\n")
        f.write(f"\nA - LC:\n{ALC}\n")
        f.write(f"\nSystem Poles eig(A - LC):\n{poles}\n")

        f.close()
    return



def print_metrics(
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:],
        dataset_path: str,
        params_path: str,
        simulation_type: str,
        precision: int,
        out_path:str
) -> None:
    """! @brief Saves the output metrics of a simulation
    @param theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @param theta_h: np.ndarray[T], simulated hotspot temperature's time series
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series
    @param dataset_path: string, path of the dataset used in the simulation
    @param params_path: string, path of the parameter dataframe used in the simulation
    @param simulation_type: string, type of the simulation
    @param precision: integer, numer of decimal digits to round to
    @param out_path: string, path of the output file
    @return None

    This function saves into a text file the comparison metrics of a
    Power Transformer simulation where the model outputs, top-oil and
    hotspot, are compared with their reference values. It also reports
    some information about the simulation for reproducibility purposes.
    """

    with open(out_path, "w") as f:
        f.write("--- Simulation Data:\n")

        f.write("\n")
        f.write(f"dataset_path = {dataset_path}\n")
        f.write(f"params_path = {params_path}\n")
        f.write(f"simulation_type = {simulation_type}\n")

        f.write("\n--- Error Metrics:\n")

        theta_stack = np.vstack((theta_o, theta_h))
        theta_stack_ref = np.vstack((theta_o_ref, theta_h_ref))

        f.write("\n")
        f.write(f"MAE  theta_o = {np.round(mae(theta_o, theta_o_ref), decimals=precision)} °C\n")
        f.write(f"MAE  theta_h = {np.round(mae(theta_h, theta_h_ref), decimals=precision)} °C\n")
        f.write(f"MAE  total   = {np.round(mae(theta_stack, theta_stack_ref), decimals=precision)} °C\n")

        f.write("\n")
        f.write(f"RMSE theta_o = {np.round(rmse(theta_o, theta_o_ref), decimals=precision)} °C\n")
        f.write(f"RMSE theta_h = {np.round(rmse(theta_h, theta_h_ref), decimals=precision)} °C\n")
        f.write(f"RMSE total   = {np.round(rmse(theta_stack, theta_stack_ref), decimals=precision)} °C\n")

        f.write("\n")
        f.write(f"MSE  total   = {np.round(mse(theta_stack, theta_stack_ref), decimals=precision)} °C^2\n")

        f.close()
    return



def print_ll(
        ll: np.ndarray[:],
        ll_ref: np.ndarray[:],
        dataset_path: str,
        params_path: str,
        simulation_type: str,
        life_type: str,
        precision: int,
        out_path:str
) -> None:
    """! @brief Saves the loss of life's metrics of a simulation
    @param ll: np.ndarray[T], simulated loss of life's time series
    @param ll_ref: np.ndarray[T], reference loss of life's time series
    @param dataset_path: string, path of the dataset used in the simulation
    @param params_path: string, path of the parameter dataframe used in the simulation
    @param simulation_type: string, type of the simulation
    @param life_type: string, type of loss of life used in the simulation
    @param precision: integer, numer of decimal digits to round to
    @param out_path: string, path of the output file
    @return None

    This function saves into a text file the comparison metrics of a
    Power Transformer simulation where the loss of life is compared with
    its reference values. It also reports some information about the
    simulation for reproducibility purposes.
    """

    with open(out_path, "w") as f:
        f.write("--- Simulation Data:\n")

        f.write("\n")
        f.write(f"dataset_path = {dataset_path}\n")
        f.write(f"params_path = {params_path}\n")
        f.write(f"simulation_type = {simulation_type}\n")
        f.write(f"life_type = {life_type}\n")

        f.write("\n--- Loss of Life Metrics:\n")
        f.write("\n")
        f.write(f"MAE  = {np.round(mae(ll, ll_ref), decimals=precision)} min\n")
        f.write(f"RMSE = {np.round(rmse(ll, ll_ref), decimals=precision)} min\n")

        f.write("\n--- Final Loss of Life:\n")
        f.write("\n")
        f.write(f"Reference = {np.round(ll_ref[-1], decimals=precision)} min\n")
        f.write(f"Model     = {np.round(ll[-1], decimals=precision)} min\n")

        f.write("\n")
        f.write(f"Error     = {np.round( ll[-1] - ll_ref[-1], decimals=precision)} min\n")

        f.close()
    return



# ################ #
# Training Outputs #
# ################ #

def print_primes(
        A_prime: np.ndarray[3,3],
        B_prime: np.ndarray[3,3],
        out_path: str
) -> None:
    """! @brief Saves the improvement matrices into a text file
    @param A_prime: np.ndarray[3,3], system matrix improvement
    @param B_prime: np.ndarray[3,3], input matrix improvement
    @param out_path: string, path of the output file
    @return None

    This function is used after the training to save the
    final value of the improvement matrices obtained by the network.
    """

    with open(out_path, "w") as f:
        f.write("--- Improvement Matrices:\n")
        f.write(f"\nA_prime:\n{A_prime}\n")
        f.write(f"\nB_prime:\n{B_prime}\n")

        f.close()
    return



def print_train_state(
        itr: int,
        test_loss: float,
        minibatch_loss: float,
        time_ms: int,
        A_prime: np.ndarray[3,3],
        B_prime: np.ndarray[3,3],
        out_path: str
) -> None:
    """! @brief Prints the state of the current training iteration
    @param itr: integer, current iteration
    @param test_loss: float, current loss on the test dataset
    @param minibatch_loss: float, average loss on the training dataset
    @param time_ms: integer, average time to complete an epoch
    @param A_prime: np.ndarray[3,3], current system matrix improvement
    @param B_prime: np.ndarray[3,3], current input matrix improvement
    @return None

    This function is used during the training to save the
    network state of each epoch into a text file.
    Like this its evolution can be monitored for debugging purposes.
    """

    with open(out_path, "w") as f:
        f.write("--- Training State:\n")

        f.write(f"\nEpoch: {itr}\n")
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Avg Minibatch Loss: {minibatch_loss}\n")
        f.write(f"Avg Epoch Time: {time_ms} ms\n")

        f.write(f"\nA_prime:\n{A_prime}\n")
        f.write(f"\nB_prime:\n{B_prime}\n")

        f.close()
    return



def train_log(out_path: str) -> None:
    """! @brief Prints the parameters of the training
    @param out_path: string, path of the output file
    @return None

    This function saves into a text file the parameters of the
    current training, so that they can be retrieved for reproducibility purposes.
    The parameters are automatically extracted from the <code>params_train</code>
    file, no need to pass them to the function.
    """

    with open(out_path, "w") as f:
        f.write("--- Training Logs:\n")

        f.write(f"\ntorch device: {par.device}\n")
        f.write(f"torch dtype: {par.model_type}\n")

        f.write(f"\nreg_A: {par.reg_A}\n")
        f.write(f"reg_B: {par.reg_B}\n")
        f.write(f"learning_rate: {par.learning_rate}\n")

        f.write(f"\nRNNFunc: {par.RNNFunc.get_name()}\n")

        f.write(f"\nepochs: {par.epochs}\n")
        f.write(f"n_batches: {par.n_batches}\n")
        f.write(f"batch_duration: {par.batch_duration}\n")

        f.write(f"\ntest_freq: {par.test_freq}\n")
        f.write(f"visualize_train: {par.visualize_train}\n")

        f.write(f"\ndataset_path: {par.dataset_path}\n")
        f.write(f"params_path: {par.params_path}\n")
        f.write(f"out_directory: {par.out_directory}\n")

        f.close()
    return
