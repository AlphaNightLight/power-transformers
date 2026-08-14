"""!
@file control.py
@brief Functions regarding DCT manipulations

This file contains the implementations of the Deterministic
Control Theory's tools, as well as the comparison metrics.
"""

import numpy as np
from modules.iec import IECparams



# ################### #
# Comparisons Metrics #
# ################### #

def mae(a: np.ndarray, b: np.ndarray) -> float:
    """! @brief Computes mean average error
    @param a: np.ndarray, first term of comparison
    @param b: np.ndarray, second term of comparison
    @return mae: float, mean average error between a and b
    """
    return np.average( np.abs( a - b ))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """! @brief Computes mean square error
    @param a: np.ndarray, first term of comparison
    @param b: np.ndarray, second term of comparison
    @return mae: float, mean square error between a and b
    """
    return np.average( ( a - b ) ** 2 )

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """! @brief Computes root-mean-square error
    @param a: np.ndarray, first term of comparison
    @param b: np.ndarray, second term of comparison
    @return mae: float, root-mean-square error between a and b
    """
    return np.sqrt( mse(a,b) )



# ############## #
# Observer Tools #
# ############## #

def std_to_matrices(
        theta_o_std: float,
        theta_h_std: float,
        measurement_std: float,
        p: IECparams
) -> (
        np.ndarray[3,3], np.ndarray[1,1]
):
    """! @brief Computes covariance matrices from standard deviations
    @param theta_o_std: float, standard deviation of top-oil temperature
    @param theta_h_std: float, standard deviation of hotspot temperature
    @param measurement_std: float, standard deviation of the measurements
    @param p: IECparams, parameter object
    @return Q: np.ndarray[3,3], prediction covariance matrix
    @return R: np.ndarray[1,1], measurement covariance matrix

    This functions converts the standard deviation of the outputs into
    the prediction covariance matrix <code>Q</code>, assuming
    <i>independence</i> between the variables and then producing a
    diagonal matrix.
    Similarly, the standard deviation of the measurement is used to generate
    the measurement covariance matrix <code>R</code> under the same assumption.
    """

    Q = np.asarray([
        [theta_o_std ** 2, 0., 0.],
        [0., p.k_21 * (theta_h_std ** 2), 0.],
        [0., 0., (1-p.k_21) * (theta_h_std ** 2)]
    ])
    R = np.asarray([
        [measurement_std ** 2]
    ])

    return Q, R



def optimal_L(
        C: np.ndarray[1,3],
        Q: np.ndarray[3,3],
        R: np.ndarray[1,1]
) -> np.ndarray[3,1]:
    """! @brief Computes Luemberger gain from covariance matrices
    @param C: np.ndarray[1,3], measurable output matrix
    @param Q: np.ndarray[3,3], prediction covariance matrix
    @param R: np.ndarray[1,1], measurement covariance matrix
    @return L: np.ndarray[3,1], Luemberger gain

    This function computes a Luemberger gain from the Kalman matrices
    and the measurable output matrix, using the approximation of the
    <i>time invariant Kalman Filter</i> presented in the Thesis.
    """

    L = Q @ C.T @ np.linalg.inv( C @ Q @ C.T + R )
    return L



# ############# #
# Observability #
# ############# #

def obs_matrix(
        A:np.ndarray[3,3], C:np.ndarray[1,3]
) -> ( np.ndarray[3,3], int ):
    """! @brief Computes the observability matrix and its rank
    @param A: np.ndarray[3,3], system matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @return O: np.ndarray[3,3], observability matrix
    @return rank_O: integer, rank of the observability matrix

    This function returns the observability matrix of a system,
    which for a <code>3x3</code> system is <code>[C|CA|CAA]</code>.
    It also computes and return the rank of this matrix, useful to
    assess which output are measurable and which are unmeasurable.
    """

    O = np.vstack((
        C, C @ A, C @ A @ A
    ))
    rank_O = np.linalg.matrix_rank(O)

    return O, rank_O



def system_poles(
        A: np.ndarray[3,3],
        C: np.ndarray[1,3],
        L: np.ndarray[3,1]
) -> ( np.ndarray[3,3], np.ndarray[3] ):
    """! @brief Computes the error dynamic matrix and its poles
    @param A: np.ndarray[3,3], system matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param L: np.ndarray[3,1], Luemberger gain
    @return ALC: np.ndarray[3,3], error dynamic matrix (A-LC)
    @return poles: np.ndarray[3], poles of the error dynamic matrix

    This function returns the error dynamic matrix of a Luemberger
    observer, which is equal to <code>A-LC</code>.
    It also computes and return the poles of this matrix, useful to
    assess the stability of the system and the observer.
    """

    ALC = A - L @ C
    poles = np.linalg.eig(ALC).eigenvalues

    return ALC, poles
