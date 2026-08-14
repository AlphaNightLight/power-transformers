"""!
@file simulation.py
@brief Functions to simulate a DCT system

This file contains the functions to simulate a Deterministic
Control Theory system in State Space Representation with Euler
Discretization and the following options:

1) No Improvement

2) Luemberger Observer

3) Kalman Filter
"""

import numpy as np



# ########### #
# Simulations #
# ########### #

def run_simulation(
        Theta_0: np.ndarray[3],
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        u: np.ndarray[3,:],
        delta_t: np.ndarray[:]
) -> (np.ndarray[3,:], np.ndarray[:], np.ndarray[:]):
    """! @brief Simulates a discretized SSR system with no additional improvements
    @param Theta_0: np.ndarray[3], initial state vector
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param u: np.ndarray[3,T], input vector's time series
    @param delta_t: np.ndarray[T], time between current and previous sample

    @return Theta: np.ndarray[3,T], simulated state vector's time series
    @return theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @return theta_h: np.ndarray[T], simulated hotspot temperature's time series

    This function performs the simulation of a Dynamical System in State Space
    Representation, through the usage of Euler Discretization. It returns a
    time series for the state vector as well as the time series of the
    outputs, both the measurable and the unmeasurable ones.
    """

    assert u.shape[1] == delta_t.shape[0], f"u.shape[1]={u.shape[1]} but delta_t.shape[0]={delta_t.shape[0]}"

    Theta = np.ndarray(u.shape)
    Theta[:,0] = Theta_0

    for i in range(1, u.shape[1]):
        Theta_dot = A @ Theta[:,i-1] + B @ u[:,i]
        Theta[:,i] = Theta[:,i-1] + delta_t[i] * Theta_dot

    theta_o = np.squeeze(C @ Theta)
    theta_h = np.squeeze(M @ Theta)

    return Theta, theta_o, theta_h



def run_simulation_observer(
        Theta_0: np.ndarray[3],
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        u: np.ndarray[3,:],
        delta_t: np.ndarray[:],
        L: np.ndarray[3,1],
        theta_o_ref: np.ndarray[:]
) -> (np.ndarray[3,:], np.ndarray[:], np.ndarray[:]):
    """! @brief Simulates a discretized SSR system with a Luemberger Observer
    @param Theta_0: np.ndarray[3], initial state vector
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param u: np.ndarray[3,T], input vector's time series
    @param delta_t: np.ndarray[T], time between current and previous sample
    @param L: np.ndarray[3,1], Luemberger gain
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series

    @return Theta: np.ndarray[3,T], simulated state vector's time series
    @return theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @return theta_h: np.ndarray[T], simulated hotspot temperature's time series

    This function performs the simulation of a Dynamical System in State Space
    Representation, through the usage of Euler Discretization. The simulation,
    is improved with a Luemberger Observer, which requires a reference value for
    the measurable output. It returns a time series for the state vector as
    well as the time series of the two outputs, measurable and unmeasurable.
    """

    assert u.shape[1] == delta_t.shape[0], f"u.shape[1]={u.shape[1]} but delta_t.shape[0]={delta_t.shape[0]}"
    assert u.shape[1] == theta_o_ref.shape[0], f"u.shape[1]={u.shape[1]} but theta_o_ref.shape[0]={theta_o_ref.shape[0]}"

    Theta = np.ndarray(u.shape)
    Theta[:,0] = Theta_0
    theta_o = np.ndarray(theta_o_ref.shape)
    theta_o[0] = C @ Theta_0
    Ls = L.squeeze() # (3,1) to (3,)

    for i in range(1, u.shape[1]):
        Theta_dot = A @ Theta[:,i-1] + B @ u[:,i]
        Theta_dot += Ls * (theta_o_ref[i-1] - theta_o[i-1])

        Theta[:,i] = Theta[:,i-1] + delta_t[i] * Theta_dot
        theta_o[i] = C @ Theta[:,i]

    theta_h = np.squeeze(M @ Theta)

    return Theta, theta_o, theta_h



def run_simulation_kalman(
        Theta_0: np.ndarray[3],
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        u: np.ndarray[3,:],
        delta_t: np.ndarray[:],
        Q: np.ndarray[3,3],
        R: np.ndarray[1,1],
        theta_o_ref: np.ndarray[:]
) -> (np.ndarray[3,:], np.ndarray[:], np.ndarray[:]):
    """! @brief Simulates a discretized SSR system with a Kalman FIlter
    @param Theta_0: np.ndarray[3], initial state vector
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param u: np.ndarray[3,T], input vector's time series
    @param delta_t: np.ndarray[T], time between current and previous sample
    @param Q: np.ndarray[3,3], prediction covariance matrix
    @param R: np.ndarray[1,1], measurement covariance matrix
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series

    @return Theta: np.ndarray[3,T], simulated state vector's time series
    @return theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @return theta_h: np.ndarray[T], simulated hotspot temperature's time series

    This function performs the simulation of a Dynamical System in State Space
    Representation, through the usage of Euler Discretization. The simulation,
    is improved with a Kalman Filter, which requires a reference value for
    the measurable output as well as the covariance matrices.
    .It returns a time series for the state vector as well as the time series
    of the two outputs, measurable and unmeasurable.
    """

    assert u.shape[1] == delta_t.shape[0], f"u.shape[1]={u.shape[1]} but delta_t.shape[0]={delta_t.shape[0]}"
    assert u.shape[1] == theta_o_ref.shape[0], f"u.shape[1]={u.shape[1]} but theta_o_ref.shape[0]={theta_o_ref.shape[0]}"

    Theta = np.ndarray(u.shape) # (3,T)
    Theta[:,0] = Theta_0
    theta_o = np.ndarray(theta_o_ref.shape) # (T,)
    theta_o[0] = C @ Theta_0
    S = np.zeros((3,3))

    for i in range(1, u.shape[1]):
        # Predict
        S_bar = Q + A @ S @ A.T # (3,3)
        Theta_dot = A @ Theta[:,i-1] + B @ u[:,i] # (3,)
        Theta_bar = Theta[:,i-1] + delta_t[i] * Theta_dot # (3,)
        theta_o_bar = C @ Theta_bar # (1,)

        # Update
        K = S_bar @ C.T @ np.linalg.inv( C @ S_bar @ C.T + R ) # (3,1)
        S = S_bar - K @ C @ S_bar # (3,3)
        Theta[:,i] = Theta_bar + delta_t[i] * K @ (theta_o_ref[i-1] - theta_o_bar) #!!!!! theta_o_ref[i]?
        theta_o[i] = C @ Theta[:,i]

    theta_h = np.squeeze(M @ Theta)

    return Theta, theta_o, theta_h
