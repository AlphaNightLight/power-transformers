"""!
@file iec.py
@brief Functions to deal with the IEC model

This file contains the functions to link the IEC
differential formulation of a power Transformer with the Deterministic
Control Theory's equivalents used for the simulation. Its definitions
are grouped in the following objectives:

1) Convert IEC model to State Space Representation

2) Different ways to determine the initial state

3) Compute the Loss of Life
"""

import numpy as np



# ##################### #
# Parameters Definition #
# ##################### #

class IECparams:
    """! @brief The class to store IEC parameters

    This class is a compact wrapper to store all the parameters of a
    Power Transformer according to the IEC model. With it, it is possible
    to pass a single parameter object to a function rather than having to
    manage many variables.
    """

    def __init__(
            self,
            delta_theta_or: float,
            delta_theta_hr: float,
            tau_o: float,
            tau_w: float,
            R: float,
            x: float,
            y: float,
            k_11: float,
            k_21: float,
            k_22: float
    ):
        """! @brief Class constructor
        @param delta_theta_or: float, Top oil temperature rise in steady
        state at rated losses
        @param delta_theta_hr: float, Hotspot to top oil gradient at rated current
        @param tau_o: float, Average oil time constant
        @param tau_w: float, Winding time constant
        @param R: float, Loss Ratio
        @param x: float, Exponential of total losses
        @param y: float, Winding exponent
        @param k_11: float, Fitting constant 11
        @param k_21: float, Fitting constant 21
        @param k_22: float, Fitting constant 22
        """

        ## Top oil temperature rise in steady state at rated losses
        self.delta_theta_or = delta_theta_or
        ## Hotspot to top oil gradient at rated current
        self.delta_theta_hr = delta_theta_hr

        ## Average oil time constant
        self.tau_o = tau_o
        ## Winding time constant
        self.tau_w = tau_w

        ## Loss Ratio
        self.R = R
        ## Exponential of total losses
        self.x = x
        ## Winding exponent
        self.y = y

        ## Fitting constant 11
        self.k_11 = k_11
        ## Fitting constant 21
        self.k_21 = k_21
        ## Fitting constant 22
        self.k_22 = k_22

        ## Complement to 1 of k_21
        self.k_21_bar = 1 - k_21
        ## Used in a previous model version
        self.G = 1/(k_22*tau_w) + k_22/tau_o
        ## Used in a previous model version
        self.G_21 = k_21/(k_22*tau_w) + (1-k_21)*(k_22/tau_o)

    def __str__(self):
        """! @brief Converts the class to a string
        @return s: string, the text summary of the class
        """
        s = "Transformer Parameters:\n   "
        s += f"delta_theta_or = {self.delta_theta_or}, "
        s += f"delta_theta_hr = {self.delta_theta_hr}\n   "
        s += f"tau_o = {self.tau_o}, "
        s += f"tau_w = {self.tau_w}\n   "
        s += f"R = {self.R}, "
        s += f"x = {self.x}, "
        s += f"y = {self.y}\n   "
        s += f"k_11 = {self.k_11}, "
        s += f"k_21 = {self.k_21}, "
        s += f"k_22 = {self.k_22}\n   "
        s += f"k_21_bar = {self.k_21_bar}, "
        s += f"G = {self.G}, "
        s += f"G_21 = {self.G_21}"
        return s



# ########## #
# IEC to DCT #
# ########## #

def convert_inputs(
        K: np.ndarray[:],
        theta_a: np.ndarray[:],
        p: IECparams
) -> np.ndarray[3,:]:
    """! @brief Converts IEC inputs into SSR inputs
    @param K: np.ndarray[T], load factor's time series
    @param theta_a: np.ndarray[T], ambient temperature's time series
    @param p: IECparams, parameter object
    @return u: np.ndarray[3,T], input vector's time series

    This function converts the Power Transformer's inputs expressed
    in the IEC model into the input vector required by the State
    Space Formulation. It already applies the nonlinear transformations
    <code>K_x</code> and <code>K_y</code> to the load factor.
    """

    assert K.shape[0] == theta_a.shape[0], f"K.shape[0]={K.shape[0]} but theta_a.shape[0]={theta_a.shape[0]}"

    Kx = ( (1 + p.R * K**2) / (1+p.R) ) ** p.x
    Ky = K ** p.y
    u = np.vstack([theta_a, Kx, Ky])

    return u



def params_to_matrices(p: IECparams) -> (
        np.ndarray[3,3], np.ndarray[3,3], np.ndarray[1,3], np.ndarray[1,3]
):
    """! @brief Converts IEC parameters into DCT matrices
    @param p: IECparams, parameter object
    @return A: np.ndarray[3,3], system matrix
    @return B: np.ndarray[3,3], input matrix
    @return C: np.ndarray[1,3], measurable output matrix
    @return M: np.ndarray[1,3], unmeasurable output matrix

    This function converts the Power Transformer's parameters expressed
    as an <code>IECparams</code> object into the matrices used in
    the State Space Formulation: <code>A</code>, <code>B</code>
    <code>C</code> and <code>M</code>.
    """

    A = np.array([
        [-1 / (p.k_11 * p.tau_o), 0, 0],
        [0, -1 / (p.k_22 * p.tau_w), 0],
        [0, 0, -p.k_22 / p.tau_o]
    ])

    B = np.array([
        [1 / (p.k_11 * p.tau_o), p.delta_theta_or / (p.k_11 * p.tau_o), 0],
        [0, 0, p.k_21 * (p.delta_theta_hr / (p.k_22 * p.tau_w))],
        [0, 0, (1-p.k_21) * p.delta_theta_hr * (p.k_22 / p.tau_o)]
    ])

    C = np.asarray([
        [1., 0., 0.]
    ])

    M = np.asarray([
        [1., 1., 1.]
    ])

    return A, B, C, M



# ############### #
# Initializations #
# ############### #

def initialize_from_deltas(
        u_0: np.ndarray[3],
        delta_theta_oi: float,
        delta_theta_hi: float,
        p: IECparams
) -> np.ndarray[3]:
    """! @brief Computes initial state vector from the full initial conditions
    @param u_0: np.ndarray[3], initial input vector
    @param delta_theta_oi: float, top-oil temperature rise at start
    @param delta_theta_hi: float, hotspot to top-oil gradient at start
    @param p: IECparams, parameter object
    @return Theta_0: np.ndarray[3], initial state vector

    This function calculates the initial state vector of a Power Transformer system
    when the full initial conditions, i.e. the top-oil temperature rise at start
    and the hotspot to top-oil gradient at start, are available together with the
    initial input and the parameters.
    """

    Theta_0 = np.asarray([
        u_0[0] + delta_theta_oi,
        p.k_21 * delta_theta_hi,
        (1 - p.k_21) * delta_theta_hi
    ])
    return Theta_0



def initialize_from_steady(
        u_0: np.ndarray[3],
        p: IECparams
) -> np.ndarray[3]:
    """! @brief Computes initial state vector using the steady state assumption
    @param u_0: np.ndarray[3], initial input vector
    @param p: IECparams, parameter object
    @return Theta_0: np.ndarray[3], initial state vector

    This function calculates the initial state vector of a Power Transformer system
    when the full initial conditions are NOT available. In this case, the <i>Steady
    State Assumption</i> is used to infer them from the initial input and the
    IEC parameters.
    """

    Theta_0 = initialize_from_deltas(
        u_0 = u_0,
        delta_theta_oi = p.delta_theta_or * u_0[1],
        delta_theta_hi = p.delta_theta_hr * u_0[2],
        p = p
    )
    return Theta_0



def initialize_from_data(
        theta_o_0: float,
        theta_h_0: float,
        p: IECparams
) -> np.ndarray[3]:
    """! @brief Computes initial state vector from the initial outputs
    @param theta_o_0: float, initial top-oil temperature
    @param theta_h_0: float, initial hotspot temperature
    @param p: IECparams, parameter object
    @return Theta_0: np.ndarray[3], initial state vector

    This function calculates the initial state vector of a Power Transformer system
    when we have access to the initial outputs from both the top-oil and the
    hotspot.
    """

    Theta_0 = np.asarray([
        theta_o_0,
        p.k_21 * (theta_h_0 - theta_o_0),
        (1 - p.k_21) * (theta_h_0 - theta_o_0)
    ])
    return Theta_0



def initialize_from_array(
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:],
        p: IECparams
) -> np.ndarray[3,:]:
    """! @brief Computes a time series of initial states from the reference outputs
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series
    @param p: IECparams, parameter object
    @return Theta_0_ref: np.ndarray[3,T], initial state vector's time series

    This function is used for training purposes: given the time series of both the
    reference top-oil temperature and the reference hotspot temperature, it computes
    a time series of the initial state vector they would generate if selected as
    initial time.
    This is particularly useful in <i>Batch Training</i>, as it consent to
    immediately extract the initial condition of a batch simply knowing its
    time index.
    """

    Theta_0_ref = np.vstack((
        theta_o_ref,
        p.k_21 * (theta_h_ref - theta_o_ref),
        (1 - p.k_21) * (theta_h_ref - theta_o_ref)
    ))
    return Theta_0_ref



# ############ #
# Loss of Life #
# ############ #

def V_tu( theta_h:np.ndarray[:] ) -> np.ndarray[:]:
    """! @brief Computes relative aging rate for thermally upgraded papers
    @param theta_h: np.ndarray[T], hotspot temperature's time series
    @return V_tu: np.ndarray[T], relative aging rate's time series
    """
    return np.exp( 15_000/(110+273) - 15_000/(theta_h+273) )



def V_ntu( theta_h:np.ndarray[:] ) -> np.ndarray[:]:
    """! @brief Computes relative aging rate for NON thermally upgraded papers
    @param theta_h: np.ndarray[T], hotspot temperature's time series
    @return V_ntu: np.ndarray[T], relative aging rate's time series
    """
    return 2**( (theta_h-98) / 6 )



def loss_of_life(
        ll_0: float,
        theta_h: np.ndarray[:],
        delta_t: np.ndarray[:],
        life_type: str="tu"
) -> np.ndarray[:]:
    """! @brief Converts hotspot temperature to loss of life
    @param ll_0: float, initial loss of life's value
    @param theta_h: np.ndarray[T], hotspot temperature's time series
    @param delta_t: np.ndarray[T], time between current and previous sample
    @param life_type: string, type of loss of life to be used
    @return ll: np.ndarray[T], loss of life's time series

    This function converts the hotspot temperature of a Power Transformer
    into the correspondent loss of life. The <code>life_type</code> variable
    allows to select the desired relative aging rate between <code>"tu"</code>
    for thermally upgraded paper, and <code>"ntu"</code> for NON thermally
    upgraded paper.
    """

    assert theta_h.shape[0] == delta_t.shape[0], f"theta_h.shape[0]={theta_h.shape[0]} but delta_t.shape[0]={delta_t.shape[0]}"

    if life_type == "ntu":
        V = V_ntu(theta_h)
    else:
        V = V_tu(theta_h)

    ll = np.cumsum(V * delta_t) + ll_0

    return ll
