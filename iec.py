import numpy as np
import pandas as pd



#########################
# Parameters Definition #
#########################

class IECparams:
    def __init__(
            self,
            delta_theta_or: float,  # Top oil temperature rise in steady state at rated losses
            delta_theta_hr: float,  # Hotspot to top oil gradient at rated current
            tau_o: float,           # Average oil time constant
            tau_w: float,           # Winding time constant
            R: float,               # Loss Ratio
            x: float,               # Exponential of total losses
            y: float,               # Winding exponent
            k_11: float,            # Fitting constant 11
            k_21: float,            # Fitting constant 21
            k_22: float             # Fitting constant 22
    ):
        self.delta_theta_or = delta_theta_or
        self.delta_theta_hr = delta_theta_hr
        self.tau_o = tau_o
        self.tau_w = tau_w
        self.R = R
        self.x = x
        self.y = y
        self.k_11 = k_11
        self.k_21 = k_21
        self.k_22 = k_22

        self.k_21_bar = 1 - k_21
        self.G = 1/(k_22*tau_w) + k_22/tau_o
        self.G_21 = k_21/(k_22*tau_w) + (1-k_21)*(k_22/tau_o)

    @classmethod
    def from_df(cls, params:pd.DataFrame):
        return cls(
            delta_theta_or = params["delta_theta_or"][0],
            delta_theta_hr = params["delta_theta_hr"][0],
            tau_o = params["tau_o"][0],
            tau_w = params["tau_w"][0],
            R = params["R"][0],
            x = params["x"][0],
            y = params["y"][0],
            k_11 = params["k_11"][0],
            k_21 = params["k_21"][0],
            k_22 = params["k_22"][0]
        )

    def __str__(self):
        s = "Transformer Parameters:\n   "
        s += f"delta_theta_or = {self.delta_theta_or},"
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



################
# Loss of Life #
################

# Relative aging rate for thermally upgraded papers
def V_tu(theta_h:float)->float:
    return np.exp( 15_000/(110+273) - 15_000/(theta_h+273) )

# Relative aging rate for NON thermally upgraded papers
def V_ntu(theta_h:float)->float:
    return 2**( (theta_h-98) / 6 )

def loss_of_life(L_prev:float,theta_h:float,delta_t:float,V_type:str="tu")->float:
    if V_type == "ntu":
        L_dot = V_ntu(theta_h)
    else:
        L_dot = V_tu(theta_h)
    L = L_prev + delta_t * L_dot
    return L



#####################
# Exponential Model #
#####################

def iec_exp_o(t:float, theta_a:float, K:float, delta_theta_oi:float, p:IECparams):
    Kx = ( (1 + p.R * K**2) / (1 + p.R) ) ** p.x
    f_o = np.exp( -t / (p.k_11 * p.tau_o) )
    return theta_a + p.delta_theta_or*Kx + f_o*( delta_theta_oi - p.delta_theta_or*Kx )

def iec_exp_h(t:float, theta_o:float, K:float, delta_theta_hi:float, p:IECparams):
    Ky = K ** p.y
    f_h = p.k_21 * np.exp( -t / (p.k_22 * p.tau_w) )
    f_h += p.k_21_bar * np.exp( -t * (p.k_22/p.tau_o) )
    return theta_o + p.delta_theta_hr*Ky + f_h*( delta_theta_hi - p.delta_theta_hr*Ky )



##################################
# Differential Model First Order #
##################################

def iec_diff_init(
        theta_a_init: float,
        K_init: float,
        p: IECparams,
        delta_theta_oi: float=None,
        delta_theta_hi: float=None
) -> (float, float, float):
    Kx_init = ( (1 + p.R * K_init**2) / (1 + p.R) ) ** p.x
    Ky_init = K_init ** p.y

    # ASSUMPTION: If we don't know delta_theta_oi or delta_theta_hi we
    # use their steady state values.
    if delta_theta_oi is None:
        delta_theta_oi = p.delta_theta_or * Kx_init
    if delta_theta_hi is None:
        delta_theta_hi = p.delta_theta_hr * Ky_init

    theta_o_init = theta_a_init + delta_theta_oi
    theta_h1_init = p.k_21 * delta_theta_hi
    theta_h2_init = p.k_21_bar * delta_theta_hi

    return theta_o_init, theta_h1_init, theta_h2_init

def iec_diff(
        theta_o_prev: float,
        theta_h1_prev: float,
        theta_h2_prev: float,
        delta_t: float,
        theta_a: float,
        K: float,
        p: IECparams
) -> (float, float, float):
    Theta_prev = np.array([
        [theta_o_prev],
        [theta_h1_prev],
        [theta_h2_prev]
    ])

    u = np.array([
        [theta_a],
        [( (1 + p.R * K**2) / (1 + p.R) ) ** p.x],
        [K ** p.y]
    ])

    A = np.array([
        [-1/(p.k_11 * p.tau_o), 0, 0],
        [0, -1/(p.k_22 * p.tau_w), 0],
        [0, 0, -p.k_22/p.tau_o]
    ])

    B = np.array([
        [1/(p.k_11 * p.tau_o), p.delta_theta_or/(p.k_11 * p.tau_o), 0],
        [0, 0, p.k_21 * (p.delta_theta_hr/(p.k_22 * p.tau_w))],
        [0, 0, p.k_21_bar * p.delta_theta_hr * (p.k_22/p.tau_o)]
    ])

    Theta_dot = A @ Theta_prev + B @ u
    Theta = Theta_prev + delta_t * Theta_dot

    return Theta[0,0], Theta[1,0], Theta[2,0]



###################################
# Differential Model Second Order #
###################################

def iec_diff_2_init(
        theta_a_init: float,
        K_init: float,
        p: IECparams,
        delta_theta_oi: float=None,
        delta_theta_hi: float=None
) -> (float, float, float):
    Kx_init = ( (1 + p.R * K_init**2) / (1 + p.R) ) ** p.x
    Ky_init = K_init ** p.y

    # ASSUMPTION: If we don't know delta_theta_oi or delta_theta_hi we
    # use their steady state values.
    if delta_theta_oi is None:
        delta_theta_oi = p.delta_theta_or * Kx_init
    if delta_theta_hi is None:
        delta_theta_hi = p.delta_theta_hr * Ky_init

    theta_o_init = theta_a_init + delta_theta_oi
    theta_ho_init = delta_theta_hi
    theta_ho_dot_init = (p.delta_theta_hr * Ky_init - delta_theta_hi) * p.G_21

    return theta_o_init, theta_ho_init, theta_ho_dot_init

def iec_diff_2(
        theta_o_prev: float,
        theta_ho_prev: float,
        theta_ho_dot_prev: float,
        delta_t: float,
        theta_a: float,
        K: float,
        p: IECparams
) -> (float, float, float):
    Theta_prev = np.array([
        [theta_o_prev],
        [theta_ho_prev],
        [theta_ho_dot_prev]
    ])

    u = np.array([
        [theta_a],
        [( (1 + p.R * K**2) / (1 + p.R) ) ** p.x],
        [K ** p.y]
    ])

    A = np.array([
        [-1/(p.k_11 * p.tau_o), 0, 0],
        [0, 0, 1],
        [0, -1/(p.tau_o * p.tau_w), -p.G]
    ])

    B = np.array([
        [1/(p.k_11 * p.tau_o), p.delta_theta_or/(p.k_11 * p.tau_o), 0],
        [0, 0, 0],
        [0, 0, p.delta_theta_hr/(p.tau_o * p.tau_w)]
    ])

    Theta_dot = A @ Theta_prev + B @ u
    Theta = Theta_prev + delta_t * Theta_dot

    return Theta[0,0], Theta[1,0], Theta[2,0]



############################################
# Differential Model Second Order Reactive #
############################################

def iec_diff_2r_init(
        theta_a_init: float,
        K_init: float,
        p: IECparams,
        delta_theta_oi: float=None,
        delta_theta_hi: float=None
) -> (float, float, float):
    Kx_init = ( (1 + p.R * K_init**2) / (1 + p.R) ) ** p.x
    Ky_init = K_init ** p.y

    # ASSUMPTION: If we don't know delta_theta_oi or delta_theta_hi we
    # use their steady state values.
    if delta_theta_oi is None:
        delta_theta_oi = p.delta_theta_or * Kx_init
    if delta_theta_hi is None:
        delta_theta_hi = p.delta_theta_hr * Ky_init

    theta_o_init = theta_a_init + delta_theta_oi
    theta_ho_init = delta_theta_hi
    theta_ho_dot_init = - delta_theta_hi * p.G_21

    return theta_o_init, theta_ho_init, theta_ho_dot_init

def iec_diff_2r(
        theta_o_prev: float,
        theta_ho_prev: float,
        g_dot_prev: float,
        delta_t: float,
        theta_a: float,
        K: float,
        p: IECparams
) -> (float, float, float):
    Theta_prev = np.array([
        [theta_o_prev],
        [theta_ho_prev],
        [g_dot_prev]
    ])

    u = np.array([
        [theta_a],
        [( (1 + p.R * K**2) / (1 + p.R) ) ** p.x],
        [K ** p.y]
    ])

    A = np.array([
        [-1/(p.k_11 * p.tau_o), 0, 0],
        [0, 0, 1],
        [0, -1/(p.tau_o * p.tau_w), -p.G]
    ])

    B = np.array([
        [1/(p.k_11 * p.tau_o), p.delta_theta_or/(p.k_11 * p.tau_o), 0],
        [0, 0, p.delta_theta_hr * p.G_21],
        [0, 0, p.delta_theta_hr * ( 1/(p.tau_o * p.tau_w) - p.G * p.G_21 )]
    ])

    Theta_dot = A @ Theta_prev + B @ u
    Theta = Theta_prev + delta_t * Theta_dot

    return Theta[0,0], Theta[1,0], Theta[2,0]
