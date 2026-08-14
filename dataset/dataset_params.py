# ####### #
# Imports #
# ####### #

from modules.inout import save_params
from modules.iec import IECparams

from modules.inout import mkdir

dataset_dir = "parsed"
mkdir(dataset_dir)





# ######## #
# Defaults #
# ######## #

# The international standard proposes default values depending on
# the cooling mode for all parameters, except [delta_theta_or, delta_theta_hr, R]
# which depend on the individual transformer.

p_default = {
    "ONAN_d": # Oil Natural Air Natural, for Distribution
        {"tau_o":180., "tau_w": 4., "x":0.8, "y":1.6, "k_11":1. , "k_21":1.,   "k_22":2.},
    "ONAN_r": # Oil Natural Air Natural, Restricted
        {"tau_o":210., "tau_w":10., "x":0.8, "y":1.3, "k_11":0.5, "k_21":3.,   "k_22":2.},
    "ONAN":   # Oil Natural Air Natural
        {"tau_o":210., "tau_w":10., "x":0.8, "y":1.3, "k_11":0.5, "k_21":2.,   "k_22":2.},
    "ONAF_r": # Oil Natural Air Forced, Restricted
        {"tau_o":150., "tau_w": 7., "x":0.8, "y":1.3, "k_11":0.5, "k_21":3.,   "k_22":2.},
    "ONAF":   # Oil Natural Air Forced
        {"tau_o":150., "tau_w": 7., "x":0.8, "y":1.3, "k_11":0.5, "k_21":2.,   "k_22":2.},
    "OF_r":   # Oil Forced, Restricted
        {"tau_o": 90., "tau_w": 7., "x":1. , "y":1.3, "k_11":1. , "k_21":1.45, "k_22":1.},
    "OF":     # Oil Forced
        {"tau_o": 90., "tau_w": 7., "x":1. , "y":1.3, "k_11":1. , "k_21":1.3,  "k_22":1.},
    "OD":     # Oil Directed
        {"tau_o": 90., "tau_w": 7., "x":1. , "y":1.2, "k_11":1. , "k_21":1.,   "k_22":1.},
}





# #### #
# TTSa #
# #### #

cm = "ONAF"
p = IECparams(
    delta_theta_or = 51.2,
    delta_theta_hr = 76.6 - 51.2,
    tau_o = p_default[cm]["tau_o"],
    tau_w = p_default[cm]["tau_w"],
    R     = 8.44,
    x     = p_default[cm]["x"],
    y     = p_default[cm]["y"],
    k_11  = p_default[cm]["k_11"],
    k_21  = p_default[cm]["k_21"],
    k_22  = p_default[cm]["k_22"],
)

save_params(p, out_path=dataset_dir+"/tts-a-params.csv")
