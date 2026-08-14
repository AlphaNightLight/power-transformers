import numpy as np



# ################ #
# Simulation Types #
# ################ #

simulation_type = "simple"
assert simulation_type in ["simple", "observer", "kalman"], f"Invalid Simulation Type, {simulation_type}"

life_type = "tu"
assert life_type in ["tu", "ntu"], f"Invalid Life Type, {life_type}"

# ################### #
# Matrix Improvements #
# ################### #

A_prime = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
# A_prime = np.array([
#     [0, 0, -6.4e-3], # [0, -0.00013, -0.00013],
#     [0, -3e-3, -1.4e-3],
#     [0, -1.7e-3, -6.3e-3]
# ])
# L1 interesting 1

B_prime = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
# B_prime = np.array([
#     [0, -4e-3, 0],
#     [0, 0, 0],
#     [0, 0, 0]
# ])

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/tts-a-summer-test.csv"
params_path = "dataset/tts-a-params.csv"
out_directory = "out-hitachi"

# ############# #
# Uncertainties #
# ############# #

theta_o_std = 1.
theta_h_std = 5.
measurement_std = 2.
