import numpy as np



# ################ #
# Simulation Types #
# ################ #

simulation_type = "kalman"
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

B_prime = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-test.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/simulation-iec-kalman"

# ############# #
# Uncertainties #
# ############# #

theta_o_std = 1.
theta_h_std = 5.
measurement_std = 2.
