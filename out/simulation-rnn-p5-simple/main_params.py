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
    [0.00269415, -0.00127828,  0],
    [0,           0,           0],
    [0,          -0.00372104, -0.00988817]
])

B_prime = np.array([
    [-0.00365991, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-test.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/simulation-rnn-p5-simple"

# ############# #
# Uncertainties #
# ############# #

theta_o_std = 1.
theta_h_std = 5.
measurement_std = 2.
