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
    [ 0.00128948, -0.00105583, -0.00059940],
    [ 0.00035246,  0.00073835, -0.00064652],
    [-0.00041934,  0.00133251, -0.00161678]
])

B_prime = np.array([
    [-1.26603390e-03, -3.86870236e-05, -5.59664073e-05],
    [ 2.67492403e-05, -1.05974013e-05, -7.56672390e-06],
    [-8.32092926e-04, -1.37798448e-05, -2.17514900e-06]
])

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-test.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/simulation-rnn-L2-simple"

# ############# #
# Uncertainties #
# ############# #

theta_o_std = 1.
theta_h_std = 5.
measurement_std = 2.
