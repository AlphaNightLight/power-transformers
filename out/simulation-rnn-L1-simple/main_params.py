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
    [1.71199040e-03, -1.66344274e-03, -1.01515609e-04],
    [-5.04147596e-05,  7.50059986e-04, -1.41800127e-03],
    [-6.88376332e-04,  8.72199851e-04, -6.49028342e-03]
])

B_prime = np.array([
    [-1.45259370e-03, -3.95326511e-05, -4.81923278e-05],
    [-7.05152926e-05, -2.06334643e-05,  6.17221089e-05],
    [-1.96005392e-03,  2.54652794e-05, -1.37804226e-05]
])

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-test.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/simulation-rnn-L1-simple"

# ############# #
# Uncertainties #
# ############# #

theta_o_std = 1.
theta_h_std = 5.
measurement_std = 2.
