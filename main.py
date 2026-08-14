# ####### #
# Imports #
# ####### #

from main_params import simulation_type, life_type, A_prime, B_prime
from main_params import dataset_path, params_path, out_directory
from main_params import theta_o_std, theta_h_std, measurement_std

from modules.inout import df_to_inputs, df_to_outputs, df_to_params
from modules.inout import save_simulation, print_system_info, print_metrics, print_ll

from modules.iec import convert_inputs, params_to_matrices, initialize_from_data, loss_of_life
from modules.control import std_to_matrices, optimal_L
from modules.simulation import run_simulation, run_simulation_observer, run_simulation_kalman
from modules.plots import sim_plot, com_plot, error_plot, ll_plot, pmf_plot

from modules.inout import mkdir
mkdir(out_directory)



# ######### #
# Load Data #
# ######### #

t, delta_t, K, theta_a = df_to_inputs(dataset_path)
theta_o_ref, theta_h_ref = df_to_outputs(dataset_path)
p = df_to_params(params_path)



# ########## #
# Parse Data #
# ########## #

u = convert_inputs(K=K, theta_a=theta_a, p=p)

theta_o_0 = theta_o_ref[0]
theta_h_0 = theta_h_ref[0]
Theta_0 = initialize_from_data(theta_o_0=theta_o_0, theta_h_0=theta_h_0, p=p)

A, B, C, M = params_to_matrices(p)
A += A_prime
B += B_prime

Q, R = std_to_matrices(
    theta_o_std=theta_o_std,
    theta_h_std=theta_h_std,
    measurement_std=measurement_std,
    p=p
)

L = optimal_L(C=C, Q=Q, R=R)



# ######## #
# Simulate #
# ######## #

if simulation_type == "observer":
    _, theta_o, theta_h = run_simulation_observer(
        Theta_0=Theta_0,
        A=A, B=B, C=C, M=M,
        u=u, delta_t=delta_t,
        L=L, theta_o_ref=theta_o_ref
    )
elif simulation_type == "kalman":
    _, theta_o, theta_h = run_simulation_kalman(
        Theta_0=Theta_0,
        A=A, B=B, C=C, M=M,
        u=u, delta_t=delta_t,
        Q=Q, R=R, theta_o_ref=theta_o_ref
    )
else: # simulation_type == "simple"
    _, theta_o, theta_h = run_simulation(
        Theta_0=Theta_0,
        A=A, B=B, C=C, M=M,
        u=u, delta_t=delta_t
    )

ll = loss_of_life(
    ll_0=0., theta_h=theta_h, delta_t=delta_t, life_type=life_type
)
ll_ref = loss_of_life(
    ll_0=0., theta_h=theta_h_ref, delta_t=delta_t, life_type=life_type
)



# #################### #
# Save Output and Info #
# #################### #

save_simulation(
    t=t, K=K, theta_a=theta_a,
    theta_o=theta_o_ref, theta_h=theta_h_ref, ll=ll,
    out_path=out_directory+"/out.csv"
)

print_system_info(
    A=A, B=B, C=C, M=M,
    L=L, Q=Q, R=R,
    out_path=out_directory+"/info.txt"
)

print_metrics(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type=simulation_type, precision=2,
    out_path=out_directory+"/metrics.txt"
)

print_ll(
    ll=ll, ll_ref=ll_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type=simulation_type, life_type=life_type, precision=2,
    out_path=out_directory+"/lost-life.txt"
)



# ############ #
# Plot Results #
# ############ #

sim_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    pathname=out_directory+"/simulation.png"
)

com_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/comparison.png"
)

error_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/error.png"
)

ll_plot(
    t=t, ll=ll, ll_ref=ll_ref, pathname=out_directory+"/lost-life.png"
)

pmf_plot(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/pmf.png"
)
