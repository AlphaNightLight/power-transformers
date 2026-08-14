# ####### #
# Imports #
# ####### #

from train_params import dataset_path, params_path, out_directory

from modules.inout import df_to_inputs, df_to_outputs, df_to_params
from modules.inout import print_system_info, print_metrics, print_ll, print_primes

from modules.iec import convert_inputs, params_to_matrices, initialize_from_data
from modules.iec import initialize_from_array, loss_of_life
from modules.control import std_to_matrices, optimal_L
from modules.simulation import run_simulation
from modules.plots import com_plot, error_plot, ll_plot, pmf_plot

from modules.train_main import improve_system

from modules.inout import mkdir
mkdir(out_directory)
mkdir(out_directory+"/before")
mkdir(out_directory+"/after")
mkdir(out_directory+"/train-state")



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

Q, R = std_to_matrices(
    theta_o_std=0.,
    theta_h_std=0.,
    measurement_std=1.,
    p=p
)

L = optimal_L(C=C, Q=Q, R=R)



# ################################# #
# Simulate and Save Before Training #
# ################################# #

_, theta_o, theta_h = run_simulation(
    Theta_0=Theta_0,
    A=A, B=B, C=C, M=M,
    u=u, delta_t=delta_t
)

ll = loss_of_life(
    ll_0=0., theta_h=theta_h, delta_t=delta_t, life_type="tu"
)
ll_ref = loss_of_life(
    ll_0=0., theta_h=theta_h_ref, delta_t=delta_t, life_type="tu"
)

print_system_info(
    A=A, B=B, C=C, M=M,
    L=L, Q=Q, R=R,
    out_path=out_directory+"/before/info.txt"
)

print_metrics(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type="simple", precision=2,
    out_path=out_directory+"/before/metrics.txt"
)

print_ll(
    ll=ll, ll_ref=ll_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type="simple", life_type="tu", precision=2,
    out_path=out_directory+"/before/lost-life.txt"
)

com_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/before/comparison.png"
)

error_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/before/error.png"
)

ll_plot(
    t=t, ll=ll, ll_ref=ll_ref, pathname=out_directory+"/before/lost-life.png"
)

pmf_plot(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/before/pmf.png"
)



# ############ #
# Optimization #
# ############ #

Theta_0_ref = initialize_from_array(
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref, p=p
)

A_prime, B_prime = improve_system(
    x_0_ref=Theta_0_ref,
    A=A, B=B, C=C, M=M,
    u=u, delta_t=delta_t,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref
)

print_primes(
    A_prime=A_prime,
    B_prime=B_prime,
    out_path=out_directory+"/primes.txt"
)

A += A_prime
B += B_prime



# ################################ #
# Simulate and Save After Training #
# ################################ #

_, theta_o, theta_h = run_simulation(
    Theta_0=Theta_0,
    A=A, B=B, C=C, M=M,
    u=u, delta_t=delta_t
)

ll = loss_of_life(
    ll_0=0., theta_h=theta_h, delta_t=delta_t, life_type="tu"
)
ll_ref = loss_of_life(
    ll_0=0., theta_h=theta_h_ref, delta_t=delta_t, life_type="tu"
)

print_system_info(
    A=A, B=B, C=C, M=M,
    L=L, Q=Q, R=R,
    out_path=out_directory+"/after/info.txt"
)

print_metrics(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type="simple", precision=2,
    out_path=out_directory+"/after/metrics.txt"
)

print_ll(
    ll=ll, ll_ref=ll_ref,
    dataset_path=dataset_path, params_path=params_path,
    simulation_type="simple", life_type="tu", precision=2,
    out_path=out_directory+"/after/lost-life.txt"
)

com_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/after/comparison.png"
)

error_plot(
    t=t, theta_a=theta_a, K=K,
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/after/error.png"
)

ll_plot(
    t=t, ll=ll, ll_ref=ll_ref, pathname=out_directory+"/after/lost-life.png"
)

pmf_plot(
    theta_o=theta_o, theta_h=theta_h,
    theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref,
    pathname=out_directory+"/after/pmf.png"
)
