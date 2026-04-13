import pandas as pd

from iec import IECparams
from simulation import simulate_iec_exp, simulate_iec_diff
from simulation import simulate_iec_diff_2, simulate_iec_diff_2r
from simulation import df_rmse, transformer_plot, transformer_comparison_plot



#############
# Load Data #
#############

params_df = pd.read_csv("dataset/indian-params.csv",delimiter=";",skiprows=[0,2])
p = IECparams.from_df(params_df)

input_df = pd.read_csv("dataset/indian-data.csv",delimiter=";",skiprows=[0,2],index_col="Step")
delta_theta_oi = 10.6 # Steady: 33.60901666696968
delta_theta_hi = 8.3 # Steady: 26.613290151743797

input_df["K"] = 0.81
input_df["theta_a"] = 30.3



############
# Simulate #
############

print(p)

output_df_exp = simulate_iec_exp(input_df.copy(), input_df.loc[0, "K"], delta_theta_oi, delta_theta_hi, p)
output_df_exp.to_csv("const/outputs/output-exp.csv",sep=";")
transformer_plot(output_df_exp,"const/sim-plots/exp.png")

output_df_diff = simulate_iec_diff(input_df.copy(), p, delta_theta_oi, delta_theta_hi)
output_df_diff.to_csv("const/outputs/output-diff.csv",sep=";")
transformer_plot(output_df_diff,"const/sim-plots/diff.png")

output_df_diff_2 = simulate_iec_diff_2(input_df.copy(), p, delta_theta_oi, delta_theta_hi)
output_df_diff_2.to_csv("const/outputs/output-diff-2.csv",sep=";")
transformer_plot(output_df_diff_2,"const/sim-plots/diff-2.png")

output_df_diff_2r = simulate_iec_diff_2r(input_df.copy(), p, delta_theta_oi, delta_theta_hi)
output_df_diff_2r.to_csv("const/outputs/output-diff-2r.csv",sep=";")
transformer_plot(output_df_diff_2r,"const/sim-plots/diff-2r.png")



###################
# Compare Results #
###################

transformer_comparison_plot(output_df_exp, output_df_diff, "const/com-plots/exp-vs-diff.png")
transformer_comparison_plot(output_df_exp, output_df_diff_2, "const/com-plots/exp-vs-diff-2.png")
transformer_comparison_plot(output_df_exp, output_df_diff_2r, "const/com-plots/exp-vs-diff-2r.png")

rmse = df_rmse(output_df_exp, output_df_diff, ["theta_o", "theta_h", "L"])
print("\nexp vs diff:")
print(f"   RMSE theta_o = {rmse['theta_o']}")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")

rmse = df_rmse(output_df_exp, output_df_diff_2, ["theta_o", "theta_h", "L"])
print("\nexp vs diff_2:")
print(f"   RMSE theta_o = {rmse['theta_o']}")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")

rmse = df_rmse(output_df_exp, output_df_diff_2r, ["theta_o", "theta_h", "L"])
print("\nexp vs diff_2r:")
print(f"   RMSE theta_o = {rmse['theta_o']}")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")
