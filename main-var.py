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
output_df_ground = pd.read_csv("dataset/indian-output.csv",delimiter=";",skiprows=[0,2],index_col="Step")



############
# Simulate #
############

print(p)

output_df_diff = simulate_iec_diff(input_df.copy(), p)
output_df_diff.to_csv("var/outputs/output-diff.csv",sep=";")
transformer_plot(output_df_diff,"var/sim-plots/diff.png")

output_df_diff_2 = simulate_iec_diff_2(input_df.copy(), p)
output_df_diff_2.to_csv("var/outputs/output-diff-2.csv",sep=";")
transformer_plot(output_df_diff_2,"var/sim-plots/diff-2.png")

output_df_diff_2r = simulate_iec_diff_2r(input_df.copy(), p)
output_df_diff_2r.to_csv("var/outputs/output-diff-2r.csv",sep=";")
transformer_plot(output_df_diff_2r,"var/sim-plots/diff-2r.png")



###################
# Compare Results #
###################

transformer_comparison_plot(output_df_diff, output_df_diff_2, "var/com-plots/diff-vs-diff-2.png")
transformer_comparison_plot(output_df_diff, output_df_diff_2r, "var/com-plots/diff-vs-diff-2r.png")

rmse = df_rmse(output_df_ground, output_df_diff, ["theta_h", "L"])
print("\nground truth vs diff:")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")

rmse = df_rmse(output_df_diff, output_df_diff_2, ["theta_o", "theta_h", "L"])
print("\ndff vs diff_2:")
print(f"   RMSE theta_o = {rmse['theta_o']}")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")

rmse = df_rmse(output_df_diff, output_df_diff_2r, ["theta_o", "theta_h", "L"])
print("\ndff vs diff_2r:")
print(f"   RMSE theta_o = {rmse['theta_o']}")
print(f"   RMSE theta_h = {rmse['theta_h']}")
print(f"   RMSE L = {rmse['L']}")
