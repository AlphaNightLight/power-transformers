import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iec import IECparams, loss_of_life
from iec import iec_exp_o, iec_exp_h, iec_diff_init, iec_diff
from iec import iec_diff_2_init, iec_diff_2, iec_diff_2r_init, iec_diff_2r



########################
# Dataframe Comparison #
########################

def df_rmse(
        df: pd.DataFrame,
        df_2: pd.DataFrame,
        fields: list[str]
) -> dict[str,float]:
    rmse = {}
    for f in fields:
        rmse_f = df[f].values - df_2[f].values
        rmse_f = np.sqrt( np.average( rmse_f ** 2 ))
        rmse[f] = np.round(rmse_f, decimals=2)
    return rmse



###############
# Simulations #
###############

def simulate_iec_exp(
        df: pd.DataFrame,
        K: float,
        delta_theta_oi:float,
        delta_theta_hi:float,
        p: IECparams
) -> pd.DataFrame:
    df["theta_o"] = 0.0
    df["theta_h"] = 0.0
    df["L"] = 0.0

    for i in range(len(df)):
        df.loc[i, "theta_o"] = iec_exp_o(
            df.loc[i, "t"],
            df.loc[i, "theta_a"],
            K,
            delta_theta_oi,
            p
        )
        df.loc[i, "theta_h"] = iec_exp_h(
            df.loc[i, "t"],
            df.loc[i, "theta_o"],
            K,
            delta_theta_hi,
            p
        )

        if i == 0:
            df.loc[i, "L"] = 0
        else:
            df.loc[i, "L"] = loss_of_life(
                df.loc[i - 1, "L"],
                df.loc[i, "theta_h"],
                df.loc[i, "t"] - df.loc[i - 1, "t"],
                "tu"
            )

    df["theta_o"] = np.round(df["theta_o"], decimals=1)
    df["theta_h"] = np.round(df["theta_h"], decimals=1)
    df["L"] = np.round(df["L"], decimals=0)
    df["L_d"] = np.round(df["L"] / (24 * 60), decimals=2)

    return df

def simulate_iec_diff(
        df:pd.DataFrame,
        p:IECparams,
        delta_theta_oi: float = None,
        delta_theta_hi: float = None
) -> pd.DataFrame:
    df["theta_o"] = 0.0
    df["theta_h1"] = 0.0
    df["theta_h2"] = 0.0
    df["theta_h"] = 0.0
    df["L"] = 0.0

    for i in range(len(df)):
        if i == 0:
            df.loc[i, "theta_o"], df.loc[i, "theta_h1"], df.loc[i, "theta_h2"] = iec_diff_init(
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p,
                delta_theta_oi,
                delta_theta_hi
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_h1"] + df.loc[i, "theta_h2"]
            df.loc[i, "L"] = 0
        else:
            delta_t = df.loc[i, "t"] - df.loc[i - 1, "t"]

            df.loc[i, "theta_o"], df.loc[i, "theta_h1"], df.loc[i, "theta_h2"] = iec_diff(
                df.loc[i - 1, "theta_o"],
                df.loc[i - 1, "theta_h1"],
                df.loc[i - 1, "theta_h2"],
                delta_t,
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_h1"] + df.loc[i, "theta_h2"]
            df.loc[i, "L"] = loss_of_life(
                df.loc[i - 1, "L"],
                df.loc[i, "theta_h"],
                delta_t,
                "tu"
            )

    df["theta_o"] = np.round(df["theta_o"], decimals=1)
    df["theta_h1"] = np.round(df["theta_h1"], decimals=1)
    df["theta_h2"] = np.round(df["theta_h2"], decimals=1)
    df["theta_h"] = np.round(df["theta_h"], decimals=1)
    df["L"] = np.round(df["L"], decimals=0)
    df["L_d"] = np.round(df["L"] / (24 * 60), decimals=2)

    return df

def simulate_iec_diff_2(
        df:pd.DataFrame,
        p:IECparams,
        delta_theta_oi: float = None,
        delta_theta_hi: float = None
) -> pd.DataFrame:
    df["theta_o"] = 0.0
    df["theta_ho"] = 0.0
    df["theta_ho_dot"] = 0.0
    df["theta_h"] = 0.0
    df["L"] = 0.0

    for i in range(len(df)):
        if i == 0:
            df.loc[i, "theta_o"], df.loc[i, "theta_ho"], df.loc[i, "theta_ho_dot"] = iec_diff_2_init(
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p,
                delta_theta_oi,
                delta_theta_hi
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_ho"]
            df.loc[i, "L"] = 0
        else:
            delta_t = df.loc[i, "t"] - df.loc[i - 1, "t"]

            df.loc[i, "theta_o"], df.loc[i, "theta_ho"], df.loc[i, "theta_ho_dot"] = iec_diff_2(
                df.loc[i - 1, "theta_o"],
                df.loc[i - 1, "theta_ho"],
                df.loc[i - 1, "theta_ho_dot"],
                delta_t,
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_ho"]
            df.loc[i, "L"] = loss_of_life(
                df.loc[i - 1, "L"],
                df.loc[i, "theta_h"],
                delta_t,
                "tu"
            )

    df["theta_o"] = np.round(df["theta_o"], decimals=1)
    df["theta_ho"] = np.round(df["theta_ho"], decimals=1)
    df["theta_ho_dot"] = np.round(df["theta_ho_dot"], decimals=1)
    df["theta_h"] = np.round(df["theta_h"], decimals=1)
    df["L"] = np.round(df["L"], decimals=0)
    df["L_d"] = np.round(df["L"] / (24 * 60), decimals=2)

    return df

def simulate_iec_diff_2r(
        df:pd.DataFrame,
        p:IECparams,
        delta_theta_oi: float = None,
        delta_theta_hi: float = None
) -> pd.DataFrame:
    df["theta_o"] = 0.0
    df["theta_ho"] = 0.0
    df["g_dot"] = 0.0
    df["theta_h"] = 0.0
    df["L"] = 0.0

    for i in range(len(df)):
        if i == 0:
            df.loc[i, "theta_o"], df.loc[i, "theta_ho"], df.loc[i, "g_dot"] = iec_diff_2r_init(
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p,
                delta_theta_oi,
                delta_theta_hi
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_ho"]
            df.loc[i, "L"] = 0
        else:
            delta_t = df.loc[i, "t"] - df.loc[i - 1, "t"]

            df.loc[i, "theta_o"], df.loc[i, "theta_ho"], df.loc[i, "g_dot"] = iec_diff_2r(
                df.loc[i - 1, "theta_o"],
                df.loc[i - 1, "theta_ho"],
                df.loc[i - 1, "g_dot"],
                delta_t,
                df.loc[i, "theta_a"],
                df.loc[i, "K"],
                p
            )

            df.loc[i, "theta_h"] = df.loc[i, "theta_o"] + df.loc[i, "theta_ho"]
            df.loc[i, "L"] = loss_of_life(
                df.loc[i - 1, "L"],
                df.loc[i, "theta_h"],
                delta_t,
                "tu"
            )

    df["theta_o"] = np.round(df["theta_o"], decimals=1)
    df["theta_ho"] = np.round(df["theta_ho"], decimals=1)
    df["g_dot"] = np.round(df["g_dot"], decimals=1)
    df["theta_h"] = np.round(df["theta_h"], decimals=1)
    df["L"] = np.round(df["L"], decimals=0)
    df["L_d"] = np.round(df["L"] / (24 * 60), decimals=2)

    return df



####################
# Simulation Plots #
####################

def transformer_plot(df:pd.DataFrame, pathname:str=None) -> None:
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 1]})

    ax[0].plot(df["t"],df["L"], color="tab:green")
    ax[0].set_title("Loss of Life")
    ax[0].set_ylabel("L [min]")
    ax[0].set_ylim([0,10_000])
    ax[0].set_yticks(range(0, 10_000 + 1, 2_500))
    ax[0].grid()

    ax[1].plot(df["t"], df["theta_h"], label=r"$\theta_h$", color="tab:red")
    ax[1].plot(df["t"], df["theta_o"], label=r"$\theta_o$", color="tab:orange")
    ax[1].plot(df["t"], df["theta_a"], label=r"$\theta_a$", color="tab:blue")
    ax[1].legend(loc="upper left")

    ax[1].set_title("Temperatures")
    ax[1].set_ylabel(r"$\theta$ [°C]")
    ax[1].set_ylim([20, 180])
    ax[1].set_yticks(range(25, 180 + 1, 25))
    ax[1].grid()

    ax[2].plot(df["t"], df["K"], color="tab:blue")
    ax[2].set_title("Load Factor")
    ax[2].set_ylabel("K []")
    ax[2].set_ylim([0, 2])
    ax[2].set_yticks(np.arange(0, 2 + 0.1, 0.5))
    ax[2].grid()

    ax[2].set_xlabel("t [min]")
    ax[2].set_xticks(range(0,120+1,12))
    fig.tight_layout(pad=0.5)

    if pathname is not None:
        fig.savefig(pathname, dpi=200)

    return

def transformer_comparison_plot(df:pd.DataFrame, df_2:pd.DataFrame, pathname:str=None) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(df["t"], df["theta_h"], label=r"$\theta_h$ model 1", color="tab:red")
    ax[0].plot(df_2["t"], df_2["theta_h"], label=r"$\theta_h$ model 2", color="tab:purple", linestyle="dashed")
    ax[0].plot(df["t"], df["theta_o"], label=r"$\theta_o$ model 1", color="tab:orange")
    ax[0].plot(df_2["t"], df_2["theta_o"], label=r"$\theta_o$ model 2", color="tab:brown", linestyle="dashed")
    ax[0].plot(df["t"], df["theta_a"], label=r"$\theta_a$", color="tab:blue")
    ax[0].legend(loc="upper left")

    ax[0].set_title("Temperatures")
    ax[0].set_ylabel(r"$\theta$ [°C]")
    ax[0].set_ylim([20, 180])
    ax[0].set_yticks(range(25, 180 + 1, 15))
    ax[0].grid()

    ax[1].plot(df["t"], df["K"], color="tab:blue")
    ax[1].set_title("Load Factor")
    ax[1].set_ylabel("K []")
    ax[1].set_ylim([0, 2])
    ax[1].set_yticks(np.arange(0, 2 + 0.1, 0.5))
    ax[1].grid()

    ax[1].set_xlabel("t [min]")
    ax[1].set_xticks(range(0,120+1,12))
    fig.tight_layout(pad=0.5)

    if pathname is not None:
        fig.savefig(pathname, dpi=200)

    return
