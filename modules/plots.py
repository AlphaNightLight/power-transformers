"""!
@file plots.py
@brief Functions to produce and manage the plots

This file contains the functions to generate the plots relative
to the simulation of a Power Transformer System, together with a couple
of utility functions to obtain reasonable values for the figures' axis.
"""

import numpy as np
import matplotlib.pyplot as plt



# ############### #
# Axis management #
# ############### #

def tick_above(val:float, tick: int|float) -> int|float:
    """! @brief Returns smallest multiple of tick above val
    @param val: float, the threshold value to be reached
    @param tick: int|float, unit of increment to be used
    @return val_above: int|float, multiple of tick immediately above val
    """
    val_above = int(np.ceil(val / tick)) * tick
    return val_above

def tick_below(val:float, tick: int|float) -> int|float:
    """! @brief Returns biggest multiple of tick below val
    @param val: float, the threshold value to be reached
    @param tick: int|float, unit of increment to be used
    @return val_above: int|float, multiple of tick immediately below val
    """
    val_below = int(np.floor(val / tick)) * tick
    return val_below

def single_digit(val: int|float) -> int|float:
    """! @brief Rounds a value to its most significant digit
    @param val: int|float, the input value
    @return val_sd: int|float, val rounded to have a single significant digit
    """

    digits = np.floor(np.log10( np.abs(val) ))
    val_sd = np.round(val, decimals=-int(digits))

    if val_sd >= 1 or val_sd <= -1:
        val_sd = int(val_sd)

    return val_sd



# ################ #
# Plots Generation #
# ################ #

def sim_plot(
        t: np.ndarray[:],
        theta_a: np.ndarray[:],
        K: np.ndarray[:],
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        pathname:str=None
) -> None:
    """! @brief Plots the time series of a PT simulation
    @param t: np.ndarray[T], time index of the sample
    @param theta_a: np.ndarray[T], ambient temperature's time series
    @param K: np.ndarray[T], load factor's time series
    @param theta_o: np.ndarray[T], top-oil temperature's time series
    @param theta_h: np.ndarray[T], hotspot temperature's time series
    @param pathname: string|None, pathname of the output plot
    @return None

    This function generates a plot relative to a Power Transformer's
    simulation. The three temperatures are grouped into a unique plane
    with time as abscissa, while the Load factor is in a different plane
    that shares the same time coordinates.
    """

    # Boundaries

    temperature_ticks = 10
    min_temperature = tick_below(val=np.min(theta_a, initial=0.), tick=temperature_ticks)
    max_temperature = tick_above(val=np.max(theta_h), tick=temperature_ticks)

    K_ticks = 0.25
    max_K = tick_above(val=np.max(K), tick=K_ticks)

    time_ticks = single_digit(t[-1] / 10) # Around 10 ticks
    max_time = tick_above(val=t[-1], tick=time_ticks)

    # Plot

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(t, theta_h, label=r"$\theta_h$", color="tab:red")
    ax[0].plot(t, theta_o, label=r"$\theta_o$", color="tab:orange")
    ax[0].plot(t, theta_a, label=r"$\theta_a$", color="tab:blue")
    ax[0].legend(loc="upper left")

    ax[0].set_title("Temperatures")
    ax[0].set_ylabel(r"$\theta$ [°C]")
    ax[0].set_ylim([min_temperature, max_temperature])
    ax[0].set_yticks(np.arange(min_temperature, max_temperature + 1, temperature_ticks))
    ax[0].grid()

    ax[1].plot(t, K, color="tab:blue")
    ax[1].set_title("Load Factor")
    ax[1].set_ylabel("K []")
    ax[1].set_ylim([0, max_K])
    ax[1].set_yticks(np.arange(0, max_K + 0.1, K_ticks))
    ax[1].grid()

    ax[1].set_xlabel("t [min]")
    ax[1].set_xticks(np.arange(0, max_time+1, time_ticks))
    fig.tight_layout(pad=0.5)

    # Save

    if pathname is not None:
        fig.savefig(pathname, dpi=200)
        plt.close()

    return



def com_plot(
        t: np.ndarray[:],
        theta_a: np.ndarray[:],
        K: np.ndarray[:],
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:],
        pathname:str=None
) -> None:
    """! @brief Plots a PT simulation, together with the reference outputs
    @param t: np.ndarray[T], time index of the sample
    @param theta_a: np.ndarray[T], ambient temperature's time series
    @param K: np.ndarray[T], load factor's time series
    @param theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @param theta_h: np.ndarray[T], simulated hotspot temperature's time series
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series
    @param pathname: string|None, pathname of the output plot
    @return None

    This function generates a plot relative to a Power Transformer's
    simulation, together with the reference values for the outputs.
    All the temperatures are grouped into a unique plane
    with time as abscissa, simulation dashed and reference straight,
    while the Load factor is in a different plane that shares the same
    time coordinates.
    """

    # Boundaries

    temperature_ticks = 10
    min_temperature = tick_below(val=np.min(theta_a, initial=0.), tick=temperature_ticks)
    max_temperature = np.max([
        tick_above(val=np.max(theta_h), tick=temperature_ticks),
        tick_above(val=np.max(theta_h_ref), tick=temperature_ticks)
    ])

    K_ticks = 0.25
    max_K = tick_above(val=np.max(K), tick=K_ticks)

    time_ticks = single_digit(t[-1] / 10) # Around 10 ticks
    max_time = tick_above(val=t[-1], tick=time_ticks)

    # Plot

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(t, theta_h_ref, label=r"$\theta_h$ Reference", color="tab:red")
    ax[0].plot(t, theta_h, label=r"$\theta_h$ Model", color="tab:purple", linestyle="dashed")
    ax[0].plot(t, theta_o_ref, label=r"$\theta_o$ Reference", color="tab:orange")
    ax[0].plot(t, theta_o, label=r"$\theta_o$ Model", color="tab:brown", linestyle="dashed")
    ax[0].plot(t, theta_a, label=r"$\theta_a$", color="tab:blue")
    ax[0].legend(loc="upper left")

    ax[0].set_title("Temperatures")
    ax[0].set_ylabel(r"$\theta$ [°C]")
    ax[0].set_ylim([min_temperature, max_temperature])
    ax[0].set_yticks(np.arange(min_temperature, max_temperature + 1, temperature_ticks))
    ax[0].grid()

    ax[1].plot(t, K, color="tab:blue")
    ax[1].set_title("Load Factor")
    ax[1].set_ylabel("K []")
    ax[1].set_ylim([0, max_K])
    ax[1].set_yticks(np.arange(0, max_K + 0.1, K_ticks))
    ax[1].grid()

    ax[1].set_xlabel("t [min]")
    ax[1].set_xticks(np.arange(0, max_time+1, time_ticks))
    fig.tight_layout(pad=0.5)

    # Save

    if pathname is not None:
        fig.savefig(pathname, dpi=200)
        plt.close()

    return



def error_plot(
        t: np.ndarray[:],
        theta_a: np.ndarray[:],
        K: np.ndarray[:],
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:],
        pathname:str=None
) -> None:
    """! @brief Plots the errors of a PT simulation respect the reference outputs
    @param t: np.ndarray[T], time index of the sample
    @param theta_a: np.ndarray[T], ambient temperature's time series
    @param K: np.ndarray[T], load factor's time series
    @param theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @param theta_h: np.ndarray[T], simulated hotspot temperature's time series
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series
    @param pathname: string|None, pathname of the output plot
    @return None

    This function generates a plot relative to the errors of a Power
    Transformer's simulation respect the reference values for the outputs.
    Three planes share the same time coordinates as abscissa: one relative
    to the load factor, one for the ambient temperature, and one for the
    differences between the simulated and reference outputs.
    """

    theta_h_error = theta_h - theta_h_ref
    theta_o_error = theta_o - theta_o_ref

    # Boundaries

    temperature_ticks = 10
    min_ambient = tick_below(val=np.min(theta_a, initial=0.), tick=temperature_ticks)
    max_ambient = tick_above(val=np.max(theta_a), tick=temperature_ticks)

    K_ticks = 0.25
    max_K = tick_above(val=np.max(K), tick=K_ticks)

    time_ticks = single_digit(t[-1] / 10) # Around 10 ticks
    max_time = tick_above(val=t[-1], tick=time_ticks)

    min_error_float = np.min([np.min(theta_h_error), np.min(theta_o_error)])
    max_error_float = np.max([np.max(theta_h_error), np.max(theta_o_error)])

    error_ticks = single_digit( (max_error_float-min_error_float) / 10 ) # Around 10 ticks
    min_error = tick_below(val=min_error_float, tick=error_ticks)
    max_error = tick_above(val=max_error_float, tick=error_ticks)

    # Plot

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    ax[0].plot(t, theta_h_error, label=r"$\Delta\theta_h$", color="tab:red")
    ax[0].plot(t, theta_o_error, label=r"$\Delta\theta_o$", color="tab:orange")
    ax[0].legend(loc="upper left")

    ax[0].set_title("Temperature Errors")
    ax[0].set_ylabel(r"$\Delta\theta$ [°C]")
    ax[0].set_ylim([min_error, max_error])
    ax[0].set_yticks(np.arange(min_error, max_error + 1, error_ticks))
    ax[0].grid()

    ax[1].plot(t, theta_a, color="tab:blue")
    ax[1].set_title("Ambient Temperature")
    ax[1].set_ylabel(r"$\theta_a$ [°C]")
    ax[1].set_ylim([min_ambient, max_ambient])
    ax[1].set_yticks(np.arange(min_ambient, max_ambient + 1, temperature_ticks))
    ax[1].grid()

    ax[2].plot(t, K, color="tab:blue")
    ax[2].set_title("Load Factor")
    ax[2].set_ylabel("K []")
    ax[2].set_ylim([0, max_K])
    ax[2].set_yticks(np.arange(0, max_K + 0.1, K_ticks))
    ax[2].grid()

    ax[2].set_xlabel("t [min]")
    ax[2].set_xticks(np.arange(0, max_time+1, time_ticks))
    fig.tight_layout(pad=0.5)

    # Save

    if pathname is not None:
        fig.savefig(pathname, dpi=200)
        plt.close()

    return



def ll_plot(
        t: np.ndarray[:],
        ll: np.ndarray[:],
        ll_ref: np.ndarray[:],
        pathname:str=None
) -> None:
    """! @brief Plots the loss of life resulting from a PT simulation
    @param t: np.ndarray[T], time index of the sample
    @param ll: np.ndarray[T], simulated loss of life's time series
    @param ll_ref: np.ndarray[T], reference loss of life's time series
    @param pathname: string|None, pathname of the output plot
    @return None

    This function generates a plot relative to the loss of life of a
    Power Transformer. A plane directly compare the simulated loss of life
    with its reference value, while the second one presents the difference
    between the two. The time coordinates are shared.
    """

    ll_error = ll - ll_ref

    # Boundaries

    time_ticks = single_digit(t[-1] / 10) # Around 10 ticks
    max_time = tick_above(val=t[-1], tick=time_ticks)

    min_ll_error_float = np.min(ll_error)
    max_ll_error_float = np.max(ll_error)
    max_ll_float = np.max([np.max(ll), np.max(ll_ref)])

    ll_error_ticks = single_digit( (max_ll_error_float-min_ll_error_float) / 10 ) # Around 10 ticks
    min_ll_error = tick_below(val=min_ll_error_float, tick=ll_error_ticks)
    max_ll_error = tick_above(val=max_ll_error_float, tick=ll_error_ticks)

    ll_ticks = single_digit( max_ll_float / 10 ) # Around 10 ticks
    max_ll = tick_above(val=max_ll_float, tick=ll_ticks)

    # Plot

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    ax[0].plot(t, ll_error, color="tab:green")

    ax[0].set_title("Loss of Life Error")
    ax[0].set_ylabel(r"$\Delta\ell$ [min]")
    ax[0].set_ylim([min_ll_error, max_ll_error])
    ax[0].set_yticks(np.arange(min_ll_error, max_ll_error+0.1, ll_error_ticks))
    ax[0].grid()

    ax[1].plot(t, ll_ref, label="Reference", color="tab:green")
    ax[1].plot(t, ll, label="Model", color="tab:olive", linestyle="dashed")
    ax[1].legend(loc="upper left")

    ax[1].set_title("Loss of Life")
    ax[1].set_ylabel(r"$\ell$ [min]")
    ax[1].set_ylim([0, max_ll])
    ax[1].set_yticks(np.arange(0, max_ll+1, ll_ticks))
    ax[1].grid()

    ax[1].set_xlabel("t [min]")
    ax[1].set_xticks(np.arange(0, max_time+1, time_ticks))
    fig.tight_layout(pad=0.5)

    # Save

    if pathname is not None:
        fig.savefig(pathname, dpi=200)
        plt.close()

    return



def pmf_plot(
        theta_o: np.ndarray[:],
        theta_h: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:],
        pathname:str=None
) -> None:
    """! @brief Plots the error distribution of PT simulation
    @param theta_o: np.ndarray[T], simulated top-oil temperature's time series
    @param theta_h: np.ndarray[T], simulated hotspot temperature's time series
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series
    @param pathname: string|None, pathname of the output plot
    @return None

    This function generates a plot relative to the error distribution of
    a Power Transformer. On the abscissa, error intervals are presented: the
    histogram line above them, indicates the percentage of the simulation
    error that are in their range.
    Hence, this plot represents the <i>Probability Mass Function</i> of the error
    to belong to a given interval.
    Two planes like this are provided: one for the top-oil temperature and one
    for the hotspot temperature.
    """

    theta_h_error = theta_h - theta_h_ref
    theta_o_error = theta_o - theta_o_ref

    # Boundaries

    min_error_float = np.min([np.min(theta_h_error), np.min(theta_o_error)])
    max_error_float = np.max([np.max(theta_h_error), np.max(theta_o_error)])

    error_ticks = single_digit( (max_error_float - min_error_float) / 10 ) # Around 10 ticks
    min_error = tick_below(val=min_error_float, tick=error_ticks)
    max_error = tick_above(val=max_error_float, tick=error_ticks)

    n_bins = int(2 * (max_error-min_error) // error_ticks ) # 2 bins per tick

    # Plot

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    ax[0].hist(
        theta_o_error,
        bins=n_bins,
        range=[min_error, max_error],
        weights= 100/theta_o_error.size * np.ones(theta_o_error.size),
        color='tab:orange',
        edgecolor='black'
    )
    ax[0].set_title("Top-oil Temperature Error Distribution")
    ax[0].set_ylabel(r"pmf($\Delta\theta_o$) [%]")
    ax[0].set_ylim([0, 100])
    ax[0].set_yticks(np.arange(0, 100 + 1, 10))
    ax[0].grid(alpha=0.75)

    ax[1].hist(
        theta_h_error,
        range=[min_error, max_error],
        weights= 100/theta_h_error.size * np.ones(theta_h_error.size),
        bins=n_bins,
        color='tab:red',
        edgecolor='black'
    )
    ax[1].set_title("Hotspot Temperature Error Distribution")
    ax[1].set_ylabel(r"pmf($\Delta\theta_h$) [%]")
    ax[1].set_ylim([0, 100])
    ax[1].set_yticks(np.arange(0, 100 + 1, 10))
    ax[1].grid(alpha=0.75)

    ax[1].set_xlabel(r"$\Delta\theta$ [°C]")
    ax[1].set_xticks(np.arange(min_error, max_error + 1, error_ticks))
    fig.tight_layout(pad=0.5)

    # Save

    if pathname is not None:
        fig.savefig(pathname, dpi=200)
        plt.close()

    return
