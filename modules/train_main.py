"""!
@file train_main.py
@brief The training loop algorithm

This file contains a single function, <code>improve_system</code>, which
implements the main training loop.
"""

import numpy as np
import time

from modules.train_utils import numpy_to_torch, ExponentialMovingAverage, get_batch, train_plot
from modules.inout import train_log, print_train_state

import torch
import torch.nn as nn
import torch.optim as optim

from train_params import device, model_type, learning_rate, RNNFunc, out_directory
from train_params import epochs, test_freq, n_batches, batch_duration



# ################# #
# The Training Loop #
# ################# #

def improve_system(
        x_0_ref: np.ndarray[3,:],
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        u: np.ndarray[3,:],
        delta_t: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:]
) -> ( np.ndarray[3,3], np.ndarray[3,3] ):
    """! @brief The training loop
    @param x_0_ref: np.ndarray[3,T], initial state vector's time series
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param u: np.ndarray[3,T], input vector's time series
    @param delta_t: np.ndarray[T], time between current and previous sample
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series

    @return best_A_prime: np.ndarray[3,3], system matrix improvement
    @return best_B_prime: np.ndarray[3,3], input matrix improvement

    This function implements the training loop of the Recurrent Neural Network.
    It automatically performs the PyTorch conversion internally, so both its
    inputs and its outputs are in Numpy format. Each iteration consist of a
    simulation, a loss computation and a backpropagation. After a certain amount
    of iterations, a simulation is carried on the whole dataset to evaluate
    the performance and report it together with plots. The final outputs of this
    function are the improvement matrices <code>best_A_prime</code> and
    <code>best_B_prime</code>, i.e. the correction masks to sum to the model.
    """

    # Input Parsing to Torch

    x_0_ref, y_ref, A, B, CM, u, delta_t = numpy_to_torch(
        x_0_ref=x_0_ref,
        A=A, B=B, C=C, M=M,
        u=u, delta_t=delta_t,
        theta_o_ref=theta_o_ref, theta_h_ref=theta_h_ref
    )

    # Model Instantiation

    model = RNNFunc(A=A, B=B, CM=CM).to(device, dtype=model_type)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    time_meter = ExponentialMovingAverage(0.97)
    loss_meter = ExponentialMovingAverage(0.97)

    # Model Log

    print(f"Torch device: {device}")
    print(f"Torch dtype: {model_type}")
    print(f"RNNFunc: {model.get_name()}")

    print(f"Network has {len(list(model.parameters()))} parameters, shaped:")
    for par in model.parameters():
        print("    " + str(par.shape))

    train_log(out_directory+"/logs.txt")

    # Initial Test

    print("\nTraining Started")

    test_x0, test_delta_t, test_y_ref, test_u = get_batch(
        n_batches=1,
        batch_duration=delta_t.size(0), # T
        u=u, delta_t=delta_t, x_0_ref=x_0_ref, y_ref=y_ref
    )  # (1, 3, 1) (T, 1, 1, 1) (T, 1, 2, 1) (T, 1, 3, 1)

    with torch.no_grad():
        x_pred = model.forward(test_x0, test_u, test_delta_t)  # (T, 1, 3, 1)
        y_pred = model.x_to_y(x_pred)  # (T, 1, 2, 1)
        loss = model.total_loss(y_pred, test_y_ref)
        print(type(loss))
        print(loss)
        print(loss.shape)

        best_loss = loss
        best_A_prime = model.get_A_prime_np()
        best_B_prime = model.get_B_prime_np()

        itr = 0
        print("Epoch {:4d} | Test Loss {:9.6f}".format(itr, loss.item()))
        train_plot(test_y_ref[:,0], y_pred[:,0], itr=itr)
        print_train_state(
            itr=itr, test_loss=loss.item(),
            minibatch_loss=0., time_ms=0,
            A_prime=best_A_prime, B_prime=best_B_prime,
            out_path=out_directory+"/train-state/"+"{:04d}".format(itr)+".txt"
        )

    # Training Loop

    for itr in range(1, epochs+1):
        timer_start = time.time()
        optimizer.zero_grad()

        batch_x0, batch_delta_t, batch_y_ref, batch_u = get_batch(
            n_batches=n_batches,
            batch_duration=batch_duration,
            u=u, delta_t=delta_t, x_0_ref=x_0_ref, y_ref=y_ref
        ) # (B, 3, 1) (D, B, 1, 1) (D, B, 2, 1) (D, B, 3, 1)

        x_pred = model.forward(batch_x0, batch_u, batch_delta_t) # (D, B, 3, 1)
        y_pred = model.x_to_y(x_pred) # (D, B, 2, 1)
        loss = model.total_loss(y_pred, batch_y_ref)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - timer_start)
        loss_meter.update(loss.item())

        # Periodic Test

        if itr % test_freq == 0:
            with torch.no_grad():
                x_pred = model.forward(test_x0, test_u, test_delta_t)  # (T, 1, 3, 1)
                y_pred = model.x_to_y(x_pred)  # (T, 1, 2, 1)
                loss = model.total_loss(y_pred, test_y_ref)

                curr_A_prime = model.get_A_prime_np()
                curr_B_prime = model.get_B_prime_np()

                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_A_prime = curr_A_prime
                    best_B_prime = curr_B_prime

                time_ms = int( 1000 * time_meter.avg )
                print(
                    "Epoch {:4d} | Test Loss {:9.6f} | Avg Minibatch Loss {:9.6f} | Avg Epoch Time {:4d} ms"
                    .format(itr, loss.item(), loss_meter.avg, time_ms)
                )
                train_plot(test_y_ref[:,0], y_pred[:,0], itr=itr)
                print_train_state(
                    itr=itr, test_loss=loss.item(),
                    minibatch_loss=loss_meter.avg, time_ms=time_ms,
                    A_prime=curr_A_prime, B_prime=curr_B_prime,
                    out_path=out_directory+"/train-state/"+"{:04d}".format(itr)+".txt"
                )

    print("Training Completed Successfully")

    return best_A_prime, best_B_prime
