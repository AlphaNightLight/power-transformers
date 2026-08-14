"""!
@file train_utils.py
@brief Functions used in the training process

This file contains the functions invoked by the training loop
algorithm, which mainly includes tensor manipulation with PyTorch.
"""

import torch
import numpy as np

from modules.inout import mkdir

from train_params import device, model_type
from train_params import visualize_train, out_directory



# ############# #
# Torch Parsing #
# ############# #

def get_batch(
        n_batches: int, # B
        batch_duration: int, # D
        u: torch.tensor, # (T, 3, 1)
        delta_t: torch.tensor, # (T)
        x_0_ref: torch.tensor, # (T, 3, 1)
        y_ref: torch.tensor, # (T, 2, 1)
) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
    """! @brief Samples the batches for training
    @param n_batches: integer, the number of batches (B)
    @param batch_duration: integer, the time duration of a batch (D)
    @param u: torch.tensor[T,3,1], input vector's time series
    @param delta_t: torch.tensor[T], time between current and previous sample
    @param x_0_ref: torch.tensor[T,3,1], initial state vector's time series
    @param y_ref: torch.tensor[T,2,1], reference output's time series

    @return batch_x0: torch.tensor[B,3,1], initial state vector's batch
    @return batch_delta_t: torch.tensor[D,B,1,1], time delta's batch
    @return batch_y_ref: torch.tensor[D,B,2,1], reference output's batch
    @return batch_u: torch.tensor[D,B,3,1], input vector's batch

    This function is used in <i>Batch Training</i> to convert a monolithic
    dataset into chunks. First, a set of time instants <code>s</code> containing
    <code>B</code> values is randomly generated. Those times are used to extract
    the corresponded initial state from the initial state vector, obtaining its
    batch. For the other tensors, we do not extract only the indicated samples,
    but also the following <code>D</code>, which generates little time series.
    """

    data_size = delta_t.size(0) # T

    s = torch.from_numpy(
        np.random.choice(
            np.arange(data_size - batch_duration + 1, dtype=np.int64),
            n_batches,
            replace=False
    ))

    batch_x0 = x_0_ref[s] # (B, 3, 1)

    batch_delta_t = torch.stack(
        [delta_t[s + i] for i in range(batch_duration)],
        dim=0
    ).unsqueeze(2).unsqueeze(3) # (D, B, 1, 1)

    batch_y_ref = torch.stack(
        [y_ref[s + i] for i in range(batch_duration)],
        dim=0
    ) # (D, B, 2, 1)

    batch_u = torch.stack(
        [u[s + i] for i in range(batch_duration)],
        dim=0
    ) # (D, B, 3, 1)

    return (
        batch_x0.to(device, dtype=model_type),
        batch_delta_t.to(device, dtype=model_type),
        batch_y_ref.to(device, dtype=model_type),
        batch_u.to(device, dtype=model_type)
    )



def numpy_to_torch(
        x_0_ref: np.ndarray[3,:],
        A: np.ndarray[3,3],
        B: np.ndarray[3,3],
        C: np.ndarray[1,3],
        M: np.ndarray[1,3],
        u: np.ndarray[3,:],
        delta_t: np.ndarray[:],
        theta_o_ref: np.ndarray[:],
        theta_h_ref: np.ndarray[:]
) -> (
        torch.tensor, torch.tensor,
        torch.tensor, torch.tensor, torch.tensor,
        torch.tensor, torch.tensor
):
    """! @brief Converts numpy variables into PyTorch tensors
    @param x_0_ref: np.ndarray[3,T], initial state vector's time series
    @param A: np.ndarray[3,3], system matrix
    @param B: np.ndarray[3,3], input matrix
    @param C: np.ndarray[1,3], measurable output matrix
    @param M: np.ndarray[1,3], unmeasurable output matrix
    @param u: np.ndarray[3,T], input vector's time series
    @param delta_t: np.ndarray[T], time between current and previous sample
    @param theta_o_ref: np.ndarray[T], reference top-oil temperature's time series
    @param theta_h_ref: np.ndarray[T], reference hotspot temperature's time series

    @return x_0_ref: torch.tensor[T,3,1], x_0_ref in PyTorch tensor format
    @return y_ref: torch.tensor[T,2,1], theta_o_ref and theta_h_ref
    stacked as a PyTorch tensor
    @return A: torch.tensor[3,3], A in PyTorch tensor format
    @return B: torch.tensor[3,3], B in PyTorch tensor format
    @return CM: torch.tensor[2,3], C and M stacked as a PyTorch tensor
    @return u: torch.tensor[T,3,1], u in PyTorch tensor format
    @return delta_t: torch.tensor[T], delta_t in PyTorch tensor format

    The system simulation has been implemented in NumPy, while the RNN training
    requires PyTorch tensors to work. This function has the duty to convert all
    the system matrices, inputs, and references from the former to the latter format.
    Additionally, it stacks the measurable and unmeasurable outputs (and consequently
    their relative matrices) into a unique tensor, since both of them are accessible
    during training.
    """

    x_0_ref = torch.from_numpy(
        x_0_ref
    ).unsqueeze(2).permute((1,0,2)).to(device, dtype=model_type) # (T, 3, 1)

    y_ref = torch.from_numpy(
        np.vstack((theta_o_ref, theta_h_ref))
    ).unsqueeze(2).permute((1,0,2)).to(device, dtype=model_type) # (T, 2, 1)

    A = torch.from_numpy(A).to(device, dtype=model_type) # (3,3)
    B = torch.from_numpy(B).to(device, dtype=model_type) # (3,3)
    CM = torch.from_numpy( np.vstack((C,M)) ).to(device, dtype=model_type)  # (2,3)

    u = torch.from_numpy(
        u
    ).unsqueeze(2).permute((1,0,2)).to(device, dtype=model_type) # (T, 3, 1)
    delta_t = torch.from_numpy(delta_t).to(device, dtype=model_type)  # (T)

    return x_0_ref, y_ref, A, B, CM, u, delta_t



# ############# #
# Visualization #
# ############# #

if visualize_train:
    import matplotlib.pyplot as plt
    mkdir(out_directory+"/train-pics")

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax = fig.add_subplot(111, frameon=True)
    plt.show(block=False)

def train_plot(
        y_ref: torch.tensor, # (T, 2, 1)
        y_pred: torch.tensor, # (T, 2, 1)
        itr: int
) -> None:
    """! @brief Quick plot for training monitoring
    @param y_ref: torch.tensor[T,2,1], the reference output
    @param y_pred: torch.tensor[T,2,1], the predicted output
    @param itr: integer, the current training iteration
    @return None

    This function provides a simple plot of the outputs predicted this iteration
    versus their reference values. It is used during training to graphically monitor
    the performance evolution of the network.
    """

    if visualize_train:
        ax.cla()
        ax.grid()
        ax.set_title("Temperatures, Training Iteration "+"{:4d}".format(itr))
        ax.set_xlabel("t [min]")
        ax.set_ylabel(r"$\theta$ [°C]")

        ax.plot(y_ref.cpu().numpy()[:, 0, 0], label=r"$\theta_o$ Reference", color="tab:orange")
        ax.plot(y_ref.cpu().numpy()[:, 1, 0], label=r"$\theta_h$ Reference", color="tab:red")
        ax.plot(y_pred.cpu().numpy()[:, 0, 0], label=r"$\theta_o$ Model", color="tab:brown", linestyle="dashed")
        ax.plot(y_pred.cpu().numpy()[:, 1, 0], label=r"$\theta_h$ Model", color="tab:purple", linestyle="dashed")
        ax.legend(loc="upper left")

        fig.tight_layout()
        plt.savefig(out_directory+"/train-pics/"+"{:04d}".format(itr))
        plt.draw()
        plt.pause(0.001)

    return



# ############## #
# Moving Average #
# ############## #

class ExponentialMovingAverage:
    """! @brief Class that implements an exponential moving average

    This class stores an <code>avg</code> value containing an
    exponential moving average, which can be updated and reset with
    the appropriate methods.
    """

    def __init__(self, momentum:int|float):
        """! @brief Class constructor
        @param momentum: int|float, the weight of the average
        """
        ## Momentum
        self.m = momentum
        ## Current Average
        self.avg = None

    def reset(self) -> None:
        """! @brief Resets the average to None
        @return None
        """
        self.avg = None
        return

    def update(self, val:int|float) -> None:
        """! @brief Updates the average
        @param val: int|float, the new value to insert
        @return None
        """

        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.m * self.avg + (1-self.m) * val
        return
