"""!
@file train_rnn.py
@brief All the RNN classes

This file contains the implementation of all the Recurrent Neural Network
architectures examined in this project. Each network is represented by a class which
inherits from <code>RNN_base</code>, specializing the structure of
<code>A_prime</code> and <code>B_prime</code>.
"""

import numpy as np

from abc import abstractmethod
from typing_extensions import override

import torch
import torch.nn as nn
import torch.optim as optim

from train_params import device, model_type, reg_A, reg_B





# ############ #
# The Base RNN #
# ############ #

class RNN_base(nn.Module):
    """! @brief The base RNN class

    This is the class from which all other Recurrent Neural Networks
    Inherits the base functionalities. In contains abstract methods
    so it cannot be instantiated but only used to create subclasses,
    which shall implement <code>get_A_prime</code>, <code>get_B_prime</code>
    and <code>get_name</code>.
    """

    # Constructor

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__()

        # Model Matrices

        ## System Matrix
        self.A = A # (3,3)
        ## Input Matrix
        self.B = B # (3,3)
        ## Output Matrix
        self.CM = CM # (2,3)

        # Loss

        ## Mean Square Error Loss
        self.loss_mse = nn.MSELoss(reduction='mean')
        ## Regularizer for A_prime
        self.lambda_A = torch.tensor(reg_A).to(device, dtype=model_type)
        ## Regularizer for B_prime
        self.lambda_B = torch.tensor(reg_B).to(device, dtype=model_type)



    # Getters

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """! @brief <b>Abstract Method</b> to get the RNN name
        @return name: string, The RNN name
        """
        raise NotImplementedError()

    @abstractmethod
    def get_A_prime(self) -> torch.tensor:
        """! @brief <b>Abstract Method</b> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        raise NotImplementedError()

    @abstractmethod
    def get_B_prime(self) -> torch.tensor:
        """! @brief <b>Abstract Method</b> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        raise NotImplementedError()

    def get_A_prime_np(self) -> np.ndarray[3,3]:
        """! @brief Returns <code>A_prime</code> in NumPy format
        @return A_prime: np.ndarray[3,3], system matrix improvement
        """
        return self.get_A_prime().cpu().numpy()

    def get_B_prime_np(self) -> np.ndarray[3,3]:
        """! @brief Returns <code>B_prime</code> in NumPy format
        @return B_prime: np.ndarray[3,3], input matrix improvement
        """
        return self.get_B_prime().cpu().numpy()



    # System Dynamic

    @staticmethod
    def next_x(
            x_prev: torch.tensor, # (B, 3, 1)
            A_tot: torch.tensor, # (3, 3)
            B_tot: torch.tensor, # (3, 3)
            u_i: torch.tensor, # (B, 3, 1)
            delta_t_i: torch.tensor, # (B, 1, 1)
    ) -> torch.tensor:
        """! @brief Implements the system dynamic
        @param x_prev: torch.tensor[B,3,1], previous state vector's batch
        @param A_tot: torch.tensor[3,3], system matrix with improvement
        @param B_tot: torch.tensor[3,3], input matrix with improvement
        @param u_i: torch.tensor[B,3,1], current input vector's batch
        @param delta_t_i: torch.tensor[B,1,1], current time delta's batch
        @return x_i: torch.tensor[B,3,1], current state vector's batch
        """

        x_dot = A_tot @ x_prev + B_tot @ u_i # (B, 3, 1)
        return x_prev + delta_t_i * x_dot  # (B, 3, 1)

    @staticmethod
    def forward_static(
            x_0: torch.tensor, # (B, 3, 1)
            A_tot: torch.tensor,  # (3, 3)
            B_tot: torch.tensor,  # (3, 3)
            u: torch.tensor, # (D, B, 3, 1)
            delta_t:torch.tensor, # (D, B, 1, 1)
    ) -> torch.tensor:
        """! @brief Executes a system simulation
        @param x_0: torch.tensor[B,3,1], initial state vector's batch
        @param A_tot: torch.tensor[3,3], system matrix with improvement
        @param B_tot: torch.tensor[3,3], input matrix with improvement
        @param u: torch.tensor[D,B,3,1], input vector's batch
        @param delta_t: torch.tensor[D,B,1,1], time delta's batch
        @return x: torch.tensor[D,B,3,1], state vector's batch
        """

        batch_duration = delta_t.size(0) # D

        x_curr = x_0 # (B, 3, 1)
        x = x_0.unsqueeze(0) # (1, B, 3, 1)
        for i in range(1, batch_duration):  # 1..D-1
            x_curr = RNN_base.next_x(
                    x_prev=x_curr,
                    A_tot=A_tot, B_tot=B_tot,
                    u_i=u[i], delta_t_i=delta_t[i]
                ) # (B, 3, 1)

            x = torch.vstack((
                x, x_curr.unsqueeze(0)
            ))

        return x # (D, B, 3, 1)

    def forward(self,
            x_0: torch.tensor,  # (B, 3, 1)
            u: torch.tensor,  # (D, B, 3, 1)
            delta_t: torch.tensor,  # (D, B, 1, 1)
    ) -> torch.tensor:
        """! @brief Wrapper of <code>forward_static</code> to execute a system
        simulation with the internal <code>A_prime</code> and <code>B_prime</code>
        @param x_0: torch.tensor[B,3,1], initial state vector's batch
        @param u: torch.tensor[D,B,3,1], input vector's batch
        @param delta_t: torch.tensor[D,B,1,1], time delta's batch
        @return x: torch.tensor[D,B,3,1], state vector's batch
        """

        A_tot = self.A + self.get_A_prime()
        B_tot = self.B + self.get_B_prime()

        x = self.forward_static(
            x_0=x_0, A_tot=A_tot, B_tot=B_tot,
            u=u, delta_t=delta_t
        )
        return x

    def x_to_y(self,
            x: torch.tensor, # (D, B, 3, 1)
    ) -> torch.tensor:
        """! @brief Converts the state vector to the output vector
        @param x: torch.tensor[D,B,3,1], state vector's batch
        @return y: torch.tensor[D,B,2,1], output vector's batch
        """

        return self.CM @ x # (D, B, 2, 1)



    def loss_L1(self,
            y_pred: torch.tensor,
            y_ref: torch.tensor,
    ) -> torch.tensor:
        """! @brief Computes MSE loss with L1 regularizer
        @param y_pred: torch.tensor, predicted output
        @param y_ref: torch.tensor, reference output
        @return loss: torch.tensor, MSE loss with L1 regularizer
        """

        loss = self.loss_mse(y_pred, y_ref)
        loss += self.lambda_A * self.get_A_prime().abs().sum()
        loss += self.lambda_B * self.get_B_prime().abs().sum()

        return loss

    def loss_L2(self,
            y_pred: torch.tensor,
            y_ref: torch.tensor,
    ) -> torch.tensor:
        """! @brief Computes MSE loss with L2 regularizer
            @param y_pred: torch.tensor, predicted output
            @param y_ref: torch.tensor, reference output
            @return loss: torch.tensor, MSE loss with L2 regularizer
            """

        loss = self.loss_mse(y_pred, y_ref)
        loss += self.lambda_A * self.get_A_prime().pow(2).sum()
        loss += self.lambda_B * self.get_B_prime().pow(2).sum()

        return loss





# ############################# #
# How to inherit from RNN_base: #
# ############################# #

# class RNN_subclass(RNN_base):
#     def __init__(self, A:torch.tensor, B:torch.tensor, CM:torch.tensor):
#         super().__init__(A, B, CM)
#         self.total_loss = self.loss_mse # OR self.loss_L2 OR self.loss_L1
#         # Define your parameters
#
#     @staticmethod
#     @override
#     def get_name() -> str:
#         return "Network Name"
#
#     @override
#     def get_A_prime(self) -> torch.tensor:
#         return # A_prime obtained from parameters
#
#     @override
#     def get_B_prime(self) -> torch.tensor:
#         return # B_prime obtained from parameters





# ################## #
# The RNN Subclasses #
# ################## #

class RNN_a(RNN_base):
    """! @brief RNN subclass with 1 parameter in <code>A_prime</code>,
    fixed <code>B_prime</code> and MSE loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} 0 & a & a \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    Loss = MSE
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_mse



        ## Parameter <code>a</code>
        self.a = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        ) # scalar

        ## Structure masks of parameter <code>a</code>
        self.S_a = torch.as_tensor([
            [0., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Constant value of <code>B_prime</code>
        self.fixed_B = torch.zeros((3, 3)).to(device, dtype=model_type)  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_a"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.S_a * self.a

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.fixed_B





class RNN_b(RNN_base):
    """! @brief RNN subclass with fixed <code>A_prime</code>,
    1 parameter in <code>B_prime</code> and MSE loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} b & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    Loss = MSE
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_mse



        ## Constant value of <code>A_prime</code>
        self.fixed_A = torch.zeros((3, 3)).to(device, dtype=model_type)  # (3,3)

        ## Parameter <code>b</code>
        self.b = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        ) # scalar

        ## Structure masks of parameter <code>b</code>
        self.S_b = torch.as_tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_b"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.fixed_A

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.S_b * self.b





class RNN_tot_L1(RNN_base):
    """! @brief RNN subclass with full selectable <code>A_prime</code>,
    full selectable <code>B_prime</code> and L1 loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} * & * & * \\ * & * & * \\ * & * & * \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} * & * & * \\ * & * & * \\ * & * & * \end{matrix}
    \qquad\qquad\qquad
    Loss = L_1
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_L1



        ## Full value of <code>A_prime</code>
        self.full_A = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type))  # (3,3)

        ## Full value of <code>B_prime</code>
        self.full_B = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type))  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_tot_L1"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.full_A

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.full_B





class RNN_tot_L2(RNN_base):
    """! @brief RNN subclass with full selectable <code>A_prime</code>,
    full selectable <code>B_prime</code> and L2 loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} * & * & * \\ * & * & * \\ * & * & * \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} * & * & * \\ * & * & * \\ * & * & * \end{matrix}
    \qquad\qquad\qquad
    Loss = L_2
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_L2



        ## Full value of <code>A_prime</code>
        self.full_A = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type))  # (3,3)

        ## Full value of <code>B_prime</code>
        self.full_B = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type))  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_tot_L2"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.full_A

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.full_B





class RNN_p6(RNN_base):
    """! @brief RNN subclass with 4 parameters in <code>A_prime</code>,
    2 parameters in <code>B_prime</code> and MSE loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} a & b & 0 \\ 0 & 0 & c \\ 0 & 0 & d \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} e & 0 & 0 \\ 0 & 0 & 0 \\ f & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    Loss = MSE
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_L2



        ## Parameter <code>a</code>
        self.a = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>b</code>
        self.b = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>c</code>
        self.c = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>d</code>
        self.d = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>e</code>
        self.e = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>f</code>
        self.f = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Structure masks of parameter <code>a</code>
        self.S_a = torch.as_tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>b</code>
        self.S_b = torch.as_tensor([
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>c</code>
        self.S_c = torch.as_tensor([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>d</code>
        self.S_d = torch.as_tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>e</code>
        self.S_e = torch.as_tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>f</code>
        self.S_f = torch.as_tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_p6"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.S_a * self.a + self.S_b * self.b + self.S_c * self.c + self.S_d * self.d

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.S_e * self.e + self.S_f * self.f





class RNN_p5(RNN_base):
    """! @brief RNN subclass with 4 parameters in <code>A_prime</code>,
    1 parameter in <code>B_prime</code> and MSE loss

    Subclass of <code>RNN_base</code>, with the following specifications:
    \f[
    A' = \begin{matrix} a & b & 0 \\ 0 & 0 & 0 \\ 0 & c & d \end{matrix}
    \qquad\qquad\qquad
    B\ ' = \begin{matrix} e & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix}
    \qquad\qquad\qquad
    Loss = MSE
    \f]
    """

    def __init__(self,
            A: torch.tensor, # (3,3)
            B: torch.tensor, # (3,3)
            CM: torch.tensor, # (2,3)
    ):
        """! @brief Class constructor
        @param A: torch.tensor[3,3], system matrix
        @param B: torch.tensor[3,3], input matrix
        @param CM: torch.tensor[2,3], output matrix
        """

        super().__init__(A, B, CM)
        ## Selected Loss
        self.total_loss = self.loss_L2



        ## Parameter <code>a</code>
        self.a = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>b</code>
        self.b = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>c</code>
        self.c = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>d</code>
        self.d = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Parameter <code>e</code>
        self.e = nn.Parameter(
            torch.tensor(0.).to(device, dtype=model_type)
        )  # scalar

        ## Structure masks of parameter <code>a</code>
        self.S_a = torch.as_tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>b</code>
        self.S_b = torch.as_tensor([
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>c</code>
        self.S_c = torch.as_tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>d</code>
        self.S_d = torch.as_tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.]
        ]).to(device, dtype=model_type)  # (3,3)

        ## Structure masks of parameter <code>e</code>
        self.S_e = torch.as_tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]).to(device, dtype=model_type)  # (3,3)



    @staticmethod
    @override
    def get_name() -> str:
        """! @brief <i>Override Method</i> to get the RNN name
        @return name: string, The RNN name
        """
        return "RNN_p5"

    @override
    def get_A_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>A_prime</code>
        @return A_prime: torch.tensor[3,3], system matrix improvement
        """
        return self.S_a * self.a + self.S_b * self.b + self.S_c * self.c + self.S_d * self.d

    @override
    def get_B_prime(self) -> torch.tensor:
        """! @brief <i>Override Method</i> to get <code>B_prime</code>
        @return B_prime: torch.tensor[3,3], input matrix improvement
        """
        return self.S_e * self.e


