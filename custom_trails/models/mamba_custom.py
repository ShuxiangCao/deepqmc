import math
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, Mapping


@dataclass
class ModelArgs:
    """
    Data class for storing model-specific arguments.

    Args:
        d_model (int): Model dimension.
        n_layer (int): Number of layers.
        d_state (int, optional): State dimension (default: 16).
        expand (int, optional): Expansion factor (default: 2).
        dt_rank (Union[int, str], optional): Rank for Δ (default: "auto").
        d_conv (int, optional): Convolution dimension (default: 4).
        conv_bias (bool, optional): Whether to use bias in convolution layers (default: True).
        bias (bool, optional): Whether to use bias in linear layers (default: False).

    Attributes:
        d_inner (int): Inner dimension calculated as expand * d_model.

    Notes:
        - If dt_rank is set to "auto", it computes it as the ceiling of d_model / 16.
        - Ensures that vocab_size is a multiple of pad_vocab_size_multiple.
    """ # noqa: E501

    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)



class ResidualBlock:
    def __init__(
        self, layer_id: int, weights: Mapping[str, jnp.ndarray], args: ModelArgs
    ):
        """
        Residual block for Mamba-based models.

        Args:
            layer_id (int): Identifier for the layer.
            weights (Mapping[str, np.ndarray]): Pre-trained weights.
            args (ModelArgs): Model-specific arguments.
        """
        self.args = args
        self.mixer = MambaBlock(
            in_proj=hk.Linear(64),
            conv1d=hk.Conv1D(
            ),
            x_proj=hk.Linear(64),
            dt_proj=hk.Linear(64),
            out_proj=hk.Linear(64),
            args=args,
        )

        self.norm = hk.RMSNorm(axis=-1, create_scale=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the residual block.

        Args:
            x (np.ndarray): shape (b, l, d).

        Returns:
            np.ndarray: shape (b, l, d).

        Official Implementation:
            Block.forward(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L142

            Note: The official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ...

        """ # noqa: E501
        output = self.mixer(self.norm(x)) + x
        return output


class MambaBlock:
    def __init__(
        self,
        in_proj: hk.Linear,
        conv1d: hk.Conv1D,
        x_proj: hk.Linear,
        dt_proj: hk.Linear,
        A_log: jnp.ndarray,
        D: jnp.ndarray,
        out_proj: hk.Linear,
        args: ModelArgs,
    ):
        """
        A single Mamba block, as described in Figure 3 in Section 3.4 of the Mamba paper [1].

        Args:
            in_proj (Linear): shape (d, 2*d_in). Linear layer for input projection.
            conv1d (MambaConv1d): shape (d_in, 1, d_conv). Mamba-specific 1D convolutional layer.
            x_proj (Linear): shape (d_in, dt_rank+2*d_state). Linear layer for projecting input-specific Δ, B, and C.
            dt_proj (Linear): shape (dt_rank, d_in). Linear layer for projecting Δ dt_rank to d_in.
            A_log (np.ndarray): shape (d_in, d). Matrix A_log.
            D (np.ndarray): shape (d_in,). Vector D.
            out_proj (Linear): shape (d_in, d). Linear layer for output projection.
            args (ModelArgs): Model-specific arguments.
        """ # noqa: E501
        self.args = args
        self.in_proj: hk.Linear = in_proj
        self.conv1d: hk.Conv1D = conv1d
        self.x_proj: hk.Linear = x_proj
        self.dt_proj: hk.Linear = dt_proj
        self.A_log: jnp.ndarray = A_log
        self.D: jnp.ndarray = D
        self.out_proj: hk.Linear = out_proj

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Mamba block.

        Args:
            x (np.ndarray): Input tensor of shape (b, l, d).

        Returns:
            np.ndarray: Output tensor of shape (b, l, d).
        """
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = jnp.split(
            x_and_res,
            indices_or_sections=(self.args.d_inner, 2 * self.args.d_inner),
            axis=-1,
        )[:-1]

        x = self.conv1d(x)
        x = jax.nn.silu(x)

        y = self.ssm(x)
        y = y * jax.nn.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1] [1].
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x (np.ndarray): shape (b, l, d_in).

        Returns:
            np.ndarray: shape (b, l, d_in).

        Official Implementation:
            mamba_inner_ref(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        References:
            [1] Mamba paper: https://arxiv.org/abs/2106.16067
            [2] The Annotated S4: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        """ # noqa: E501
        (d_in, n) = self.A_log.shape

        # Compute ∆, A, B, C, D (state space parameters)
        # A and D are input-independent (see Mamba paper [1], Section 3.5.2 for A's interpretation) # noqa: E501
        # ∆, B, C are input-dependent (a key difference between Mamba and linear time-invariant S4) # noqa: E501

        A = -jnp.exp(self.A_log.astype(float))  # shape (d_in, n)
        D = self.D.astype(float)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = jnp.split(
            x_dbl,
            indices_or_sections=(
                self.args.dt_rank,
                self.args.dt_rank + n,
                self.args.dt_rank + 2 * n,
            ),
            axis=-1,
        )[
            :-1
        ]  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = jax.nn.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # Similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(
        self,
        u: jnp.ndarray,
        delta: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        D: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Performs the selective scan algorithm as described in the Mamba paper [1].
        This function computes the output based on input data and state space parameters.
        See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u (np.ndarray): shape (b, l, d_in). Input tensor.
            delta (np.ndarray): shape (b, l, d_in). Step size tensor.
            A (np.ndarray): shape (d_in, n). Matrix A.
            B (np.ndarray): shape (b, l, n). Tensor B.
            C (np.ndarray): shape (b, l, n). Tensor C.
            D (np.ndarray): shape (d_in,). Vector D.

        Returns:
            np.ndarray: Output tensor of shape (b, l, d_in).

        Official Implementation:
            selective_scan_ref(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: Some parts have been refactored from `selective_scan_ref`, so the functionality may not match exactly.

        References:
            [1] Mamba paper: https://arxiv.org/abs/2106.16067
            [2] The Annotated S4: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        """ # noqa: E501
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        deltaA = jnp.exp(jnp.einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = jnp.einsum(
            delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n"
        )

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that is additionally hardware-aware (like FlashAttention). # noqa: E501
        x = jnp.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = jnp.einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        y = jnp.stack(ys, axis=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class Mamba:
    def __init__(self, weights: Mapping[str, jnp.ndarray], args: ModelArgs):
        """
        Full Mamba model.

        Args:
            weights (Mapping[str, np.ndarray]): Pre-trained weights.
            args (ModelArgs): Model-specific arguments.
        """
        self.args = args
        self.layers = [
            ResidualBlock(i, weights, args) for i in range(args.n_layer)
        ]
        self.norm_f = hk.RMSNorm(axis=-1, create_scale=True)

        # Tie output projection to embedding weights.
        # See "Weight Tying" paper
        self.lm_head = hk.Linear()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Mamba model.

        Args:
            x (jnp.ndarray): the embedding tensor of shape (b, l, d_model).

        Returns:
            np.ndarray: shape (b, l, vocab_size). The output logits tensor.

        Official Implementation:
            class MambaLMHeadModel, see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L118
        """

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids: jnp.ndarray, max_new_tokens: int):
        _, L = input_ids.shape
        for _ in range(L, max_new_tokens):
            logits = self(input_ids)[:, -1]
            next_id = jnp.argmax(logits, axis=-1, keepdims=True)
            input_ids = jnp.concatenate([input_ids, next_id], axis=-1)
            yield next_id


class MambaForElectron(hk.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, name: str = "SimpleRNN", **kwargs):
        """
        A simple RNN model for embedding sequences.
        Args:
            hidden_size: The size of the hidden state of the RNN.
            num_layers: The number of RNN layers.
            name: The name of the Haiku module.
        """
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs
