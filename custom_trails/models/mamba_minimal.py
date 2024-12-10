from dataclasses import dataclass

# A Haiku Implementation of Mamba2 Model
# Loosely translated for simplicity and readability
# Inspired by the detailed PyTorch version

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = (self.d_model + 15) // 16  # ceil equivalent

class RMSNorm(hk.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model

    def __call__(self, x):
        scale = hk.get_parameter("scale", shape=[self.d_model], init=hk.initializers.Constant(1.0))
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(mean_sq + self.eps)
        return normed * scale

class MambaBlock(hk.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.d_inner = args.d_inner
        self.conv1d = hk.Conv1D(output_channels=self.d_inner, kernel_shape=args.d_conv,
                                stride=1, padding='SAME',
                                with_bias=args.conv_bias)
        self.d_in_proj = hk.Linear(self.d_inner * 2, with_bias=args.bias)
        self.dt_proj = hk.Linear(args.d_inner, with_bias=True)
        self.x_proj = hk.Linear(args.dt_rank + args.d_state * 2, with_bias=False)
        self.out_proj = hk.Linear(args.d_model, with_bias=args.bias)

    def __call__(self, x):
        x_and_res = self.d_in_proj(x)
        x, res = jnp.split(x_and_res, 2, axis=-1)
        x = jax.nn.silu(self.conv1d(x))
        res = jax.nn.silu(res)

        # Compute selective state space elements
        delta, B, C = jnp.split(self.x_proj(x), [self.args.dt_rank, self.args.dt_rank + self.args.d_state], axis=-1)
        delta = jax.nn.softplus(self.dt_proj(delta))

        # Further computations would go here, implementing the SSM

        output = self.out_proj(x + res)  # Placeholder for the actual operation
        return output

class MambaModel(hk.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.blocks = [MambaBlock(args) for _ in range(args.n_layer)]
        self.norm_f = RMSNorm(args.d_model)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return x


class SimpleRNN(hk.Module):
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
        self.mamba = MambaModel(
            ModelArgs(
                d_model=hidden_size,
                n_layer=num_layers,
                d_state=16,
                expand=2,
                dt_rank='auto',
                d_conv=4,
                conv_bias=True,
                bias=False
        ))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the RNN model.
        Args:
            x: Input tensor of shape [n_elem, dim].
        Returns:
            Tensor of shape [n_elem, hidden_size] representing the embeddings.
        """
        # Initialize hidden states
        states = None # batch size = 1
        outputs = []
        output = x

        # Loop over the sequence length manually
        for t in range(output.shape[0]):
            output_t = self.mamba(output[t][None,...])
            outputs.append(output_t)

        # Stack the outputs to form the final output tensor
        output = jnp.stack(outputs)

        # Return the embeddings for all elements
        return output

model = SimpleRNN