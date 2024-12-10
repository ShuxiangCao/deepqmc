# A Haiku Implementation of Mamba2 Model
# Loosely translated for simplicity and readability
# Inspired by the detailed PyTorch version

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple


class Mamba2Config:
    def __init__(self, d_model: int, n_layers: int, d_head: int, d_state: int = 64,
                 expand_factor: int = 2, d_conv: int = 4, n_groups: int = 1):
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_head = d_head
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.n_groups = n_groups

        self.d_inner = self.expand_factor * self.d_model
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0


class RMSNorm(hk.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = hk.get_parameter("weight", shape=[d_model], init=hk.initializers.Constant(1.0))

    def __call__(self, x):
        norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class ResidualBlock(hk.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model)
        self.mixer = Mamba2Block(config)

    def __call__(self, x, cache=None):
        out, cache = self.mixer(self.norm(x), cache)
        return x + out, cache


class Mamba2Block(hk.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.in_proj = hk.Linear(2 * config.d_inner + 2 * config.n_groups * config.d_state + config.n_heads)
        self.conv = hk.Conv1D(
            output_channels=config.d_inner + 2 * config.n_groups * config.d_state,
            kernel_shape=config.d_conv,
            padding="VALID",
            feature_group_count=1#config.d_inner + 2 * config.n_groups * config.d_state
        )
        self.out_proj = hk.Linear(config.d_model)

    def __call__(self, x, cache=None):
        zxbcdt = self.in_proj(x)
        z, xBC, dt = jnp.split(zxbcdt, [self.config.d_inner,
                                        self.config.d_inner + 2 * self.config.n_groups * self.config.d_state], axis=-1)

        xBC = jax.nn.silu(self.conv(xBC))
        x, B, C = jnp.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state], axis=-1)

        # Simplified recurrence
        y = x  # For brevity; replace with actual recurrence calculations

        y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(y)
        out = self.out_proj(y)
        return out, cache


class Mamba2(hk.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.layers = [ResidualBlock(config) for _ in range(config.n_layers)]

    def __call__(self, x, cache=None):
        if cache is None:
            cache = [None] * self.config.n_layers

        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, cache[i])

        return x, cache if cache[0] is not None else x

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
        self.mamba_config = Mamba2Config(d_model=64, n_layers=4, d_head=16)
        self.mamba = Mamba2(self.mamba_config)

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
        current_state = None

        # Loop over the sequence length manually
        for t in range(output.shape[0]):
            output_t, current_state = self.mamba(output[t][None,...], current_state)
            outputs.append(output_t)

        # Stack the outputs to form the final output tensor
        output = jnp.stack(outputs)

        # Return the embeddings for all elements
        return output

model = SimpleRNN