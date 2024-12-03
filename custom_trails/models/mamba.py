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
            feature_group_count=config.d_inner + 2 * config.n_groups * config.d_state
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


def forward_fn(inputs, config: Mamba2Config):
    model = Mamba2(config)
    return model(inputs)


def main():
    # Example usage
    config = Mamba2Config(d_model=9, n_layers=4, d_head=18)
    inputs = jnp.ones([1, 10, config.d_model])

    init, apply = hk.transform_with_state(lambda x: forward_fn(x, config))
    rng = jax.random.PRNGKey(42)
    params, state = init(rng, inputs)
    outputs, _ = apply(params, state, rng, inputs)
    print(outputs)


if __name__ == "__main__":
    main()
