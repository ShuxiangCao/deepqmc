import haiku as hk
import jax

activations = {
    'sin': jax.lax.sin,
    'tan': jax.lax.tan,
}


class AttentionBlock(hk.Module):
    def __init__(self, n_up, n_down, **kwargs):
        super().__init__()

        d_model = 64

        self.mha = hk.MultiHeadAttention(num_heads=8, key_size=64, value_size=64,
                                         w_init=hk.initializers.TruncatedNormal(stddev=d_model ** -0.5)
                                         )

        self.in_proj = hk.Linear(64)

        self.ffn = hk.Sequential([
            hk.Linear(128),
            jax.nn.tanh,
            hk.Linear(64)
        ])
        self.layer_norm_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x):
        attn_output = self.mha(x, x, x)
        out_1 = self.layer_norm_1(x + self.in_proj(attn_output))
        ffn_output = self.ffn(out_1)
        x = self.layer_norm_2(out_1 + ffn_output)
        return x


class EmbeddingAtt(hk.Module):
    def __init__(self, n_up, n_down, **kwargs):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.kwargs = kwargs

        self.attention = [AttentionBlock(n_up, n_down, **kwargs) for _ in range(3)]
        self.linear = hk.Linear(n_up + n_down)

    def __call__(self, x):
        kwargs = self.kwargs
        n_elec = self.n_up + self.n_down

        for block in self.attention:
            x = block(x)

        print(x.shape)

        return x

model = EmbeddingAtt