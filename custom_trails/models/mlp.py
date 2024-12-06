import haiku as hk
import jax

activations = {
    'sin': jax.lax.sin,
    'tan': jax.lax.tan,
}

class EmbeddingMLP(hk.Module):
    def __init__(self, n_up, n_down, **kwargs):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.kwargs = kwargs

    def __call__(self, x):
        kwargs = self.kwargs
        n_elec = self.n_up + self.n_down

        activation_name = kwargs.get('mlp_activation', 'sigmoid')
        activation_func = activations.get(activation_name, None)
        if activation_func is None:
            activation_func = getattr(jax.nn, activation_name)

        mlp = hk.nets.MLP(
            output_sizes=[64, 128, n_elec],
            # Example architecture: two hidden layers with 64 and 128 neurons, output matches `n_elec`
            activation=activation_func,
            activate_final=False  # Do not activate the final layer
        )
        x = mlp(x)

        return x

model = EmbeddingMLP