import haiku as hk
import jax

activations = {
    'sin': jax.lax.sin,
    'tan': jax.lax.tan,
}

class EmbeddingResNet(hk.Module):
    def __init__(self, n_up, n_down, **kwargs):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.kwargs = kwargs

    def __call__(self, x):
        def resnet_block(x):
            residual = x
            h = hk.Linear(64)(x)
            h = jax.nn.softmax(h)
            h = hk.Linear(64)(h)
            return h + residual  # Add the residual connection

        # Apply ResNet block to embeddings
        x = resnet_block(x)
        x = resnet_block(x)
        x = resnet_block(x)

        return x

model = EmbeddingResNet