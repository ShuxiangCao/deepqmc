import haiku as hk
import jax
import jax.numpy as jnp

class SimpleRNN(hk.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 3, name: str = "SimpleRNN", **kwargs):
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the RNN model.
        Args:
            x: Input tensor of shape [n_elem, dim].
        Returns:
            Tensor of shape [n_elem, hidden_size] representing the embeddings.
        """
        # Initialize the RNN
        rnn_layers = [
            hk.GRU(hidden_size=self.hidden_size) for _ in range(self.num_layers)
        ]

        # Initialize hidden states
        initial_states = [rnn.initial_state(x.shape[0]) for rnn in rnn_layers] # batch size = 1

        # Forward pass through RNN layers
        output = x[None, ...]
        states = initial_states
        for rnn, state in zip(rnn_layers, states):
            output, state = hk.dynamic_unroll(rnn, output, state)

        # Return the embeddings for all elements
        return output

model = SimpleRNN