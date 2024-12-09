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
            hk.VanillaRNN(hidden_size=self.hidden_size) for _ in range(self.num_layers)
        ]

        # Initialize hidden states
        initial_states = [rnn.initial_state(x.shape[0]) for rnn in rnn_layers] # batch size = 1
        output = x[None, ...]
        states = initial_states

        for rnn, state in zip(rnn_layers, states):
            outputs = []
            current_state = state

            # Loop over the sequence length manually
            for t in range(output.shape[0]):
                output_t, current_state = rnn(output[t], current_state)
                outputs.append(output_t)

            # Stack the outputs to form the final output tensor
            output = jnp.stack(outputs)

        # Return the embeddings for all elements
        return output

model = SimpleRNN