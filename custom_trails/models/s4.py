import haiku as hk
import jax
import jax.numpy as jnp


import jax
import jax.numpy as jnp
import haiku as hk
import dataclasses
from hippox.main import Hippo
from typing import Optional
import jax
import jax.numpy as jnp
import haiku as hk
from jax.numpy.linalg import inv, matrix_power
from functools import partial
from jax.numpy import broadcast_to
from jax.tree_util import tree_map
from typing import Optional

# Most of these are taken directly from either https://github.com/srush/annotated-s4 or
# https://github.com/lindermanlab/S5, with some minor changes here and there.

def add_batch(nest, batch_size: Optional[int]):
    broadcast = lambda x: broadcast_to(x, (batch_size,) + x.shape)
    return tree_map(broadcast, nest)


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


def discretize_bilinear(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize(A, B, step, mode="zoh"):
    if mode == "bilinear":
        num, denom = 1 + .5 * step*A, 1 - .5 * step*A
        return num / denom, step * B / denom
    elif mode == "zoh":
        return jnp.exp(step*A), (jnp.exp(step*A)-1)/A * B


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def s4d_ssm(A, B, C, step):
    N = A.shape[0]
    Abar, Bbar = discretize(A, B, step, mode="zoh")
    Abar = jnp.diag(Abar)
    Bbar = Bbar.reshape(N, 1)
    Cbar = C.reshape(1, N)
    return Abar, Bbar, Cbar


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


@partial(jax.jit, static_argnums=3)
def s4d_kernel_zoh(A, C, step, L):
    kernel_l = lambda l: (C * (jnp.exp(step * A) - 1) / A * jnp.exp(l * step * A)).sum()
    return jax.vmap(kernel_l)(jnp.arange(L)).real


@jax.jit
def cauchy(v, omega, lambd):
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def causal_convolution(u, K):
    assert K.shape[0] == u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]


def linear_recurrence(A, B, C, inputs, prev_state):
    new_state = A @ prev_state + B @ inputs
    out = C @ new_state
    return out, new_state


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(shape, dtype):
        uniform = hk.initializers.RandomUniform()
        return uniform(shape, dtype)*(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
    return init


def init_log_steps(shape, dtype):
    H = shape[0]
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer()(shape=(1,), dtype=dtype)
        log_steps.append(log_step)

    return jnp.array(log_steps)


def trunc_standard_normal(key, shape):
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = jax.random.split(key)
        C = jax.nn.initializers.lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


@jax.vmap
def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(A, B, C, input_sequence, conj_sym, bidirectional):
    A_elements = A * jnp.ones((input_sequence.shape[0], A.shape[0]))
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (A_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (A_elements, Bu_elements),
                                          reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C @ x).real)(xs)
class S4(hk.Module):
    def __init__(self,
                 state_size: int,
                 basis_measure: str,
                 seq_length: int,
                 dplr: bool,
                 inference_mode: bool = False,
                 name: Optional[str] = None
    ):
        super(S4, self).__init__(name=name)
        self._state_size = state_size
        self._inference_mode = inference_mode

        _hippo = Hippo(
            state_size=state_size,
            basis_measure=basis_measure,
            dplr=dplr,
        )
        _hippo_params = _hippo()
        self._lambda_real = hk.get_parameter(
            'lambda_real',
            [self._state_size,],
            init=_hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            [self._state_size,],
            init=_hippo.lambda_initializer('imaginary')
        )
        self._lambda = jnp.clip(self._lambda_real, None, -1e-4) + 1j * self._lambda_imag

        if dplr:
            self._p = hk.get_parameter(
                'p',
                [self._state_size],
                init=_hippo.low_rank_initializer()
            )

        self._b = hk.get_parameter(
            'b',
            [self._state_size],
            init=_hippo.b_initializer()
        )

        self._c = hk.get_parameter(
            'c',
            [self._state_size, 2],
            init=hk.initializers.RandomNormal(stddev=0.5**0.5)
        )
        self._c = self._c[..., 0] + 1j * self._c[..., 1]

        self._d = hk.get_parameter(
            'd',
            [1,],
            init=jnp.ones,
        )

        self._delta = hk.get_parameter(
            'delta',
            [1,],
            dtype=jnp.float32,
            init=log_step_initializer()
        )
        self._timescale = jnp.exp(self._delta)

        if not self._inference_mode:
            if dplr:
                self._kernel = kernel_DPLR(self._lambda, self._p, self._p, self._b, self._c, self._delta, seq_length)
            else:
                self._kernel = s4d_kernel_zoh(self._lambda, self._c, self._timescale, seq_length)
        else:
            if dplr:
                self._ssm = discrete_DPLR(self._lambda, self._p, self._p, self._b, self._c, self._delta, seq_length)
            else:
                self._ssm = s4d_ssm(self._lambda, self._b, self._c, self._timescale)

            self._state = hk.get_state('state', [self._state_size], jnp.complex64, jnp.zeros)


    def __call__(self, u):
        if not self._inference_mode:
            return causal_convolution(u, self._kernel) + self._d * u
        else:
            x_k, y_s = scan_SSM(*self._ssm, u[:, jnp.newaxis], self._state)
            hk.set_state('state', x_k)
            return y_s.reshape(-1).real + self._d * u



@dataclasses.dataclass
class S4Block(hk.Module):
    ssm: S4
    d_model: int
    dropout_rate: float
    prenorm: bool = True
    glu: bool = True
    istraining: bool = False
    inference_mode: bool = False
    name: Optional[str] = None

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = layer_norm(x)
        x = hk.vmap(self.ssm, in_axes=1, out_axes=1, split_rng=True)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if self.glu:
            x = hk.Linear(self.d_model)(x) * jax.nn.sigmoid(hk.Linear(self.d_model)(x))
        else:
            x = hk.Linear(self.d_model)(x)
        x = skip + hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if not self.prenorm:
            x = layer_norm(x)
        return x


@dataclasses.dataclass
class Embedding(hk.Module):
    n_embeddings: int
    n_features: int

    def __call__(self, x):
        y = hk.Embed(self.n_embeddings, self.n_features)(x[..., 0])
        return jnp.where(x > 0, y, 0.0)


@dataclasses.dataclass
class S4Stack(hk.Module):
    ssm: S4
    d_model: int
    n_layers: int
    d_output: int
    prenorm: bool = True
    dropout_rate: float = 0.0
    embedding: bool = False
    classification: bool = False
    istraining: bool = True
    inference_mode: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        super(S4Stack, self).__post_init__(name=self.name)
        if self.embedding:
            self._encoder = Embedding(self.d_output, self.d_model)
        else:
            self._encoder = hk.Linear(self.d_model)
        self._decoder = hk.Linear(self.d_output)
        self._layers = [
            S4Block(
                ssm=self.ssm,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                istraining=self.istraining,
                inference_mode=self.inference_mode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0
            if not self.inference_mode:
                x = jnp.pad(x[:-1], [(1, 0), (0, 0)])
        x = self._encoder(x)
        for layer in self._layers:
            x = layer(x)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self._decoder(x)
        return jax.nn.log_softmax(x, axis=-1)

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

        output = S4(state_size=64,basis_measure='legs',seq_length=self.kwargs['n_up'] + self.kwargs['n_down'],dplr=True)(x)

        # Return the embeddings for all elements
        return output

model = SimpleRNN