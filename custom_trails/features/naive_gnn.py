from functools import partial

from deepqmc.wf.nn_wave_function import eval_log_slater, Psi
from deepqmc.wf.env import ExponentialEnvelopes
from deepqmc.physics import pairwise_diffs
from deepqmc.app import instantiate_ansatz
from deepqmc.types import PhysicalConfiguration
from deepqmc.utils import triu_flat, flatten
import jax
import jax.numpy as jnp
import haiku as hk

class NaiveGNN(hk.Module):
    def __init__(
            self,
            name='NaiveGNNFeatures',
            **kwargs

    ):
        self.kwargs = kwargs
        self.n_up = kwargs['n_up']
        self.n_down = kwargs['n_down']
        super().__init__(name=name)

    def __call__(self, phys_conf: PhysicalConfiguration):
        n_elec = self.kwargs['n_up'] + self.kwargs['n_down']

        # physconf.r -> [n_elec, 3(x,y,z)]
        # physconf.R -> [n_nuc, 3(x,y,z)]

        elec_nuc_diffs = pairwise_diffs(phys_conf.r, phys_conf.R)

        # elec_nuc_diffs -> [n_elec, n_nuc * 3(x,y,z)]
        elec_nuc_dist, elec_nuc_diffs = elec_nuc_diffs[:, :, -1] ** 0.5, elec_nuc_diffs[:, :, :-1]
        elec_elec_diffs = pairwise_diffs(phys_conf.r, phys_conf.r)
        # Transform it to [n_elec, n_elec, 1]

        # INPUT FEATURES
        # ___________________________
        rescaled_diffs = elec_nuc_diffs * jnp.log(1 + elec_nuc_dist)[..., None] / elec_nuc_dist[..., None]

        local_embeddings = jnp.concatenate((
            rescaled_diffs.reshape([n_elec, -1]),
            jnp.log(1 + elec_nuc_dist)), axis=1)

        elec_emebeddings = jnp.concatenate((
            local_embeddings,
            jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(self.n_down)))[..., None]),
            axis=-1)

        # output_shape : n_elec, embedding
        # ___________________________

        # GNN BLOCK
        # ___________________________
        h = hk.Linear(64)(elec_emebeddings)
        for _ in range(3):
            h = h + jnp.einsum('ijk->ik',
                               hk.nets.MLP([32, 64], activation=jax.nn.sigmoid, with_bias=False)(elec_elec_diffs))

        return h

features = NaiveGNN