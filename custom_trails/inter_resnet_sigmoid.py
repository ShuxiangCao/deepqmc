
from deepqmc.wf.nn_wave_function import eval_log_slater, Psi
from deepqmc.wf.env import ExponentialEnvelopes
from deepqmc.physics import pairwise_diffs
from deepqmc.app import instantiate_ansatz
from deepqmc.types import PhysicalConfiguration
import jax.numpy as jnp
import jax

import haiku as hk
class MyWF(hk.Module):
    def __init__(
        self,
        hamil,
    ):
        super().__init__()
        self.mol = hamil.mol
        self.n_up, self.n_down = hamil.n_up, hamil.n_down
        self.charges = hamil.mol.charges
        self.env = ExponentialEnvelopes(hamil,1,isotropic=False, per_shell=False, per_orbital_exponent=False, spin_restricted=False, init_to_ones=False, softplus_zeta=False)

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def __call__(self, phys_conf:PhysicalConfiguration, _):
        n_elec = self.n_up + self.n_down
        orb = self.env(phys_conf, None)
        elec_nuc_diffs = pairwise_diffs(phys_conf.r,phys_conf.R).reshape(n_elec, -1)
        elec_elec_diffs = pairwise_diffs(phys_conf.r, phys_conf.r)
        elec_emebeddings = jnp.concatenate((elec_nuc_diffs,jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(self.n_down)))[...,None]),axis=-1)

        # Project the embedding to dimension of 32
        elec_emebeddings = hk.Linear(64)(elec_emebeddings)

        # ResNet-style block initialization
        def resnet_block(x):
            residual = x
            h = hk.Linear(64)(x)
            h = jax.nn.softmax(h)
            h = hk.Linear(64)(h)
            return h + residual  # Add the residual connection

        # Apply ResNet block to embeddings
        h = resnet_block(elec_emebeddings)
        for _ in range(3):
            # Update h with residual block
            h = h + jnp.einsum('ijk->ik',
                               hk.nets.MLP([32, 64], activation=jax.nn.sigmoid, with_bias=False)(elec_elec_diffs))

        # Final projection
        f = hk.Linear(self.n_up + self.n_down)(h)

        # Modify orbital information
        orb *= f[None]

        sign_psi, log_psi = eval_log_slater(orb)
        sign_psi = jax.lax.stop_gradient(sign_psi)
        return Psi(sign_psi.squeeze(), log_psi.squeeze())

def initialize_wf(H):
    my_ansatz = instantiate_ansatz(H, MyWF)
    return my_ansatz
