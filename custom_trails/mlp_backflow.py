from deepqmc.wf.nn_wave_function import eval_log_slater, Psi
from deepqmc.wf.env import ExponentialEnvelopes
from deepqmc.physics import pairwise_diffs
from deepqmc.app import instantiate_ansatz
from deepqmc.types import PhysicalConfiguration
from deepqmc.utils import triu_flat, flatten
import jax
import jax.numpy as jnp
import haiku as hk

from deepqmc.wf.cusp import PsiformerCusp, ElectronicCuspAsymptotic


class MyWF(hk.Module):
    def __init__(
            self,
            hamil,
            n_determinants=16,
    ):
        super().__init__()
        self.mol = hamil.mol
        self.n_up, self.n_down = hamil.n_up, hamil.n_down
        self.charges = hamil.mol.charges
        self.n_determinants = n_determinants
        self.env = ExponentialEnvelopes(hamil, n_determinants, isotropic=False, per_shell=False,
                                        per_orbital_exponent=False, spin_restricted=False, init_to_ones=False,
                                        softplus_zeta=False)
        self.cusp_electrons = ElectronicCuspAsymptotic(same_scale=0.25, anti_scale=0.5, alpha=1, trainable_alpha=False,
                                                       cusp_function=PsiformerCusp())

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def __call__(self, phys_conf: PhysicalConfiguration, _):
        n_elec = self.n_up + self.n_down

        # physconf.r -> [n_elec, 3(x,y,z)]
        # physconf.R -> [n_nuc, 3(x,y,z)]

        elec_nuc_diffs = pairwise_diffs(phys_conf.r, phys_conf.R)

        # elec_nuc_diffs -> [n_elec, n_nuc * 3(x,y,z)]
        elec_nuc_dist, elec_nuc_diffs = elec_nuc_diffs[:,:,-1] ** 0.5, elec_nuc_diffs[:,:,:-1]
        elec_elec_diffs = pairwise_diffs(phys_conf.r, phys_conf.r)
        elec_elec_dist = elec_elec_diffs[:,:,-1] ** 0.5 # [1, n_elec, n_elec]
        # Transform it to [n_elec, n_elec, 1]
        elec_elec_dist = elec_elec_dist.squeeze()[..., None]

        # elec_nuc_diffs -> [3 (x,y,z), n_nuc * n_elec]
        # elec_nuc_dist -> [1, n_nuc]
        # elec_elec_diffs -> [n_elec, n_elec, 4(dx,dy,dz, dist)]
        # elec_elec_dist -> [1, n_elec, n_elec]

        # print(
        #     'elec_nuc_diffs', elec_nuc_diffs.shape,
        #     'elec_nuc_dist', elec_nuc_dist.shape,
        #     'elec_elec_diffs', elec_elec_diffs.shape,
        #     'elec_elec_dist', elec_elec_dist.shape
        # )

        # INPUT FEATURES
        # ___________________________
        rescaled_diffs = elec_nuc_diffs * jnp.log(1 + elec_nuc_dist)[..., None] / elec_nuc_dist[..., None]


        a = rescaled_diffs
        b = jnp.log(1 + elec_nuc_dist)
        c = jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(self.n_down)))

        local_embeddings = jnp.concatenate((
            rescaled_diffs.reshape([n_elec, -1]),
            jnp.log(1 + elec_nuc_dist)),axis=1)


        elec_emebeddings = jnp.concatenate((
            local_embeddings,
            jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(self.n_down)))[ ..., None]),
            axis=-1)

        # output_shape : n_elec, embedding
        # ___________________________

        # GNN BLOCK
        # ___________________________
        h = hk.Linear(64)(elec_emebeddings)
        for _ in range(3):
            h = h + jnp.einsum('ijk->ik',
                               hk.nets.MLP([32, 64], activation=jax.nn.sigmoid, with_bias=False)(elec_elec_diffs))

        f = hk.Linear(self.n_determinants * n_elec)(h).reshape(n_elec, self.n_determinants, n_elec).swapaxes(0, 1)
        # ___________________________

        # GENERALIZED SLATER MATRIX
        # ___________________________
        orb = self.env(phys_conf, None)
        orb *= f
        # jax.debug.print('backflow, {y}',y=f)

        # ___________________________

        # EVALUATE DETERMINANT
        # ___________________________
        sign, xs = eval_log_slater(orb)
        xs_shift = xs.max()
        # the exp-normalize trick, to avoid over/underflow of the exponential
        xs_shift = jnp.where(~jnp.isinf(xs_shift), xs_shift, jnp.zeros_like(xs_shift))
        # replace -inf shifts, to avoid running into nans (see sloglindet)
        xs = sign * jnp.exp(xs - xs_shift)
        psi = hk.Linear(1)(xs).squeeze()
        log_psi = jnp.log(jnp.abs(psi)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi))
        # ___________________________

        # ELECTRONIC CUSP
        # ___________________________
        # jax.debug.print('elec_elec_dist {elec_elec_dist}', elec_elec_dist=elec_elec_dist)

        # concat_list = [elec_elec_dist[idxs, idxs].shape for idxs in self.spin_slices]

        # print('elec_elec_dist',concat_list)

        same_dists = jnp.concatenate(
            [triu_flat(elec_elec_dist[idxs, idxs].squeeze()) for idxs in self.spin_slices], axis=-1,
        )

        # jax.debug.print('same_dist {same_dists}', same_dists=same_dists)
        # print(same_dists.shape)

        anti_dists = flatten(elec_elec_dist[: self.n_up, self.n_up:])

        # jax.debug.print('same_dists {same_dists}', same_dists=same_dists)
        # jax.debug.print('anti_dists {anti_dists}', anti_dists=anti_dists)
        cusp = self.cusp_electrons(same_dists, anti_dists)
        # log_psi += cusp
        # grad_test = jax.grad(log_psi)
        # ___________________________
        # jax.debug.print('cusp {cusp}', cusp=cusp)
        # jax.debug.print('grad {grad}', grad=grad_test)

        return Psi(sign_psi, log_psi)


def initialize_wf(H):
    my_ansatz = instantiate_ansatz(H, MyWF)
    return my_ansatz
