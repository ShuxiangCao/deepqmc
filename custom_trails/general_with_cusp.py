import importlib
import os
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

from deepqmc.wf.cusp import PsiformerCusp, ElectronicCuspAsymptotic

def dynamic_load(python_file,object_name):
    # Validate the Python file path
    if not os.path.isfile(python_file):
        raise RuntimeError(f"Error: The specified file '{python_file}' does not exist.")

    # Dynamically load the Python file
    module_name = os.path.splitext(os.path.basename(python_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, python_file)
    if spec is None:
        raise RuntimeError(f"Error: Unable to create a module spec for '{python_file}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Access the `initialize_wf` instance
    if not hasattr(module, object_name):
        raise RuntimeError(f"Error: The file '{python_file}' does not contain an '{object_name}' instance.")

    return getattr(module, object_name)


class MyWF(hk.Module):
    def __init__(
            self,
            hamil,
            n_determinants=16,
            **kwargs
    ):
        self.kwargs = kwargs
        super().__init__()
        self.mol = hamil.mol
        self.n_up, self.n_down = hamil.n_up, hamil.n_down
        self.charges = hamil.mol.charges
        self.n_determinants = n_determinants
        self.env = ExponentialEnvelopes(hamil, n_determinants, isotropic=False, per_shell=False,
                                        per_orbital_exponent=False, spin_restricted=False, init_to_ones=False,
                                        softplus_zeta=False)

        self.cusp_electrons = ElectronicCuspAsymptotic(same_scale=0.25, anti_scale=0.5, alpha=1,
                             trainable_alpha=True, cusp_function=PsiformerCusp())

        kwargs.update({
            'n_up': self.n_up,
            'n_down': self.n_down,
            'n_determinants': self.n_determinants,
            'env': self.env,
            'mol': self.mol,
            'charges': self.charges
        })

        assert 'features' in kwargs, "Please provide the features for the model"
        assert 'model' in kwargs, "Please provide the model name for the model"
        self.features_func = dynamic_load(f"./features/{kwargs['features']}.py",'features')(
            **kwargs)
        self.model_func = dynamic_load(f"./models/{kwargs['model']}.py",'model')(**kwargs)

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def __call__(self, phys_conf: PhysicalConfiguration, _):

        n_elec = self.n_up + self.n_down
        elec_elec_diffs = pairwise_diffs(phys_conf.r, phys_conf.r)
        elec_elec_dist = elec_elec_diffs[..., -1]

        h = self.features_func(phys_conf)
        h = self.model_func(h)

        f = hk.Linear(self.n_determinants * n_elec)(h).reshape(n_elec, self.n_determinants, n_elec).swapaxes(0, 1)
        # ___________________________

        # GENERALIZED SLATER MATRIX
        # ___________________________
        orb = self.env(phys_conf, None)
        orb *= f

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
        same_dists = jnp.concatenate(
            [triu_flat(elec_elec_dist[idxs, idxs]) for idxs in
             self.spin_slices],
            axis=-1,
        )
        anti_dists = flatten(elec_elec_dist[: self.n_up, self.n_up:])
        log_psi += self.cusp_electrons(same_dists, anti_dists)
        # ___________________________

        return Psi(sign_psi, log_psi)


def initialize_wf(H, **kwargs):
    warpped_wf = partial(MyWF, **kwargs)
    my_ansatz = instantiate_ansatz(H, warpped_wf)
    return my_ansatz
