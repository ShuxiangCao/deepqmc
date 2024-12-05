import os

import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import deepqmc
from deepqmc.app import instantiate_ansatz


def initialize_wf(H, config_name='psiformer'):
    deepqmc_dir = os.path.dirname(deepqmc.__file__)
    config_dir = os.path.join(deepqmc_dir, 'conf/ansatz')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    _ansatz = instantiate(cfg, _recursive_=True, _convert_='all')

    psiformer_ansatz = instantiate_ansatz(H, _ansatz)
    return psiformer_ansatz