from deepqmc.wf import NeuralNetworkWaveFunction, env, nn_wave_function, cusp, omni
from deepqmc.hkext import SumPool, MLP, Identity, ResidualConnection
from deepqmc.gnn import ElectronGNN, edge_features, electron_gnn
import jax.numpy as jnp
import os

import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import deepqmc
from deepqmc.app import instantiate_ansatz


def initialize_wf(H):
    # Envelope configuration
    envelope = env.ExponentialEnvelopes(
        H,
        n_determinants=1,
        isotropic=True,
        per_shell=False,
        per_orbital_exponent=True,
        spin_restricted=False,
        init_to_ones=True,
        softplus_zeta=False,
    )

    # Backflow operator configuration
    backflow_op = nn_wave_function.BackflowOp(
        mult_act=lambda x: x,
    )

    # Cusp electrons configuration
    cusp_function = cusp.PsiformerCusp()
    cusp_electrons = cusp.ElectronicCuspAsymptotic(
        same_scale=0.25,
        anti_scale=0.5,
        alpha=1.0,
        trainable_alpha=True,
        cusp_function=cusp_function,
    )

    # SumPool configuration
    conf_coeff = SumPool()

    # Backflow factory configuration
    backflow_factory = omni.Backflow(
        subnet_factory=MLP(
            hidden_layers=['log', 1],
            bias=False,
            last_linear=True,
            activation=None,
            init='ferminet',
        )
    )

    # Electron embedding configuration
    electron_embedding = electron_gnn.ElectronEmbedding(
        positional_embeddings={
            'ne': edge_features.CombinedEdgeFeature(
                features=[
                    edge_features.DistancePowerEdgeFeature(
                        powers=[1],
                        log_rescale=True,
                    ),
                    edge_features.DifferenceEdgeFeature(
                        log_rescale=True,
                    ),
                ]
            )
        },
        use_spin=True,
        project_to_embedding_dim=True,
    )

    # GNN layer factory configuration
    layer_factory = electron_gnn.ElectronGNNLayer(
        subnet_factory=Identity(),
        electron_residual=False,
        nucleus_residual=False,
        two_particle_residual=False,
        deep_features=False,
        update_rule='concatenate',
        update_features=[
            electron_gnn.update_features.NodeAttentionElectronUpdateFeature(
                num_heads=4,
                mlp_factory=MLP(
                    hidden_layers=['log', 2],
                    bias=True,
                    last_linear=False,
                    activation=jnp.tanh,
                    init='ferminet',
                ),
                attention_residual=ResidualConnection(normalize=False),
                mlp_residual=ResidualConnection(normalize=False),
            )
        ],
    )

    # GNN factory configuration
    gnn_factory = ElectronGNN(
        n_interactions=4,
        nuclei_embedding=None,
        electron_embedding=electron_embedding,
        two_particle_stream_dim=32,
        self_interaction=True,
        edge_features=None,
        layer_factory=layer_factory,
    )

    # OmniNet factory configuration
    omni_factory = omni.OmniNet(
        embedding_dim=256,
        jastrow_factory=None,
        backflow_factory=backflow_factory,
        nuclear_gnn_head=None,
        gnn_factory=gnn_factory,
    )

    # Wave function configuration
    wave_function = NeuralNetworkWaveFunction(
        envelope=envelope,
        backflow_op=backflow_op,
        n_determinants=16,
        full_determinant=True,
        cusp_electrons=cusp_electrons,
        cusp_nuclei=False,
        backflow_transform="mult",
        conf_coeff=conf_coeff,
        omni_factory=omni_factory,
    )

    _ansatz = instantiate(wave_function, _recursive_=True, _convert_='all')

    psiformer_ansatz = instantiate_ansatz(H, _ansatz)
    return psiformer_ansatz