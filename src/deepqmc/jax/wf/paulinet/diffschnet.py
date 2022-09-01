import haiku as hk
import jax.numpy as jnp
from jax import ops

from ...hkext import MLP
from .distbasis import DistanceBasis
from .graph import (
    Graph,
    GraphNodes,
    MessagePassingLayer,
    MolecularGraphEdgeBuilder,
    distance_difference_callback,
)


class DiffSchNetLayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        n_up,
        n_down,
        embedding_dim,
        kernel_dim,
        dist_feat_dim,
        distance_basis,
        shared_g=False,
        w_subnet=None,
        h_subnet=None,
        g_subnet=None,
        *,
        n_layers_w=2,
        n_layers_h=1,
        n_layers_g=1,
    ):
        super().__init__('DiffSchNetLayer', ilayer)

        def default_subnet_kwargs(n_layers):
            return {
                'hidden_layers': ('log', n_layers),
                'last_bias': False,
                'last_linear': True,
            }

        labels = ['same', 'anti', 'ne']
        self.w = {
            lbl: MLP(
                dist_feat_dim,
                kernel_dim,
                name=f'w_{lbl}',
                **(w_subnet or default_subnet_kwargs(n_layers_w)),
            )
            for lbl in labels
        }
        self.spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.h = {
            lbl: hk.Embed(1 if n_up == n_down else 2, kernel_dim, name=f'h_{lbl}')
            if self.ilayer == 0
            else MLP(
                embedding_dim,
                kernel_dim,
                name=f'h_{lbl}',
                **(h_subnet or default_subnet_kwargs(n_layers_h)),
            )
            for lbl in labels
        }
        self.g = (
            MLP(
                kernel_dim,
                embedding_dim,
                name='g',
                **(g_subnet or default_subnet_kwargs(n_layers_g)),
            )
            if shared_g
            else {
                lbl: MLP(
                    kernel_dim,
                    embedding_dim,
                    name=f'g_{lbl}',
                    **(g_subnet or default_subnet_kwargs(n_layers_g)),
                )
                for lbl in labels
            }
        )
        self.distance_basis = distance_basis
        self.labels = labels
        self.shared_g = shared_g

    def expand_diffs(self, dists, diffs):
        diffs_expanded = []
        for diff in diffs.T:
            diff_pos = jnp.abs(diff) * (diff > 0)
            diff_neg = jnp.abs(diff) * (diff < 0)
            diffs_expanded.append(self.distance_basis(diff_pos))
            diffs_expanded.append(self.distance_basis(diff_neg))
        diffs_expanded.append(self.distance_basis(diff))
        return jnp.concatenate(diffs_expanded, axis=-1)

    def get_update_edges_fn(self):
        def update_edges_fn(nodes, edges):
            return {
                k: edge._replace(
                    data={
                        'diffs': self.expand_diffs(
                            edge.data['distances'], edge.data['differences']
                        )
                    }
                )
                for k, edge in edges.items()
            }

        return update_edges_fn if self.ilayer == 0 else None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons.shape[-2]
            we_same, we_anti, we_n = (
                self.w[lbl](edges[lbl].data['diffs']) for lbl in self.labels
            )
            hx_same, hx_anti = (
                self.h[lbl](self.spin_idxs if self.ilayer == 0 else nodes.electrons)[
                    edges[lbl].senders
                ]
                for lbl in self.labels[:2]
            )
            weh_same = we_same * hx_same
            weh_anti = we_anti * hx_anti
            weh_n = we_n * nodes.nuclei[edges['ne'].senders]
            z_same = ops.segment_sum(
                data=weh_same, segment_ids=edges['same'].receivers, num_segments=n_elec
            )
            z_anti = ops.segment_sum(
                data=weh_anti, segment_ids=edges['anti'].receivers, num_segments=n_elec
            )
            z_n = ops.segment_sum(
                data=weh_n, segment_ids=edges['ne'].receivers, num_segments=n_elec
            )
            return {
                'same': z_same,
                'anti': z_anti,
                'ne': z_n,
            }

        return aggregate_edges_for_nodes_fn

    def get_update_nodes_fn(self):
        def update_nodes_fn(nodes, z):
            updated_nodes = nodes._replace(
                electrons=nodes.electrons
                + (
                    (self.g if self.shared_g else self.g['ne'])(z['ne'])
                    + (self.g if self.shared_g else self.g['same'])(z['same'])
                    + (self.g if self.shared_g else self.g['anti'])(z['anti'])
                )
            )
            return updated_nodes

        return update_nodes_fn


class DiffSchNet(hk.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        coords,
        embedding_dim,
        dist_feat_dim=32,
        kernel_dim=128,
        n_interactions=3,
        cutoff=10.0,
        layer_kwargs=None,
    ):
        super().__init__('DiffSchNet')
        labels = ['same', 'anti', 'ne']
        self.coords = coords
        self.edge_factory = MolecularGraphEdgeBuilder(
            n_nuc,
            n_up,
            n_down,
            coords,
            labels,
            kwargs_by_edge_type={
                lbl: {'cutoff': cutoff, 'data_callback': distance_difference_callback}
                for lbl in labels
            },
        )
        self.spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.X = hk.Embed(
            1 if n_up == n_down else 2, embedding_dim, name='ElectronicEmbedding'
        )
        self.init_nuc = jnp.eye(n_nuc)
        self.Y = hk.Linear(kernel_dim, with_bias=False, name='NuclearEmbedding')
        self.layers = [
            DiffSchNetLayer(
                i,
                n_up,
                n_down,
                embedding_dim,
                kernel_dim,
                7 * dist_feat_dim,
                DistanceBasis(dist_feat_dim, cutoff, envelope='nocusp')
                if i == 0
                else None,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    def __call__(self, rs):
        def init_state(shape, dtype):
            zeros = jnp.zeros(shape, dtype)
            return {'anti': (zeros, zeros), 'ne': zeros, 'same': (zeros, zeros)}

        occupancies = hk.get_state(
            'occupancies',
            shape=1,
            dtype=jnp.int32,
            init=init_state,
        )
        n_occupancies = hk.get_state(
            'n_occupancies', shape=[], dtype=jnp.int32, init=jnp.zeros
        )
        nuc_embedding = self.Y(self.init_nuc)
        elec_embedding = self.X(self.spin_idxs)
        graph_edges, occupancies, n_occupancies = self.edge_factory(
            rs, occupancies, n_occupancies
        )
        hk.set_state('occupancies', occupancies)
        hk.set_state('n_occupancies', n_occupancies)
        graph = Graph(
            GraphNodes(nuc_embedding, elec_embedding),
            graph_edges,
        )
        for layer in self.layers:
            graph = layer(graph)
        return graph.nodes.electrons