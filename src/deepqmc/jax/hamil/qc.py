from ..utils import (
    electronic_potential,
    laplacian,
    nuclear_energy,
    nuclear_potential,
)

all__ = ()


def laplacian_flat(f):
    return lambda r, **kwargs: laplacian(
        lambda r, **kwargs: f(r.reshape((-1, 3)), **kwargs).log
    )(r.flatten(), **kwargs)


class MolecularHamiltonian:
    def __init__(self, mol):
        self.mol = mol

    def local_energy(self, wf, return_grad=False):
        def loc_ene(r, graph_edges, mol=self.mol):
            lap_log_psis, quantum_force = laplacian_flat(wf)(r, graph_edges=graph_edges)
            Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=(-1)))
            Es_nuc = nuclear_energy(mol)
            Vs_nuc = nuclear_potential(r, mol)
            Vs_el = electronic_potential(r)
            Es_loc = Es_kin + Vs_nuc + Vs_el + Es_nuc
            result = (Es_loc, quantum_force) if return_grad else Es_loc
            return result

        return loc_ene
