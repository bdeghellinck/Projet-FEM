# mass.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_mass(elemTags, conn, det, xphys, w, N, tag_to_dof):
    """
    Assemble global mass matrix for axisymmetric coordinates:
        M_ij = sum_e ∫_e N_i N_j r dΩ
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)

    M = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            r = xphys[e, g, 0]   # x[0] = r

            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N[g, a]

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    Nb = N[g, b]

                    M[Ia, Ib] += wg * Na * Nb * detg * r

    return M
