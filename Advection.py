import numpy as np
from scipy.sparse import lil_matrix


def assemble_advection(elemTags, conn, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof):
    """
    Assemble global advection matrix for the axisymmetric case:

        C_ij = sum_e ∫_e N_i * (beta · grad(N_j)) * r dΩ

    with:
        x[0] = r
        x[1] = z

    Parameters
    ----------
    elemTags : array-like, shape (ne,)
        Element tags.
    conn : flattened array
        Element connectivity (ne*nloc).
    jac : flattened array
        Jacobians (ne*ngp*3*3).
    det : flattened array
        Determinants of Jacobians (ne*ngp).
    xphys : flattened array
        Physical coordinates of Gauss points (ne*ngp*3).
    w : array-like
        Quadrature weights (ngp).
    N : flattened array
        Basis function values (ngp*nloc).
    gN : flattened array
        Gradients of basis functions in reference coordinates (ngp*nloc*3).
    beta_fun : callable
        Velocity field function.
        Must return a vector-like object of size 3:
            beta_fun(x) = [b_r, b_z, 0]
    tag_to_dof : ndarray
        Mapping from gmsh node tags to compact dof indices.

    Returns
    -------
    C : lil_matrix
        Global advection matrix.
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    C = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            r = xg[0]   # coordonnee radiale

            beta_g = np.asarray(beta_fun(xg), dtype=np.float64).reshape(3,)

            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N[g, a]

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]

                    C[Ia, Ib] += wg * Na * float(np.dot(beta_g, gradNb)) * detg * r

    return C