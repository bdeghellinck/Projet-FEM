import numpy as np
from scipy.sparse import lil_matrix




def assemble_robin_wall(wall_dofs, dof_coords, h_robin, u_ext, R):
    """
    Assemblage Robin sur la paroi externe r = R, pour des éléments P1.
    
    Condition :
        -k grad(u).n = h_robin * (u - u_ext)

    Contribution faible :
        + ∫_Gamma h_robin * u * v * (2*pi*r) ds
        + ∫_Gamma h_robin * u_ext * v * (2*pi*r) ds

    Hypothèse :
        wall_dofs contient les ddl de la paroi r=R
        et on les trie selon z.
    """
    ndofs = len(dof_coords)

    Rb = lil_matrix((ndofs, ndofs), dtype=float)
    Fb = np.zeros(ndofs, dtype=float)

    # Trier les nœuds de paroi selon z
    wall_dofs = np.array(wall_dofs, dtype=int)
    z_wall = dof_coords[wall_dofs, 1]
    order = np.argsort(z_wall)
    wall_sorted = wall_dofs[order]

    # Assemblage segment par segment
    for i in range(len(wall_sorted) - 1):
        i0 = wall_sorted[i]
        i1 = wall_sorted[i + 1]

        x0 = dof_coords[i0]
        x1 = dof_coords[i1]

        # Longueur du segment dans le plan (r,z)
        ds = np.linalg.norm(x1[:2] - x0[:2])

        # matrice de masse 1D P1 : ds/6 [[2,1],[1,2]]
        # facteur axisymétrique : 2*pi*R
        coef = h_robin * 2.0 * np.pi * R * ds / 6.0

        Mloc = coef * np.array([[2.0, 1.0],
                                [1.0, 2.0]], dtype=float)

        # vecteur : ∫ h*u_ext*N_i*(2*pi*R) ds
        Floc = h_robin * u_ext * 2.0 * np.pi * R * ds * 0.5 * np.array([1.0, 1.0])

        dofs = [i0, i1]

        for a in range(2):
            Fb[dofs[a]] += Floc[a]
            for b in range(2):
                Rb[dofs[a], dofs[b]] += Mloc[a, b]

    return Rb.tocsr(), Fb