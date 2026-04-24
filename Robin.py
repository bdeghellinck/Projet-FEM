import numpy as np
from scipy.sparse import lil_matrix


def assemble_robin_wall(wall_dofs, dof_coords, h_robin, u_ext, R=None):
    """
    Assemblage Robin sur un bord 1D en coordonnées axisymétriques.

    Condition :  -k grad(T)·n = h * (T - T_ext)

    Forme faible — contribution à la matrice et au second membre :
        Rb_ij += ∫_Gamma h * N_i * N_j * r ds
        Fb_i  += ∫_Gamma h * T_ext * N_i * r ds

    Intégration exacte P1 sur chaque segment [x0, x1] avec r(ξ) = r0 + ξ(r1-r0) :

        Matrice de masse pondérée par r :
            M_r = ds * ∫_0^1 [N_i N_j] r(ξ) dξ
                = ds/12 * [[3r0+r1,  r0+r1 ],
                            [r0+r1,  r0+3r1]]

        Vecteur de masse pondérée par r (= première colonne de M_r + deuxième) :
            F_i = T_ext * h * 2π * ds * [(2r0+r1)/6, (r0+2r1)/6]

    Note : le facteur 2π de la révolution est inclus.
    Le paramètre R est conservé pour compatibilité API.

    Parameters
    ----------
    wall_dofs : array-like (int)
        DOFs (indices compacts) de la paroi Robin.
    dof_coords : ndarray, shape (num_dofs, 3)
        Coordonnées physiques de tous les DOFs : col 0 = r, col 1 = z.
    h_robin : float
        Coefficient de transfert thermique [W/(m²·K)].
    u_ext : float
        Température extérieure [K].
    R : float, optional
        Non utilisé (conservé pour compatibilité).

    Returns
    -------
    Rb : csr_matrix, shape (num_dofs, num_dofs)
        Contribution Robin à la matrice de rigidité.
    Fb : ndarray, shape (num_dofs,)
        Contribution Robin au second membre.
    """
    ndofs = len(dof_coords)

    Rb = lil_matrix((ndofs, ndofs), dtype=float)
    Fb = np.zeros(ndofs, dtype=float)

    wall_dofs = np.array(wall_dofs, dtype=int)

    # Trier selon z (paroi verticale r=R_pipe) ou r (paroi horizontale)
    r_wall = dof_coords[wall_dofs, 0]
    z_wall = dof_coords[wall_dofs, 1]
    if np.ptp(z_wall) > np.ptp(r_wall):
        order = np.argsort(z_wall)
    else:
        order = np.argsort(r_wall)
    wall_sorted = wall_dofs[order]

    for i in range(len(wall_sorted) - 1):
        i0 = wall_sorted[i]
        i1 = wall_sorted[i + 1]

        x0 = dof_coords[i0]
        x1 = dof_coords[i1]

        r0 = x0[0]
        r1 = x1[0]

        # Longueur du segment
        ds = np.linalg.norm(x1[:2] - x0[:2])

        # ----------------------------------------------------------
        # Matrice locale : Rb_ij = h * 2π * ∫_0^1 N_i N_j r(ξ) dξ * ds
        #
        #   ∫_0^1 N0² r dξ = ∫_0^1 (1-ξ)² (r0+ξΔr) dξ = (3r0+r1)/12
        #   ∫_0^1 N0 N1 r dξ = ∫_0^1 (1-ξ)ξ  (r0+ξΔr) dξ = (r0+r1)/12
        #   ∫_0^1 N1² r dξ = ∫_0^1  ξ²   (r0+ξΔr) dξ = (r0+3r1)/12
        # ----------------------------------------------------------
        coef = h_robin * 2.0 * np.pi * ds / 12.0

        Mloc = coef * np.array(
            [[3.0 * r0 + r1,       r0 +       r1],
             [      r0 + r1, r0 + 3.0 * r1      ]],
            dtype=float
        )

        # ----------------------------------------------------------
        # Vecteur local : Fb_i = h * T_ext * 2π * ∫_0^1 N_i r(ξ) dξ * ds
        #
        #   ∫_0^1 N0 r dξ = (2r0+r1)/6
        #   ∫_0^1 N1 r dξ = (r0+2r1)/6
        # ----------------------------------------------------------
        Floc = h_robin * u_ext * 2.0 * np.pi * ds * np.array(
            [(2.0 * r0 + r1) / 6.0,
             (r0 + 2.0 * r1) / 6.0],
            dtype=float
        )

        dofs = [i0, i1]
        for a in range(2):
            Fb[dofs[a]] += Floc[a]
            for b in range(2):
                Rb[dofs[a], dofs[b]] += Mloc[a, b]

    return Rb.tocsr(), Fb