import numpy as np
from scipy.sparse import lil_matrix

def assemble_rhs_neumann_outlet(outlet_dofs, dof_coords, g_neu_fun, R):
    """
    Assemble le second membre de Neumann sur la sortie.
    
    Condition :
        -k grad(u).n = g_neu

    Hypothèse :
        la sortie est le segment z = L, avec des nœuds triés selon r.
    """
    Fneu = np.zeros(len(dof_coords), dtype=float)

    outlet_dofs = np.array(outlet_dofs, dtype=int)
    r_out = dof_coords[outlet_dofs, 0]
    order = np.argsort(r_out)
    outlet_sorted = outlet_dofs[order]

    for i in range(len(outlet_sorted) - 1):
        i0 = outlet_sorted[i]
        i1 = outlet_sorted[i + 1]

        x0 = dof_coords[i0]
        x1 = dof_coords[i1]

        ds = np.linalg.norm(x1[:2] - x0[:2])

        xm = 0.5 * (x0 + x1)
        gmid = float(g_neu_fun(xm))

        rmid = xm[0]

        # ∫ g N_i (2*pi*r) ds
        Floc = gmid * 2.0 * np.pi * rmid * ds * 0.5 * np.array([1.0, 1.0])

        Fneu[i0] += Floc[0]
        Fneu[i1] += Floc[1]

    return Fneu
