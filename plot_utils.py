# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh


def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U,
    M=80, show_nodes=False, ax=None, label=None
):
    """
    Plot 1D high-order FE solution by sampling each element and evaluating gmsh basis.
    Assumes U is aligned with gmsh's compact node ordering (0..nn-1).
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    u = np.linspace(-1.0, 1.0, int(M))
    pts3 = np.zeros((len(u), 3), dtype=float)
    pts3[:, 0] = u
    uvw = pts3.reshape(-1).tolist()

    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u), nloc)

    if ax is None:
        fig, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc:(e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]

        x = coords[e, :, 0]
        uh = N @ Ue

        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


def setup_interactive_figure(xlim=None, ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax

def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, tag_to_index=None):

    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    if tag_to_index is None:
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    # take only the first 3 nodes (=geometric nodes that form the triangles)    
    all_nodes = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes = all_nodes[:, :3] 
    
    # Map to indices
    tri_indices = tag_to_index[corner_nodes.astype(int)]
    # ---------------------------------------

    mesh_triang = tri.Triangulation(x, y, tri_indices)
    # Ratio L/R pour avoir les bonnes proportions
    L_approx = np.max(coords[:, 1])
    R_approx = np.max(coords[:, 0])
    ratio = L_approx / R_approx
    fig_width = 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * ratio * 0.5))
    
    # Plot the skeleton
    ax.triplot(mesh_triang, color='black', lw=0.5, alpha=0.4)

    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]
    for i, (name, dim) in enumerate(bnds):
        tags = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]
        ax.scatter(x[indices], y[indices], label=name, s=15, zorder=3, 
                   marker="o", facecolor="None", edgecolor=colors[i % len(colors)])

    ax.set_aspect('auto')
    ax.set_xlim(np.min(coords[:, 0]) - R_approx*0.1, np.max(coords[:, 0]) + R_approx*0.1)
    ax.set_ylim(np.min(coords[:, 1]) - L_approx*0.02, np.max(coords[:, 1]) + L_approx*0.02)
    ax.legend(frameon=True, framealpha=1, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.axis(False)
    plt.savefig("figures/mesh.png")
    plt.show()


def plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof, show_mesh=False, ax=None, label=None, vmin_val = None, vmax_val = None, swap_axes=False):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Map coordinates to our compact DoF indices
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = nodeCoords.reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # 2. Determine nodes per element dynamically
    total_nodes_in_elems = len(elemNodeTags)
    # Standard Lagrange triangle node counts: Order 1=3, Order 2=6, Order 3=10, Order 4=15
    for possible_n in [3, 6, 10, 15, 21]:
        if total_nodes_in_elems % possible_n == 0:
            nodes_per_elem = possible_n
            break
    # 3. Reshape and extract ONLY the 3 corner nodes for Matplotlib
    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    # Map the GMSH tags to our 0...N-1 indices
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    # 4. Plotting
    U = np.array(U).flatten()

    vmin = vmin_val if vmin_val is not None else float(U.min())
    vmax = vmax_val if vmax_val is not None else float(U.max())

    xa, ya = (y, x) if swap_axes else (x, y)
    # adaptée avec v_min et v_max pour que ça soit cohérent avec nos valeurs
    xlabel = "z [m]" if swap_axes else "r [m]"
    ylabel = "r [m]" if swap_axes else "z [m]"

    contour = ax.tricontourf(xa, ya, triangles, U, levels=100, cmap='hot_r',
                             vmin=vmin, vmax=vmax)

    if show_mesh:
        ax.triplot(xa, ya, triangles, color='white', linewidth=0.2, alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return contour


def plot_advection_field_uniform(beta_fun, R, L, ax=None, nr=6, nz=16, arrow_len=0.22):
    """
    Champ d'advection sur une grille régulière dans tout le domaine.
    Les flèches sont uniformes, parallèles si beta est constant,
    et plus longues pour ressembler à des barres avec une pointe.

    nr : nombre de colonnes
    nz : nombre de lignes
    arrow_len : longueur visuelle des flèches (fraction de L)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Grille régulière sur tout le domaine
    r_vals = np.linspace(0.10 * R, 0.90 * R, nr)
    z_vals = np.linspace(0.08 * L, 0.92 * L, nz)

    RR, ZZ = np.meshgrid(r_vals, z_vals)

    Ur = np.zeros_like(RR, dtype=float)
    Uz = np.zeros_like(ZZ, dtype=float)

    for i in range(RR.shape[0]):
        for j in range(RR.shape[1]):
            x = np.array([RR[i, j], ZZ[i, j], 0.0], dtype=float)
            b = beta_fun(x)

            # beta peut être de taille 2 ou 3
            Ur[i, j] = b[0]
            Uz[i, j] = b[1]

    # normalisation : toutes les flèches ont la même longueur visuelle
    norm = np.sqrt(Ur**2 + Uz**2)
    norm[norm == 0.0] = 1.0

    Ur = Ur / norm
    Uz = Uz / norm

    # longueur réelle des flèches dans le repère du domaine
    # ici on choisit une longueur basée sur L
    Ur_plot = arrow_len * L * Ur
    Uz_plot = arrow_len * L * Uz

    ax.quiver(
        RR, ZZ, Ur_plot, Uz_plot,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        width=0.003,
        headwidth=4.5,
        headlength=5,
        headaxislength=5.0,
        pivot="middle",
        zorder=10
    )

    return ax