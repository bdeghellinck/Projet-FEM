import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from gmsh_utils import (
    gmsh_finalize,
    build_two_reservoir_mesh,
    prepare_quadrature_and_basis,
    get_jacobians,
    border_dofs_from_tags,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


def main():
    parser = argparse.ArgumentParser(description="Diffusion thermique deux reservoirs")

    parser.add_argument("-order",    type=int,   default=1,      help="Ordre des elements finis")
    parser.add_argument("--H",       type=float, default=0.04,   help="Demi-hauteur des reservoirs [m]")
    parser.add_argument("--h_pipe",  type=float, default=0.01,   help="Demi-hauteur du tuyeau [m]")
    parser.add_argument("--L_res",   type=float, default=0.08,   help="Longueur d un reservoir [m]")
    parser.add_argument("--L_pipe",  type=float, default=0.16,   help="Longueur du tuyeau [m]")
    parser.add_argument("--lc_res",  type=float, default=0.004,  help="Taille maille reservoirs [m]")
    parser.add_argument("--lc_pipe", type=float, default=0.001,  help="Taille maille tuyeau [m]")
    parser.add_argument("--theta",   type=float, default=1.0,    help="Schema theta (1=implicite, 0.5=CN)")
    parser.add_argument("--dt",      type=float, default=0.5,    help="Pas de temps [s]")
    parser.add_argument("--nsteps",  type=int,   default=200,    help="Nombre de pas de temps")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1) Maillage
    # ------------------------------------------------------------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        build_two_reservoir_mesh(
            H=args.H,
            h_pipe=args.h_pipe,
            L_res=args.L_res,
            L_pipe=args.L_pipe,
            lc_res=args.lc_res,
            lc_pipe=args.lc_pipe,
            order=args.order,
        )

    plot_mesh_2d(
        elemType=elemType, nodeTags=nodeTags, nodeCoords=nodeCoords,
        elemTags=elemTags, elemNodeTags=elemNodeTags,
        bnds=bnds, bnds_tags=bnds_tags,
    )

    # ------------------------------------------------------------
    # 2) Mapping Gmsh node tag -> dof compact
    # ------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs         = len(unique_dofs_tags)
    max_tag          = int(np.max(nodeTags))
    all_coords       = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)

    tag_to_dof          = np.full(max_tag + 1, -1, dtype=int)
    node_index_from_tag = np.full(max_tag + 1, -1, dtype=int)
    for i_node, tag in enumerate(nodeTags):
        node_index_from_tag[int(tag)] = i_node

    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    for i_dof, tag in enumerate(unique_dofs_tags):
        tag_int            = int(tag)
        tag_to_dof[tag_int]  = i_dof
        dof_coords[i_dof]  = all_coords[node_index_from_tag[tag_int]]

    # ------------------------------------------------------------
    # 3) Quadrature et jacobiens  (avant gmsh_finalize !)
    # ------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------
    # 4) Parametres physiques du NaK
    # ------------------------------------------------------------
    rho   = 860.5   # kg/m³
    cp    = 976.0   # J/(kg·K)
    k_nak = 22.4    # W/(m·K)

    T0    = 293.15  # K  — temperature initiale (20°C)
    T_hot = 573.15  # K  — temperature imposee sur Wall_pipe (300°C)
                    #       utilisee uniquement pour le test Dirichlet ci-dessous

    # ------------------------------------------------------------
    # 5) Fonctions physiques
    #
    #    ATTENTION : on est en 2D cartesien (x, y), PAS axisymetrique.
    #    x[0] = x  (direction axiale)
    #    x[1] = y  (direction transverse, 0 = axe de symetrie)
    #    Il n'y a PAS de facteur r = x[0] dans les assembleurs.
    #
    #    TODO etape suivante : adapter stiffness.py et mass.py pour
    #    supprimer le facteur r (ou passer r=1 partout).
    # ------------------------------------------------------------
    def kappa(x):
        return k_nak

    def f_source(x):
        return 0.0   # pas de source volumique

    # ------------------------------------------------------------
    # 6) Assemblage des matrices
    #
    #    NOTE : assemble_mass et assemble_stiffness_and_rhs contiennent
    #    actuellement un facteur r = x[0] (code axisymetrique).
    #    Pour l'instant on laisse tourner pour verifier la structure,
    #    mais les valeurs numeriques seront fausses jusqu'a correction.
    # ------------------------------------------------------------
    M_lil = assemble_mass(
        elemTags=elemTags, conn=elemNodeTags,
        det=det, xphys=coords, w=w, N=N,
        tag_to_dof=tag_to_dof,
    )

    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags=elemTags, conn=elemNodeTags,
        jac=jac, det=det, xphys=coords,
        w=w, N=N, gN=gN,
        kappa_fun=kappa,
        rhs_fun=f_source,
        tag_to_dof=tag_to_dof,
    )

    M = M_lil.tocsr()
    K = K_lil.tocsr()
    M_phys = rho * cp * M

    # ------------------------------------------------------------
    # 7) Conditions aux limites — bords nommes
    # ------------------------------------------------------------
    boundary_tags = {name: bnds_tags[i] for i, (name, _) in enumerate(bnds)}

    entree_dofs    = border_dofs_from_tags(boundary_tags["Entree"],    tag_to_dof)
    sortie_dofs    = border_dofs_from_tags(boundary_tags["Sortie"],    tag_to_dof)
    wall_pipe_dofs = border_dofs_from_tags(boundary_tags["Wall_pipe"], tag_to_dof)

    # ------------------------------------------------------------
    # 8) Condition initiale
    # ------------------------------------------------------------
    U = np.full(num_dofs, T0, dtype=float)

    # ------------------------------------------------------------
    # ETAPE ACTUELLE : test diffusion pure avec Dirichlet partout
    #
    #   - Wall_pipe : T = T_hot  (paroi chaude)
    #   - Entree    : T = T0     (reservoir froid, fixe)
    #   - Sortie    : T = T0     (reservoir chaud, fixe pour test)
    #
    # Cela permet de verifier que K et M sont corrects en comparant
    # au regime permanent analytique (simple gradient lineaire en x).
    #
    # PROCHAINES ETAPES (dans l'ordre) :
    #   1. Corriger stiffness.py et mass.py : supprimer le facteur r
    #      → remplacer "* r" par "* 1.0" ou reorganiser sans axisymetrie
    #   2. Remplacer Dirichlet sur Wall_pipe par Robin :
    #      adapter Robin.py pour 2D cartesien (supprimer 2*pi*R)
    #   3. Remplacer Dirichlet sur Entree par Neumann entrant q_in :
    #      adapter Neumann.py pour 2D cartesien
    #   4. Ajouter l'advection de Poiseuille dans le tuyeau :
    #      beta(x) = U_max*(1 - (x[1]/h_pipe)**2) si L_res < x[0] < L_res+L_pipe
    #               et x[1] <= h_pipe, sinon 0
    # ------------------------------------------------------------
    dir_dofs = np.concatenate([entree_dofs, sortie_dofs, wall_pipe_dofs])
    dir_vals = np.concatenate([
        np.full(len(entree_dofs),    T0,    dtype=float),
        np.full(len(sortie_dofs),    T0,    dtype=float),
        np.full(len(wall_pipe_dofs), T_hot, dtype=float),
    ])

    U[dir_dofs] = dir_vals

    # second membre : pas de Neumann ni Robin pour l'instant
    F = F0.copy()

    # ------------------------------------------------------------
    # 9) Boucle en temps (schema theta)
    # ------------------------------------------------------------
    frames_U = []

    for step in range(args.nsteps):
        U = theta_step(
            M=M_phys, K=K,
            F_n=F, F_np1=F,
            U_n=U, dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals,
        )
        if step % 10 == 0:
            frames_U.append((step * args.dt, U.copy()))
            print(f"  step {step:4d}  t={step*args.dt:.1f}s  "
                  f"T_min={U.min()-273.15:.1f}°C  T_max={U.max()-273.15:.1f}°C")

    # ------------------------------------------------------------
    # 10) Sauvegarde des arrays gmsh AVANT gmsh_finalize
    # ------------------------------------------------------------
    nodeTags_saved     = np.array(nodeTags,     dtype=np.int64).copy()
    nodeCoords_saved   = np.array(nodeCoords,   dtype=float).copy()
    elemNodeTags_saved = np.array(elemNodeTags, dtype=np.int64).copy()

    gmsh_finalize()

    # ------------------------------------------------------------
    # 11) Animation du champ de temperature
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    _, Uf0 = frames_U[0]
    contour0 = plot_fe_solution_2d(
        elemNodeTags=elemNodeTags_saved, nodeCoords=nodeCoords_saved,
        nodeTags=nodeTags_saved, U=Uf0, tag_to_dof=tag_to_dof,
        show_mesh=False, ax=ax, vmin_val=T0, vmax_val=T_hot,
    )
    cbar = fig.colorbar(contour0, ax=ax, label="Température [K]")
    cbar.set_ticks([T0, (T0 + T_hot) / 2, T_hot])
    cbar.set_ticklabels([
        f"{T0   - 273.15:.0f}°C",
        f"{(T0 + T_hot)/2 - 273.15:.0f}°C",
        f"{T_hot - 273.15:.0f}°C",
    ])

    def update(frame):
        t, Uf = frame
        ax.clear()
        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags_saved, nodeCoords=nodeCoords_saved,
            nodeTags=nodeTags_saved, U=Uf, tag_to_dof=tag_to_dof,
            show_mesh=False, ax=ax, vmin_val=T0, vmax_val=T_hot,
        )
        ax.set_title(f"Diffusion pure — t = {t:.1f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    ani = FuncAnimation(fig, update, frames=frames_U, interval=80)
    ani.save("figures/simulation_diffusion.gif", writer=PillowWriter(fps=10))
    print("GIF sauvegarde : figures/simulation_diffusion.gif")
    plt.show()


if __name__ == "__main__":
    main()