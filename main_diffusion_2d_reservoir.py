import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from gmsh_utils import (
    gmsh_finalize,
    build_axi_reservoir_mesh,
    prepare_quadrature_and_basis,
    get_jacobians,
    border_dofs_from_tags,
    plot_full_reservoir,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d
from Neumann import assemble_rhs_neumann_outlet
from Robin import assemble_robin_wall
from Advection import assemble_advection


def main():
    parser = argparse.ArgumentParser(description="Diffusion thermique axisymetrique deux chambres")
 
    parser.add_argument("-order",    type=int,   default=1,      help="Ordre des elements finis")
    parser.add_argument("--R_res",   type=float, default=0.1,   help="Rayon des chambres [m]")
    parser.add_argument("--R_pipe",  type=float, default=0.01,   help="Rayon du tube [m]")
    parser.add_argument("--L_res",   type=float, default=0.08,   help="Longueur d une chambre [m]")
    parser.add_argument("--L_pipe",  type=float, default=0.16,   help="Longueur du tube [m]")
    parser.add_argument("--lc_res",  type=float, default=0.004,  help="Taille maille chambres [m]")
    parser.add_argument("--lc_pipe", type=float, default=0.001,  help="Taille maille tube [m]")
    parser.add_argument("--theta",   type=float, default=1.0,    help="Schema theta (1=implicite, 0.5=CN)")
    parser.add_argument("--dt",      type=float, default=0.5,    help="Pas de temps [s]")
    parser.add_argument("--nsteps",  type=int,   default=2000,    help="Nombre de pas de temps")

    args = parser.parse_args()
    dt = args.dt
    nsteps = args.nsteps

    # ------------------------------------------------------------
    # 1) Maillage du reservoir en 2D
    # ------------------------------------------------------------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        build_axi_reservoir_mesh(
            R_res=args.R_res,
            R_pipe=args.R_pipe,
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
    # 3) Quadrature et jacobiens 
    # ------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------
    # 4) Parametres physiques du NaK
    # ------------------------------------------------------------
    rho   = 860.5   # kg/m³
    cp    = 976.0   # J/(kg·K)
    k_nak = 22.4    # W/(m·K)
 
    T0    = 293.15  # K  — temperature initiale          (20°C)

    h_robin = 500.0    # W/(m²·K) — coefficient de convection externe
    T_ext   = 473.15   # K  (200°C — température extérieure)
    T_in = 298.15      # 25°C température ambiante

    U_entree = 0.001   # m/s — vitesse moyenne à l'entrée
    U_tube   = U_entree * (args.R_res / args.R_pipe)**2   # conservation de la masse
 
    # ------------------------------------------------------------
    # 5) Fonctions physiques
    #
    #    x[0] = r,  x[1] = z  partout
    # ------------------------------------------------------------
    def kappa(x):
        return k_nak    # conductivite uniforme du NaK
 
    def f_source(x):
        return 0.0      # pas de source volumique
 
    # ------------------------------------------------------------
    # 6) Assemblage des matrices axisymetriques
    #
    #    M_ij = ∫ N_i N_j r dΩ        avec r = x[0]
    #    K_ij = ∫ k ∇N_i·∇N_j r dΩ   avec r = x[0]
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
 
    M      = M_lil.tocsr()
    K      = K_lil.tocsr()
    M_phys = rho * cp * M       # ρ cp M

    # Advection de Poiseuille dans le tube uniquement
    def beta(x):
        r = x[0]
        z = x[1]
        in_tube = (args.L_res < z < args.L_res + args.L_pipe) and (r <= args.R_pipe)
        if in_tube:
            return np.array([0., 2.0 * U_tube * (1. - (r / args.R_pipe)**2), 0.])
        return np.zeros(3)

    C_lil = assemble_advection(
        elemTags=elemTags, conn=elemNodeTags,
        jac=jac, det=det, xphys=coords,
        w=w, N=N, gN=gN,
        beta_fun=beta,
        tag_to_dof=tag_to_dof,
    )
    C = rho * cp * C_lil.tocsr()
 
    # ------------------------------------------------------------
    # 7) Conditions aux limites — recuperation des dofs par bord
    # ------------------------------------------------------------
    boundary_tags  = {name: bnds_tags[i] for i, (name, _) in enumerate(bnds)}
 
    entree_dofs    = border_dofs_from_tags(boundary_tags["Entree"],    tag_to_dof)
    wall_pipe_dofs = border_dofs_from_tags(boundary_tags["Wall_pipe"], tag_to_dof)
    # Axis, Wall_res_left/right, Contraction_left/right :
    # → Neumann homogene naturel, aucun assemblage necessaire
 
    # ------------------------------------------------------------
    # 8) Condition initiale
    # ------------------------------------------------------------
    U = np.full(num_dofs, T0, dtype=float)
 
    # ------------------------------------------------------------
    # ETAPE ACTUELLE — diffusion pure, Dirichlet partout
    #
    #   Entree    → T = T0     (chambre gauche froide, 20°C)
    #   Sortie    → T = T0     (chambre droite froide, fixe pour test)
    #
    #   Resultat attendu : le champ T diffuse de la paroi chaude
    #   vers l'axe r=0, puis vers les chambres. Le regime permanent
    #   doit montrer un profil radial de type ln(r) dans le tube
    #   (solution analytique de la diffusion axisymetrique pure).
    #
    # TODO PROCHAINES ETAPES :
    #
    #   3. Advection de Poiseuille dans le tube :
    #      def beta(x):
    #          r, z = x[0], x[1]
    #          in_tube = (args.L_res < z < args.L_res + args.L_pipe
    #                     and r <= args.R_pipe)
    #          if in_tube:
    #              return np.array([0., U_max*(1-(r/args.R_pipe)**2), 0.])
    #          return np.zeros(3)
    #      C = assemble_advection(..., beta_fun=beta, ...)
    #      A = K + Rb + C
    # ------------------------------------------------------------
    # Dirichlet sur toute la face d'entrée (r=0 à R_res)
    dir_dofs = entree_dofs
    dir_vals = np.full(len(entree_dofs), T_in, dtype=float)

    # Robin sur Wall_pipe
    Rb, Fb  = assemble_robin_wall(wall_pipe_dofs, dof_coords, h_robin, T_ext)

    # Sortie : Neumann nul → rien à assembler

    A = K + Rb + C    # diffusion + Robin + advection
    F = F0 + Fb
    U[dir_dofs] = dir_vals
 
    # ------------------------------------------------------------
    # 9) Boucle en temps (schema theta)
    # ------------------------------------------------------------
    frames_U = []
 
    for step in range(args.nsteps):
        U = theta_step(
            M=M_phys, K=A,
            F_n=F, F_np1=F,
            U_n=U, dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals,
        )
        if step % 10 == 0:
            frames_U.append((step * args.dt, U.copy()))
            print(f"  step {step:4d}  t={step*args.dt:.1f}s  "
                  f"T_min={U.min()-273.15:.1f}°C  "
                  f"T_max={U.max()-273.15:.1f}°C")
 
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

    t0, Uf0 = frames_U[0]
    contour = plot_full_reservoir(
        ax=ax,
        nodeCoords=nodeCoords_saved,
        elemNodeTags=elemNodeTags_saved,
        nodeTags=nodeTags_saved,
        U=Uf0,
        tag_to_dof=tag_to_dof,
        cmap='plasma',
        vmin=T_in, vmax=T_ext,
        swap_axes=True
    )

    fig.colorbar(contour, ax=ax, label="Température [K]")

    def update(frame):
        t, Uf = frame
        ax.clear()

        # recrée le plot
        contour = plot_full_reservoir(
            ax=ax,
            nodeCoords=nodeCoords_saved,
            elemNodeTags=elemNodeTags_saved,
            nodeTags=nodeTags_saved,
            U=Uf,
            tag_to_dof=tag_to_dof,
            cmap='plasma',
            vmin=T_in, vmax=T_ext,
            swap_axes=True
        )

        ax.set_title(f"t = {t:.1f} s")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("auto")
 
    ani = FuncAnimation(fig, update, frames=frames_U, interval=80)
    ani.save("figures/simulation_diffusion_pure_axi.gif", writer=PillowWriter(fps=10))
    print("GIF sauvegarde : figures/simulation_diffusion_pure_axi.gif")
    plt.show()
 
 
if __name__ == "__main__":
    main()