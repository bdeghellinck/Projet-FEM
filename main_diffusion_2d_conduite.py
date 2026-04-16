import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.special import j0, j1
from scipy.optimize import brentq

from gmsh_utils import (
    gmsh_finalize,
    build_conduit_mesh,
    prepare_quadrature_and_basis,
    get_jacobians,
    border_dofs_from_tags,
)
from stiffness import assemble_stiffness_and_rhs
from Neumann import assemble_rhs_neumann_outlet
from Robin import assemble_robin_wall
from mass import assemble_mass
from dirichlet import theta_step
from Advection import assemble_advection
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d,plot_advection_field_uniform


def main():
    parser = argparse.ArgumentParser(
        description="Advection-diffusion axisymetrique (r,z) dans une conduite avec Dirichlet partout"
    )

    parser.add_argument("-order", type=int, default=1, help="Ordre des elements finis")
    parser.add_argument("--R", type=float, default=0.01, help="Rayon de la conduite")
    parser.add_argument("--L", type=float, default=0.2, help="Longueur de la conduite")
    parser.add_argument("--lc", type=float, default=0.002, help="Taille caracteristique du maillage")

    parser.add_argument("--theta", type=float, default=1.0,
                        help="1: Euler implicite, 0.5: Crank-Nicolson, 0: explicite")
    parser.add_argument("--dt", type=float, default=1.0, help="Pas de temps")
    parser.add_argument("--nsteps", type=int, default=500, help="Nombre de pas de temps")

    args = parser.parse_args()

    dt = args.dt
    nsteps = args.nsteps

    # ------------------------------------------------------------
    # 1) Maillage de la conduite axisymetrique
    # ------------------------------------------------------------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = build_conduit_mesh(
        R=args.R,
        L=args.L,
        lc=args.lc,
        order=args.order
    )

    plot_mesh_2d(
        elemType=elemType,
        nodeTags=nodeTags,
        nodeCoords=nodeCoords,
        elemTags=elemTags,
        elemNodeTags=elemNodeTags,
        bnds=bnds,
        bnds_tags=bnds_tags
    )

    # ------------------------------------------------------------
    # 2) Mapping Gmsh node tag -> dof compact
    # ------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    all_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)

    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)

    node_index_from_tag = np.full(max_tag + 1, -1, dtype=int)
    for i_node, tag in enumerate(nodeTags):
        node_index_from_tag[int(tag)] = i_node

    for i_dof, tag in enumerate(unique_dofs_tags):
        tag_int = int(tag)
        tag_to_dof[tag_int] = i_dof
        dof_coords[i_dof] = all_coords[node_index_from_tag[tag_int]]

    # ------------------------------------------------------------
    # 3) Quadrature et jacobiens
    # ------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------
    # 4) Donnees physiques
    #    Convention : x[0] = r, x[1] = z
    # ------------------------------------------------------------

    # --- Paramètres physiques du NaK ---
    rho   = 860.5    # kg/m³
    cp    = 976.0    # J/(kg·K)
    k_nak = 22.4     # W/(m·K)
    # alpha = k/(rho*cp) = 2.67e-5 m²/s  (pour vérification)

    # --- Conditions aux limites ---
    T0    = 293.15   # K  (20°C, condition initiale)
    T_in  = 473.15   # K  (200°C, température à l'entrée — à choisir)
    T_ext = 353.15   # K  (80°C, température extérieure pour Robin)

    h_robin = 500    # on suppose qu'il y a de l'air autour du tuyeau

    def kappa(x):
        return k_nak 

    def beta(x):
        r = x[0]
        z = x[1]
        # transport de l'entree vers la sortie, donc selon +z
        return np.array([0.0, 1, 0.0], dtype=float)

    def f_source(x, t):
        return 0  # car pas de source volumique


    def u0(x):
        return T0  # la condition initiale uniforme

    def uD(x, t):
        return T_in  # Dirichlet à l'entrée
    
    def g_sortie(x, t):
        return 0.0   # Neumann homogène à la sortie (∂T/∂z = 0)

    # ------------------------------------------------------------
    # 5) Assemblage des matrices axisymetriques
    # ------------------------------------------------------------
    M_lil = assemble_mass(
        elemTags=elemTags,
        conn=elemNodeTags,
        det=det,
        xphys=coords,
        w=w,
        N=N,
        tag_to_dof=tag_to_dof
    )

    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags=elemTags,
        conn=elemNodeTags,
        jac=jac,
        det=det,
        xphys=coords,
        w=w,
        N=N,
        gN=gN,
        kappa_fun=kappa,
        rhs_fun=lambda x: f_source(x, 0.0),
        tag_to_dof=tag_to_dof
    )

    C_lil = assemble_advection(
        elemTags=elemTags,
        conn=elemNodeTags,
        jac=jac,
        det=det,
        xphys=coords,
        w=w,
        N=N,
        gN=gN,
        beta_fun=beta,
        tag_to_dof=tag_to_dof
    )

    M = M_lil.tocsr()
    K = K_lil.tocsr()
    C = C_lil.tocsr()

    # ------------------------------------------------------------
    # 6) Condition initiale
    # ------------------------------------------------------------
    U = np.array([u0(x) for x in dof_coords], dtype=float)

    # ------------------------------------------------------------
    # 7) Conditions sur les bords
    # ------------------------------------------------------------
    boundary_tags = {}
    for i, (name, dim) in enumerate(bnds):
        boundary_tags[name] = bnds_tags[i]

    Entree_tags = boundary_tags["Entree"]
    Sortie_tags = boundary_tags["Sortie"]
    wall_tags = boundary_tags["Wall"]

    dir_dofs = border_dofs_from_tags(Entree_tags, tag_to_dof)
    outlet_dofs = border_dofs_from_tags(Sortie_tags, tag_to_dof)
    wall_dofs = border_dofs_from_tags(wall_tags, tag_to_dof)

    dir_vals_t0 = np.array([uD(dof_coords[d], 0.0) for d in dir_dofs], dtype=float)
    U[dir_dofs] = dir_vals_t0

    # ------------------------------------------------------------
    # 7bis) Assemblage Robin sur le bord wall
    # ------------------------------------------------------------
    Rb, Fb_robin = assemble_robin_wall(
        wall_dofs=wall_dofs,
        dof_coords=dof_coords,
        h_robin=h_robin,
        u_ext=T_ext,
        R=args.R
    )
    F_neu = assemble_rhs_neumann_outlet(
        outlet_dofs=outlet_dofs,
        dof_coords=dof_coords,
        g_neu_fun=lambda x: g_sortie(x, 0.0),
        R=args.R
    )

    # opérateur total : diffusion + advection + Robin

    M_phys = rho * cp * M      # masse physique ρ·cp·M
    A = K + Rb                # pas d'advection en diffusion pure

    # ------------------------------------------------------------
    # 8) Figure interactive
    # ------------------------------------------------------------
    _, ax = setup_interactive_figure()

   # ------------------------------------------------------------
    # 9) Boucle en temps principale
    # ------------------------------------------------------------
    frames_U = []

    for step in range(nsteps):
        t = step * dt
        Fn = F0 + F_neu + Fb_robin
        Fnp1 = F0 + F_neu + Fb_robin
        dir_vals_np1 = np.array([uD(dof_coords[d], t + dt) for d in dir_dofs], dtype=float)
        U = theta_step(
            M=M_phys, K=A, F_n=Fn, F_np1=Fnp1,
            U_n=U, dt=dt, theta=args.theta,
            dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals_np1
        )
        if step % 5 == 0:
            frames_U.append((t, U.copy()))

    # ------------------------------------------------------------
    # 10) Validation visuelle FEM vs Bessel
    # ------------------------------------------------------------
    from errors import compute_L2_H1_errors

    alpha = k_nak / (rho * cp)

    def zeros_J0(N_terms, R):
        lams = []
        for n in range(1, N_terms + 1):
            a = (n - 0.75) * np.pi / R
            b = (n + 0.25) * np.pi / R
            try:
                lams.append(brentq(lambda x: j0(x * R), a, b))
            except:
                pass
        return np.array(lams)

    lams = zeros_J0(30, args.R)

    def u_exact_t(r, t):
        T = T_in
        for lam in lams:
            coeff = 2.0 * (T0 - T_in) / (args.R * lam * j1(lam * args.R))
            T += coeff * j0(lam * r) * np.exp(-alpha * lam**2 * t)
        return T

    # Dirichlet sur Entree + Wall pour correspondre à Bessel
    dir_dofs_valid = border_dofs_from_tags(
        np.concatenate([Entree_tags, wall_tags]), tag_to_dof
    )
    dir_vals_valid = np.full(len(dir_dofs_valid), T_in)

    U_valid = np.array([u0(x) for x in dof_coords], dtype=float)
    U_valid[dir_dofs_valid] = dir_vals_valid

    t_compare = 1.0  # pendant le transitoire (t_diff,r ≈ 3.75s)
    step_compare = int(t_compare / dt)
    U_valid_mid = None

    for step in range(nsteps):
        U_valid = theta_step(
            M=M_phys, K=A,
            F_n=np.zeros(len(dof_coords)),
            F_np1=np.zeros(len(dof_coords)),
            U_n=U_valid, dt=dt, theta=args.theta,
            dirichlet_dofs=dir_dofs_valid,
            dir_vals_np1=dir_vals_valid
        )
        if step == step_compare:
            U_valid_mid = U_valid.copy()

    if U_valid_mid is None:
        print("ATTENTION : step_compare > nsteps, utilise U_valid final")
        U_valid_mid = U_valid.copy()

    # Nœuds FEM à z≈L/2
    z_target = args.L / 2
    mask = np.abs(dof_coords[:, 1] - z_target) < args.lc * 2
    dofs_z = np.where(mask)[0]
    r_fem = dof_coords[dofs_z, 0]
    T_fem_mid = U_valid_mid[dofs_z]
    order = np.argsort(r_fem)

    r_exact = np.linspace(0, args.R, 200)
    T_exact = np.array([u_exact_t(r, t_compare) for r in r_exact])

    fig_val, ax_val = plt.subplots(figsize=(7, 4))
    r_exact = np.linspace(0, args.R, 200)

    for t_comp, color in [(0.5, 'blue'), (1.0, 'red'), (2.0, 'green')]:
        step_comp = int(t_comp / dt)
        
        U_v = np.array([u0(x) for x in dof_coords], dtype=float)
        U_v[dir_dofs_valid] = dir_vals_valid
        for step in range(step_comp):
            U_v = theta_step(
                M=M_phys, K=K,
                F_n=np.zeros(len(dof_coords)),
                F_np1=np.zeros(len(dof_coords)),
                U_n=U_v, dt=dt, theta=args.theta,
                dirichlet_dofs=dir_dofs_valid,
                dir_vals_np1=dir_vals_valid
            )
        
        T_ex = np.array([u_exact_t(r, t_comp) for r in r_exact])
        ax_val.plot(r_exact*1000, T_ex - 273.15, '-', color=color,
                    linewidth=2, label=f'Bessel t={t_comp}s')
        
        mask = np.abs(dof_coords[:,1] - args.L/2) < args.lc * 2
        dofs_z = np.where(mask)[0]
        r_fem = dof_coords[dofs_z, 0]
        T_fem = U_v[dofs_z]
        order = np.argsort(r_fem)
        ax_val.plot(r_fem[order]*1000, T_fem[order] - 273.15, 'o',
                    color=color, markersize=5, label=f'FEM t={t_comp}s')

    ax_val.set_xlabel("r [mm]")
    ax_val.set_ylabel("T [°C]")
    ax_val.set_title("Validation FEM vs Bessel, z=L/2")
    ax_val.legend()
    ax_val.grid(True)
    plt.tight_layout()
    plt.savefig("figures/validation_transitoire.png")
    plt.show()

    # Sauvegarde avant gmsh_finalize
    nodeTags_saved = np.array(nodeTags, dtype=np.int64).copy()
    nodeCoords_saved = np.array(nodeCoords, dtype=float).copy()
    elemNodeTags_saved = np.array(elemNodeTags, dtype=np.int64).copy()
    gmsh_finalize()

    # ------------------------------------------------------------
    # 11) Animation
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 8))

    _, Uf0 = frames_U[0]
    contour0 = plot_fe_solution_2d(
        elemNodeTags=elemNodeTags_saved, nodeCoords=nodeCoords_saved,
        nodeTags=nodeTags_saved, U=Uf0, tag_to_dof=tag_to_dof,
        show_mesh=False, ax=ax, vmin_val=T_ext, vmax_val=T_in
    )
    cbar = fig.colorbar(contour0, ax=ax, label="Température [K]")
    cbar.set_ticks([T_ext, (T_ext+T_in)/2, T_in])
    cbar.set_ticklabels([f"{T_ext-273.15:.0f}°C", f"{(T_ext+T_in)/2-273.15:.0f}°C", f"{T_in-273.15:.0f}°C"])

    def update(frame):
        t, Uf = frame
        ax.clear()
        plot_fe_solution_2d(
            elemNodeTags=elemNodeTags_saved, nodeCoords=nodeCoords_saved,
            nodeTags=nodeTags_saved, U=Uf, tag_to_dof=tag_to_dof,
            show_mesh=False, ax=ax, vmin_val=T_ext, vmax_val=T_in
        )
        ax.set_title(f"t = {t:.1f} s")
        ax.set_xlabel("r"); ax.set_ylabel("z")
        ax.set_xlim(0, args.R)
        ax.set_ylim(0, args.L)
        ax.set_aspect('auto')

    ani = FuncAnimation(fig, update, frames=frames_U, interval=100)
    ani.save("figures/simulation_NaK.gif", writer=PillowWriter(fps=10))
    print("GIF sauvegardé.")
    plt.show()

    """
        plot_advection_field_uniform(
            beta_fun=beta,
            R=args.R,
            L=args.L,
            ax=ax,
            nr=6,
            nz=14,
            arrow_len=0.12
        )

    gmsh_finalize() """


if __name__ == "__main__":
    main()