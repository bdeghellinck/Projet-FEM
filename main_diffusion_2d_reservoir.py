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
    plot_full_reservoir_3d
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d
from Neumann import assemble_rhs_neumann_outlet
from Robin import assemble_robin_wall
from Advection import assemble_advection


# ============================================================
# Propriétés NaK définies au niveau module (partagées entre
# main() et les fonctions de l'étude paramétrique)
# ============================================================

def nak_properties(T_K):
    """ valeurs et equations tirées de : https://mooseframework.inl.gov/source/fluidproperties/NaKFluidProperties.html """

    T_c = T_K - 273.15   # conversion en celsius

    # Fractions massiques
    x_Na = 0.22
    x_K = 0.78

    # Densité Na liquide (Eq. 1.5) [kg/m³]  — T_c en °C
    rho_Na = 945.3- 0.22473 * T_c

    # Densité K liquide (Eq. 1.8) [kg/m³]
    rho_K  = 841.5 - 0.2172 * T_c - 2.7e-5  * T_c**2 + 4.77e-9* T_c**3

    # Densité NaK (Eq. 1.9) — règle des volumes massiques
    rho = 1.0 / (x_K / rho_K + x_Na / rho_Na)

    # ------------------------------------------------------------------
    # Conductivité thermique (Eq. 1.53) [W/(m·K)]
    # ------------------------------------------------------------------
    k = 21.4 + 2.07e-2 * T_c - 2.2e-5 * T_c**2

    # ------------------------------------------------------------------
    # Capacité calorifique (Eq. 1.59) [J/(kg·K)]
    # ------------------------------------------------------------------
    cp = 232 - 8.82e-2 * T_c + 8.23e-5 * T_c**2

    # ------------------------------------------------------------------
    # Viscosité dynamique (Eq. 1.18-1.19) — corrélation Andrade
    # ------------------------------------------------------------------
    mu = 5.15e-4 * np.exp(695.0 / T_K)   # Pa·s

    # ------------------------------------------------------------------
    # Propriétés dérivées
    # ------------------------------------------------------------------
    nu = mu / rho       # viscosité cinématique [m²/s]
    Pr = mu * cp / k    # nombre de Prandtl [-]

    return rho, cp, k, mu, nu, Pr


# ============================================================
# Fonctions utilitaires pour l'étude paramétrique (section 13)
# ============================================================

def compute_Q_wall_fem(Rb, Fb, U, wall_pipe_dofs):
    """
    Flux thermique net à travers la paroi du tube [W], calculé à partir
    des matrices Robin FEM.

    La contribution Robin à la forme faible est :
        a_robin(u,v) = ∫_Γ h·u·v dΓ    →  Rb
        l_robin(v)   = ∫_Γ h·T_ext·v dΓ →  Fb

    En régime permanent, le flux net vaut :
        Q_wall = Fb[wall] - Rb[wall,:] @ U
               = ∫_Γ h·(T_ext − T_wall) dΓ   [W]
                 (positif = chaleur vers le fluide)
    """
    Rb_csr  = Rb.tocsr() if hasattr(Rb, 'tocsr') else Rb
    q_nodal = Fb[wall_pipe_dofs] - Rb_csr[wall_pipe_dofs, :].dot(U)
    return float(q_nodal.sum())


def compute_dP_hagen(R_pipe, mu, L_pipe, Qv):
    """Perte de charge de Hagen-Poiseuille [Pa] : ΔP = 8μLQ / (πR⁴)."""
    return 8.0 * mu * L_pipe * Qv / (np.pi * R_pipe**4)


def compute_h_robin_from_Qv(T_op, R_pipe, Qv):
    """
    Coefficient de convection h [W/(m²·K)] — corrélation Lyon-Martinelli.
    Valable pour les métaux liquides (Pe > 100).
    Signature adaptée pour l'étude paramétrique : (T_op, R_pipe, Qv) → (h, Pe, Nu, Re)
    """
    rho, cp, k, mu, nu, Pr = nak_properties(T_op)
    D_pipe = 2.0 * R_pipe
    U_mean = Qv / (np.pi * R_pipe**2)
    Re     = U_mean * D_pipe / nu
    Pe     = Re * Pr
    Nu     = 7.0 + 0.025 * Pe**0.8
    h      = Nu * k / D_pipe
    return h, Pe, Nu, Re


def run_one_simulation(args, T0, T_ext, Qv,
                       R_pipe=None, L_pipe=None,
                       nsteps=5000, conv_tol=1e-5, conv_check_every=50,
                       verbose=False):
    """
    Lance une simulation FEM complète et retourne les grandeurs d'intérêt.

    R_pipe et L_pipe peuvent être passés explicitement pour l'étude
    paramétrique ; sinon les valeurs de args sont utilisées.

    NOTE : l'appelant est responsable d'appeler gmsh_finalize() après.

    Retourne un dict : Q_wall, Re, h, Nu, Pe, dP, T_wall_mean, A_wall
    """
    if R_pipe is None:
        R_pipe = args.R_pipe
    if L_pipe is None:
        L_pipe = args.L_pipe

    rho, cp, k_nak, mu, nu, Pr = nak_properties(0.5 * (T0 + T_ext))

    lc_pipe = min(R_pipe / 6.0, 0.002)
    lc_res  = min(args.R_res / 10.0, 0.010)

    # --- Maillage ---
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        build_axi_reservoir_mesh(
            R_res=args.R_res, R_pipe=R_pipe,
            L_res=args.L_res, L_pipe=L_pipe,
            lc_res=lc_res, lc_pipe=lc_pipe,
            order=1,
        )

    # --- Mapping Gmsh node tag -> dof compact ---
    unique_dofs_tags    = np.unique(elemNodeTags)
    num_dofs            = len(unique_dofs_tags)
    max_tag             = int(np.max(nodeTags))
    all_coords          = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    tag_to_dof          = np.full(max_tag + 1, -1, dtype=int)
    node_index_from_tag = np.full(max_tag + 1, -1, dtype=int)
    for i_node, tag in enumerate(nodeTags):
        node_index_from_tag[int(tag)] = i_node
    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    for i_dof, tag in enumerate(unique_dofs_tags):
        tag_int = int(tag)
        tag_to_dof[tag_int] = i_dof
        dof_coords[i_dof]   = all_coords[node_index_from_tag[tag_int]]

    # --- Quadrature et jacobiens ---
    xi, w, N, gN     = prepare_quadrature_and_basis(elemType, 1)
    jac, det, coords = get_jacobians(elemType, xi)

    # --- Assemblage matrices ---
    M_lil = assemble_mass(
        elemTags=elemTags, conn=elemNodeTags,
        det=det, xphys=coords, w=w, N=N, tag_to_dof=tag_to_dof,
    )
    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags=elemTags, conn=elemNodeTags,
        jac=jac, det=det, xphys=coords,
        w=w, N=N, gN=gN,
        kappa_fun=lambda x: k_nak,
        rhs_fun=lambda x: 0.0,
        tag_to_dof=tag_to_dof,
    )
    M      = M_lil.tocsr()
    K      = K_lil.tocsr()
    M_phys = rho * cp * M

    # Advection de Poiseuille dans le tube uniquement
    U_mean_tube = Qv / (np.pi * R_pipe**2)

    def beta(x):
        r, z = x[0], x[1]
        in_tube = (args.L_res < z < args.L_res + L_pipe) and (r <= R_pipe)
        if in_tube:
            return np.array([0., 2.0 * U_mean_tube * (1.0 - (r / R_pipe)**2), 0.])
        return np.zeros(3)

    C_lil = assemble_advection(
        elemTags=elemTags, conn=elemNodeTags,
        jac=jac, det=det, xphys=coords,
        w=w, N=N, gN=gN,
        beta_fun=beta, tag_to_dof=tag_to_dof,
    )
    C = rho * cp * C_lil.tocsr()

    # --- Conditions aux limites ---
    boundary_tags  = {name: bnds_tags[i] for i, (name, _) in enumerate(bnds)}
    entree_dofs    = border_dofs_from_tags(boundary_tags["Entree"],    tag_to_dof)
    sortie_dofs    = border_dofs_from_tags(boundary_tags["Sortie"],    tag_to_dof)
    wall_pipe_dofs = border_dofs_from_tags(boundary_tags["Wall_pipe"], tag_to_dof)

    dir_dofs = entree_dofs
    dir_vals = np.full(len(entree_dofs), T0, dtype=float)

    U = np.full(num_dofs, T0, dtype=float)
    U[dir_dofs] = dir_vals

    # --- Boucle en temps avec critère de convergence ---
    Rb, Fb, h_last, Re_last, Nu_last, Pe_last = None, None, None, None, None, None
    T_max_prev = np.inf

    for step in range(nsteps):
        T_wall_mean         = float(np.mean(U[wall_pipe_dofs]))
        h_robin, Pe, Nu, Re = compute_h_robin_from_Qv(T_wall_mean, R_pipe, Qv)

        Rb, Fb = assemble_robin_wall(wall_pipe_dofs, dof_coords, h_robin, T_ext)
        A      = K + Rb + C
        F      = F0 + Fb

        U = theta_step(
            M=M_phys, K=A, F_n=F, F_np1=F,
            U_n=U, dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals,
        )
        h_last, Re_last, Nu_last, Pe_last = h_robin, Re, Nu, Pe

        if step % conv_check_every == 0 and step > 0:
            T_max      = U.max()
            rel_change = abs(T_max - T_max_prev) / (abs(T_max_prev) + 1e-12)
            if verbose:
                print(f"    step {step:4d}  T_max={T_max-273.15:.2f}°C  rel={rel_change:.2e}")
            if rel_change < conv_tol:
                if verbose:
                    print(f"    → Convergé à l'itération {step}")
                break
            T_max_prev = T_max

    # Q_wall via bilan enthalpique à la sortie : Q = ṁ·cp·(T_sortie − T0)
    # C'est la mesure physiquement correcte de la chaleur gagnée par le fluide.
    # Le résidu Robin compute_Q_wall_fem est cohérent en régime permanent mais
    # difficile à interpréter quand h est très grand (NTU >> 1).
    T_sortie_mean = float(np.mean(U[sortie_dofs]))
    m_dot  = rho * Qv
    Q_wall = m_dot * cp * (T_sortie_mean - T0)

    dP_val       = compute_dP_hagen(R_pipe, mu, L_pipe, Qv)
    T_wall_final = float(np.mean(U[wall_pipe_dofs]))
    A_wall       = 2.0 * np.pi * R_pipe * L_pipe

    return dict(
        Q_wall      = Q_wall,
        Re          = Re_last,
        h           = h_last,
        Nu          = Nu_last,
        Pe          = Pe_last,
        dP          = dP_val,
        T_wall_mean = T_wall_final,
        T_sortie    = T_sortie_mean,
        A_wall      = A_wall,
    )


def parametric_study(L_pipe_values, args, T0, T_ext, Qv, dP_max=500.0):
    """
    Étude paramétrique sur L_pipe (longueur du tube), R_pipe fixé.

    Q_wall = ṁ·cp·(T_ext − T0)·(1 − exp(−NTU))   avec  NTU = h·2πR·L / (ṁ·cp)

      - Pour L petit  : NTU ≪ 1 → Q ≈ h·A·(T_ext−T0)  croît quasi linéairement avec L
      - Pour L grand  : NTU ≫ 1 → Q → ṁ·cp·(T_ext−T0) sature (fluide à saturation thermique)
      → courbe en forme de genou, avec un L* au-delà duquel ajouter de la longueur
        n'apporte presque plus rien thermiquement

    ΔP = 8μLQ/(πR⁴) croît linéairement avec L → compromis clair.

    Le maillage garde la même topologie pour tout L → pas de bruit numérique
    dû au remaillage, contrairement au scan en R.
    """
    rho_r, cp_r, k_r, mu_r, nu_r, Pr_r = nak_properties(0.5 * (T0 + T_ext))
    R_pipe = args.R_pipe

    # Re et h sont indépendants de L (R_pipe fixé)
    U_mean  = Qv / (np.pi * R_pipe**2)
    Re_ref  = U_mean * 2 * R_pipe / nu_r

    print("=" * 65)
    print(f"Étude paramétrique — {len(L_pipe_values)} valeurs de L_pipe")
    print(f"R_pipe={R_pipe*1e3:.1f} mm | Qv={Qv:.2e} m³/s | "
          f"Re={Re_ref:.0f}{' (⚠ turbulent)' if Re_ref > 2300 else ''}")
    print("=" * 65)
    print(f"\n  {'L[cm]':>7} {'ΔP[Pa]':>9} {'NTU_an':>8}  statut")
    print("  " + "-" * 38)

    feasible_L = []
    for L in L_pipe_values:
        dP  = compute_dP_hagen(R_pipe, mu_r, L, Qv)
        # NTU analytique (pour info, h évalué à T_moy)
        h_ref, *_ = compute_h_robin_from_Qv(0.5*(T0+T_ext), R_pipe, Qv)
        m_dot = rho_r * Qv
        NTU   = h_ref * 2 * np.pi * R_pipe * L / (m_dot * cp_r)
        if dP > dP_max:
            tag = "✗ ΔP trop élevé"
        else:
            tag = "✓"
            feasible_L.append(L)
        print(f"  {L*100:>7.1f}  {dP:>9.3f}  {NTU:>8.3f}  {tag}")

    print(f"\n  {len(feasible_L)}/{len(L_pipe_values)} longueurs valides → simulations FEM\n")

    results = []
    for i, L_pipe in enumerate(feasible_L):
        print(f"  [{i+1}/{len(feasible_L)}]  L_pipe = {L_pipe*100:.1f} cm ...")
        # Pour les tubes courts, le temps caractéristique de diffusion radiale
        # est t_diff ~ R²/α (indépendant de L), mais le temps d'advection est
        # t_adv ~ L/U_mean → tubes courts convergent plus vite.
        # On adapte nsteps pour garantir la convergence dans tous les cas.
        U_mean  = Qv / (np.pi * args.R_pipe**2)
        t_adv   = L_pipe / U_mean           # temps de transit [s]
        nsteps_L = max(5000, int(20 * t_adv / args.dt))
        try:
            res = run_one_simulation(args, T0, T_ext, Qv, L_pipe=L_pipe, nsteps = nsteps_L)
            gmsh_finalize()
            res["L_pipe"] = L_pipe
            results.append(res)
            print(f"         Q_wall={res['Q_wall']:8.2f} W   "
                  f"ΔP={res['dP']:7.3f} Pa   "
                  f"h={res['h']:6.0f} W/m²K")
        except Exception as e:
            print(f"         ERREUR : {e}")
            try:
                gmsh_finalize()
            except Exception:
                pass

    return results


def find_optimal_L(L, Q, dP, Q_max, results):
    """
    Trouve L* selon trois critères complémentaires :

    1. COP = Q / ΔP  [W/Pa]
       Rapport chaleur transférée / coût de pompage.
       Maximiser le COP donne le meilleur compromis énergétique.

    2. Genou de Pareto (courbure maximale sur la courbe Q vs ΔP)
       Le genou est le point où la courbe Q(ΔP) tourne le plus fort,
       i.e. où le gain en Q par unité de ΔP supplémentaire chute le plus.
       Méthode : on normalise Q et ΔP dans [0,1] puis on calcule la
       courbure κ = |Q''| / (1 + Q'²)^(3/2) via différences finies.

    3. Seuil d'efficacité η = 90 %
       L minimal tel que Q ≥ 0.9 · Q_max.
       Critère d'ingénierie simple : on accepte 10 % de perte thermique.

    Retourne un dict avec les indices et valeurs de L pour chaque critère.
    """
    COP = Q / dP   # W/Pa

    # --- Critère 1 : max COP ---
    i_cop = int(np.argmax(COP))

    # --- Critère 2 : genou de Pareto (courbure max sur Q vs ΔP normalisé) ---
    # Normalisation dans [0,1]
    dP_n = (dP - dP.min()) / (dP.max() - dP.min() + 1e-12)
    Q_n  = (Q  - Q.min())  / (Q.max()  - Q.min()  + 1e-12)
    # Différences finies pour dQ_n/ddP_n et d²Q_n/ddP_n²
    if len(L) >= 3:
        dQdP   = np.gradient(Q_n, dP_n)
        d2QdP2 = np.gradient(dQdP, dP_n)
        kappa  = np.abs(d2QdP2) / (1.0 + dQdP**2)**1.5
        # Le genou est le point de courbure max (excluant les bords)
        i_knee = int(np.argmax(kappa[1:-1])) + 1
    else:
        i_knee = i_cop   # fallback si pas assez de points

    # --- Critère 3 : seuil η = 90 % ---
    eta_target = 0.90
    i_eta = None
    for j, q in enumerate(Q):
        if q >= eta_target * Q_max:
            i_eta = j
            break
    if i_eta is None:
        i_eta = int(np.argmax(Q))   # fallback : max Q si seuil jamais atteint

    return dict(
        i_cop  = i_cop,  L_cop  = L[i_cop],  Q_cop  = Q[i_cop],  dP_cop  = dP[i_cop],
        i_knee = i_knee, L_knee = L[i_knee], Q_knee = Q[i_knee], dP_knee = dP[i_knee],
        i_eta  = i_eta,  L_eta  = L[i_eta],  Q_eta  = Q[i_eta],  dP_eta  = dP[i_eta],
        COP    = COP,
    )



def plot_parametric_results(results, T0, T_ext, args, Qv):
    """
    Trace les graphes de l'étude paramétrique en L_pipe.
    Identifie L* selon trois critères : COP max, genou de Pareto, η = 90 %.
    """
    if not results:
        print("Aucun résultat à tracer.")
        return

    R_pipe = args.R_pipe
    L  = np.array([r["L_pipe"]      for r in results])
    Q  = np.array([r["Q_wall"]      for r in results])
    dP = np.array([r["dP"]          for r in results])
    h  = np.array([r["h"]           for r in results])
    Ts = np.array([r["T_sortie"]    for r in results]) - 273.15

    rho_r, cp_r, k_r, mu_r, nu_r, Pr_r = nak_properties(0.5 * (T0 + T_ext))
    m_dot = rho_r * Qv
    Q_max = m_dot * cp_r * (T_ext - T0)

    h_ref, *_ = compute_h_robin_from_Qv(T0, R_pipe, Qv)

    L_an    = np.linspace(0, L.max() * 1.1, 300)
    NTU_an  = h_ref * 2 * np.pi * R_pipe * L_an / (m_dot * cp_r)
    Q_an    = Q_max * (1.0 - np.exp(-NTU_an))
    dP_an   = compute_dP_hagen(R_pipe, mu_r, L_an, Qv)
    NTU_fem = h_ref * 2 * np.pi * R_pipe * L / (m_dot * cp_r)
    dQdL    = np.gradient(Q, L)

    # --- Trouver les trois L* ---
    opt = find_optimal_L(L, Q, dP, Q_max, results)
    COP = opt["COP"]

    # (mpl_marker, symbol pour texte/légende)
    opt_styles = [
        ("COP max",  opt["i_cop"],  opt["L_cop"],  "gold",        "*",  "★"),
        ("Genou",    opt["i_knee"], opt["L_knee"], "deepskyblue", "D",  "◆"),
        ("η = 90 %", opt["i_eta"],  opt["L_eta"],  "limegreen",   "o",  "●"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Étude paramétrique NaK — R_pipe={R_pipe*1e3:.0f} mm | "
        f"Qv={Qv:.1e} m³/s | T_ext={T_ext-273.15:.0f}°C",
        fontsize=13, fontweight='bold'
    )

    # Q_wall vs L
    ax = axes[0, 0]
    ax.plot(L_an*100, Q_an, '--', color='gray', lw=1.5, label='NTU analytique')
    ax.plot(L*100, Q, 'o-', color='crimson', lw=2, ms=5, label='FEM')
    ax.axhline(Q_max, ls=':', color='black', alpha=0.4, label=f'Q_max = {Q_max:.0f} W')
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        ax.axvline(L_opt*100, ls='--', color=color, alpha=0.8, lw=1.5)
        ax.plot(L_opt*100, Q[idx], marker=mpl_marker, color=color,
                ms=12, zorder=6, label=f'{sym} {label} ({L_opt*100:.0f} cm)')
    ax.set_xlabel("L_pipe [cm]"); ax.set_ylabel("Q_wall [W]")
    ax.set_title("Flux thermique pariétal")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ΔP vs L
    ax = axes[0, 1]
    ax.plot(L_an*100, dP_an, '--', color='lightblue', lw=1.5, label='Hagen-Poiseuille')
    ax.plot(L*100, dP, 's-', color='steelblue', lw=2, ms=6, label='points FEM')
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        ax.axvline(L_opt*100, ls='--', color=color, alpha=0.8, lw=1.5, label=f'{sym} {label}')
    ax.set_xlabel("L_pipe [cm]"); ax.set_ylabel("ΔP [Pa]")
    ax.set_title("Pertes de charge  (∝ L)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Front de Pareto : Q vs ΔP
    ax = axes[0, 2]
    ax.plot(dP_an, Q_an, '--', color='gray', lw=1.5, label='NTU analytique')
    sc = ax.scatter(dP, Q, c=L*100, cmap='viridis', s=60, zorder=4)
    plt.colorbar(sc, ax=ax, label='L_pipe [cm]')
    for j in range(len(L)):
        ax.annotate(f"{L[j]*100:.0f}cm", (dP[j], Q[j]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6)
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        ax.plot(dP[idx], Q[idx], marker=mpl_marker, color=color, ms=12, zorder=6,
                label=f'{sym} {label}')
    ax.set_xlabel("ΔP [Pa]  ← minimiser"); ax.set_ylabel("Q_wall [W]  ↑ maximiser")
    ax.set_title("Front de Pareto  (genou = L*)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # COP = Q / ΔP vs L
    ax = axes[1, 0]
    ax.plot(L*100, COP, 'o-', color='mediumorchid', lw=2, ms=6)
    ax.plot(opt["L_cop"]*100, COP[opt["i_cop"]], marker='*', color='gold', ms=14, zorder=6,
            label=f'★ max COP = {COP[opt["i_cop"]]:.1f} W/Pa  @ {opt["L_cop"]*100:.0f} cm')
    ax.set_xlabel("L_pipe [cm]"); ax.set_ylabel("COP = Q / ΔP  [W/Pa]")
    ax.set_title("Coefficient de performance  (↑ = mieux)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Gain marginal dQ/dL
    ax = axes[1, 1]
    dQdL_an = Q_max * h_ref * 2*np.pi*R_pipe / (m_dot*cp_r) * np.exp(-NTU_an)
    ax.plot(L_an*100, dQdL_an, '--', color='gray', lw=1.5, label='analytique')
    ax.plot(L*100, dQdL, '^-', color='darkorange', lw=2, ms=6, label='FEM (diff. finies)')
    ax.axhline(0, color='black', lw=0.5)
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        ax.axvline(L_opt*100, ls='--', color=color, alpha=0.8, lw=1.5)
    ax.set_xlabel("L_pipe [cm]"); ax.set_ylabel("dQ/dL  [W/m]")
    ax.set_title("Gain marginal  (↓ exponentiellement)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # T_sortie vs L
    ax = axes[1, 2]
    T_sortie_an = (T_ext - 273.15) - (T_ext - T0) * np.exp(-NTU_an)
    ax.plot(L_an*100, T_sortie_an, '--', color='gray', lw=1.5, label='NTU analytique')
    ax.plot(L*100, Ts, 'x-', color='slategray', lw=2, ms=8, label='FEM (T_sortie)')
    ax.axhline(T_ext - 273.15, ls=':', color='red', alpha=0.5,
               label=f'T_ext = {T_ext-273.15:.0f}°C')
    ax.axhline(T0 - 273.15, ls=':', color='blue', alpha=0.5,
               label=f'T_0 = {T0-273.15:.0f}°C')
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        ax.axvline(L_opt*100, ls='--', color=color, alpha=0.8, lw=1.5)
    ax.set_xlabel("L_pipe [cm]"); ax.set_ylabel("T_sortie [°C]")
    ax.set_title("Température de sortie du fluide")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/parametric_study.png", dpi=150)
    print("Figure sauvegardée : figures/parametric_study.png")
    plt.show()

    # --- Tableau récapitulatif ---
    print("\n" + "=" * 78)
    print(f"  {'L[cm]':>7} {'Q[W]':>9} {'ΔP[Pa]':>9} "
          f"{'η[%]':>7} {'NTU':>7} {'COP[W/Pa]':>11}")
    print("  " + "-" * 62)
    for j in range(len(L)):
        markers = []
        if j == opt["i_cop"]:  markers.append("★ COP")
        if j == opt["i_knee"]: markers.append("◆ genou")
        if j == opt["i_eta"]:  markers.append("● η90%")
        tag = "  " + " | ".join(markers) if markers else ""
        eta = Q[j] / Q_max * 100
        print(f"  {L[j]*100:>7.1f}  {Q[j]:>9.2f}  {dP[j]:>9.3f}  "
              f"{eta:>7.1f}  {NTU_fem[j]:>7.3f}  {COP[j]:>11.1f}{tag}")
    print("=" * 78)

    print(f"\n  Q_max (saturation thermique) = {Q_max:.0f} W")
    print(f"\n  {'Critère':<18} {'L* [cm]':>9} {'Q [W]':>9} {'ΔP [Pa]':>9} {'η [%]':>7} {'COP':>8}")
    print("  " + "-" * 62)
    for label, idx, L_opt, color, mpl_marker, sym in opt_styles:
        eta = Q[idx] / Q_max * 100
        print(f"  {sym} {label:<16} {L_opt*100:>9.1f} {Q[idx]:>9.2f} "
              f"{dP[idx]:>9.3f} {eta:>7.1f} {COP[idx]:>8.1f}")
    print(f"\n  Lecture physique :")
    print(f"    ★ COP max  → meilleur rapport Q/ΔP  (critère énergétique)")
    print(f"    ◆ Genou    → courbure max du front de Pareto  (compromis géométrique)")
    print(f"    ● η = 90 % → L minimal pour 90 % de saturation  (critère d'ingénierie)")




# ============================================================
# main()
# ============================================================

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
    # --- étude paramétrique ---
    parser.add_argument("--study",  action="store_true",
                        help="Lance l'étude paramétrique sur L_pipe (section 13)")
    parser.add_argument("--n_L",    type=int,   default=8,
                        help="Nombre de valeurs de L_pipe pour le scan [--study]")
    parser.add_argument("--L_min",  type=float, default=0.05,
                        help="L_pipe minimal [m] pour le scan [--study]")
    parser.add_argument("--L_max",  type=float, default=0.80,
                        help="L_pipe maximal [m] pour le scan [--study]")
    parser.add_argument("--Qv",     type=float, default=1e-4,
                        help="Débit volumique [m³/s]")
    parser.add_argument("--dP_max", type=float, default=500.0,
                        help="Perte de charge maximale [Pa] [--study]")

    args = parser.parse_args()
    dt = args.dt
    nsteps = args.nsteps

    # ------------------------------------------------------------
    # 13) Etude paramétrique sur L_pipe
    #
    #     Objectif : trouver la longueur L_pipe qui maximise Q_wall
    #     tout en minimisant les pertes de charge ΔP (front de Pareto).
    #
    #     Q_wall sature exponentiellement avec L (modèle NTU) :
    #       Q = ṁ·cp·(T_ext−T0)·(1−exp(−NTU))   NTU = h·2πRL/(ṁ·cp)
    #     → courbe en genou, L* au-delà duquel le gain devient négligeable.
    #
    #     ΔP = 8μLQ/(πR⁴) croît linéairement avec L.
    #
    #     Avantage vs scan en R : même topologie de maillage pour tout L
    #     → pas de bruit numérique dû au remaillage.
    #
    #     Lancé avec : python main_diffusion_2d_reservoir.py --study
    #     Options    : --n_L, --L_min, --L_max, --Qv, --dP_max
    # ------------------------------------------------------------
    if args.study:
        import os
        os.makedirs("figures", exist_ok=True)

        T0    = 300 + 273.15
        T_ext = 500 + 273.15
        Qv    = args.Qv

        L_pipe_values = np.linspace(args.L_min, args.L_max, args.n_L)

        results = parametric_study(
            L_pipe_values=L_pipe_values,
            args=args,
            T0=T0, T_ext=T_ext, Qv=Qv,
            dP_max=args.dP_max,
        )

        plot_parametric_results(results, T0, T_ext, args, Qv)
        return   # pas de simulation principale en mode --study

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

    T0 = 300 + 273.15
    T_ext = 500 + 273.15

    rho   = 860.5   # kg/m³
    cp    = 976.0   # J/(kg·K)
    k_nak = 22.4    # W/(m·K)

    U_tube   = 0.1 # m/s -> valeur standard réelle pour le NaK dans la conduite de refroidissement 
    U_entree = U_tube * (args.R_pipe / args.R_res)**2  #conservation de la masse 

    def compute_h_robin(T_op, U_entree, R_res, R_pipe, k_NaK=None):

        rho, cp, k, mu, nu, Pr = nak_properties(T_op)

        D_pipe = 2.0 * R_pipe

        # Vitesse dans le tube (conservation masse)
        U_tube = U_entree * (R_res / R_pipe)**2

        Re = U_tube * D_pipe / nu 
        Pe = Re * Pr               

        # Corrélation Lyon-Martinelli (métaux liquides)
        Nu = 7.0 + 0.025 * Pe**0.8

        h  = Nu * k / D_pipe

        return h, Pe, Nu, Re, Pr, rho, cp, k, nu, mu
 
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

    # Dirichlet sur toute la face d'entrée (r=0 à R_res)
    dir_dofs = entree_dofs
    dir_vals = np.full(len(entree_dofs), T0, dtype=float)

    # Sortie : Neumann nul → rien à assembler
    U[dir_dofs] = dir_vals
 
    # ------------------------------------------------------------
    # 9) Boucle en temps (schema theta)
    # ------------------------------------------------------------
    frames_U = []

    # test cohérence des valeurs numériques 
    T_wall = np.mean(U[wall_pipe_dofs])
    h, Pe, Nu, Re, Pr, rho, cp, k, nu, mu = compute_h_robin(T_wall, U_entree, args.R_res, args.R_pipe)
    print('h = ', h, 'Pe = ', Pe, 'Nu = ', Nu, 'Re = ', Re, 'Pr = ', Pr, 'rho = ', rho, 
          'mu = ', mu, 'k = ', k, 'nu = ', nu)
 
    for step in range(args.nsteps):
        
        # condition de Robin, avec h évalué à chaque pas de temps 
        T_wall = np.mean(U[wall_pipe_dofs])
        h_robin, Pe, *_ = compute_h_robin(T_wall, U_entree, args.R_res, args.R_pipe)

        Rb, Fb = assemble_robin_wall(wall_pipe_dofs, dof_coords, h_robin, T_ext)

        A = K + Rb + C
        F = F0 + Fb

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

    # Q_wall via résidu Robin FEM (voir docstring compute_Q_wall_fem)
    Q_wall = compute_Q_wall_fem(Rb, Fb, U, wall_pipe_dofs)
    print(f"Q_wall = {Q_wall:.2f} W")
 
    # ------------------------------------------------------------
    # 10) Sauvegarde des arrays gmsh
    # ------------------------------------------------------------
    nodeTags_saved     = np.array(nodeTags,     dtype=np.int64).copy()
    nodeCoords_saved   = np.array(nodeCoords,   dtype=float).copy()
    elemNodeTags_saved = np.array(elemNodeTags, dtype=np.int64).copy()
 
    gmsh_finalize()

    # ------------------------------------------------------------
    # 11) 2D animation 
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
        vmin=T0, vmax=T_ext,
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
            vmin=T0, vmax=T_ext,
            swap_axes=True
        )

        ax.set_title(f"t = {t:.1f} s")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("auto")
 
    ani = FuncAnimation(fig, update, frames=frames_U, interval=80)
    ani.save("figures/simulation_2d.gif", writer=PillowWriter(fps=10))
    print("GIF sauvegarde : figures/simulation_2d.gif")
    plt.show()

    """"
    # ------------------------------------------------------------
    # 12) Static 3D plot of the final frame
    # ------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D  # registers '3d' projection, do not remove

    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection='3d')

    _, U_final = frames_U[-1]
    sm = plot_full_reservoir_3d(
        ax3d,
        nodeCoords=nodeCoords_saved,
        elemNodeTags=elemNodeTags_saved,
        nodeTags=nodeTags_saved,
        U=U_final,
        tag_to_dof=tag_to_dof,
        cmap='plasma',
        vmin=T0, vmax=T_ext,
    )
    fig3d.colorbar(sm, ax=ax3d, shrink=0.5, label="Température [K]")
    ax3d.set_title("Champ T — régime permanent (révolution 3D)")
    plt.tight_layout()
    plt.savefig("figures/solution_3d_final.png", dpi=150)
    plt.show()
    """


if __name__ == "__main__":
    main()



# pour run étude paramétrique uniquement : 
# python main_diffusion_2d_reservoir.py --study --R_pipe 0.02 --n_L 10 --L_min 0.05 --L_max 0.80