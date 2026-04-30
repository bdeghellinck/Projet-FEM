""" TODO
 1. enlever scipy.optimize et juste faire avec une liste de valeurs de R
 2. quand R augmente normalement delta_P décroit mais Q_wall aussi ! Car h prop to 1/R, 
    mais ici c'est pas le cas, grand Q_wall pour grand R --> probleme
 3. régler le 2 et apres trouver le R optimal pour maximiser Q_wall tout en minimisant les pertes de charge (delta_P)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

from gmsh_utils import (
    gmsh_finalize,
    build_axi_reservoir_mesh,
    prepare_quadrature_and_basis,
    get_jacobians,
    border_dofs_from_tags,
)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from Robin import assemble_robin_wall
from Advection import assemble_advection


# ============================================================
# 0)  Global parameters
# ============================================================

Re_max = 5000        # strictly laminar threshold
dP_max = 50.0        # Pa

R_res  = 0.10        # m  — reservoir radius (fixed)
L_res  = 0.08        # m  — reservoir length (fixed)
L_pipe = 0.50        # m  — pipe length (fixed)
Qv     = 1.0e-4      # m3/s
T0     = 300 + 273.15  # K  inlet temperature
T_ext  = 500 + 273.15  # K  external NaK temperature

THETA  = 1.0         # fully implicit
DT     = 1.0         # s
NSTEPS = 2000        # max steps — convergence checked below
CONV_TOL = 1e-5      # relative change in T_max to declare steady state
CONV_CHECK_EVERY = 50


# ============================================================
# 1)  NaK-78 properties
# ============================================================

def nak_properties(T_K):
    T_c   = T_K - 273.15
    x_Na, x_K = 0.22, 0.78
    rho_Na = 945.3  - 0.22473 * T_c
    rho_K  = 841.5  - 0.2172  * T_c - 2.7e-5 * T_c**2 + 4.77e-9 * T_c**3
    rho    = 1.0 / (x_K / rho_K + x_Na / rho_Na)
    k      = 21.4  + 2.07e-2 * T_c - 2.2e-5  * T_c**2
    cp     = 232   - 8.82e-2 * T_c + 8.23e-5 * T_c**2
    mu     = 5.15e-4 * np.exp(695.0 / T_K)
    nu     = mu / rho
    Pr     = mu * cp / k
    return rho, cp, k, mu, nu, Pr


def compute_h_robin(T_op, R_pipe):
    rho, cp, k, mu, nu, Pr = nak_properties(T_op)
    D      = 2.0 * R_pipe
    U_mean = Qv / (np.pi * R_pipe**2)
    Re     = U_mean * D / nu
    Pe     = Re * Pr
    # Sleicher–Rouse correlation for liquid metals (valid Pe > 100)
    Nu     = 7.0 + 0.025 * Pe**0.8
    h      = Nu * k / D
    return h, Re, Pe


def compute_Re_global(R_pipe, rho, mu):
    return (2.0 * rho * Qv) / (np.pi * mu * R_pipe)


def compute_dP(R_pipe, mu):
    return 8.0 * mu * L_pipe * Qv / (np.pi * R_pipe**4)


# ============================================================
# FIX 1 — Correct Q_wall from Robin FEM matrices
# ============================================================

def compute_Q_wall_robin(Rb, Fb, U, wall_pipe_dofs):
    """
    The Robin contribution to the weak form is:
        a_robin(u,v) = ∫_Γ h·u·v dΓ      →  Rb
        l_robin(v)   = ∫_Γ h·T_ext·v dΓ   →  Fb

    At steady state the Robin residual on the wall dofs is:
        Q_wall = Fb[wall] - Rb[wall,:] @ U
                = ∫_Γ h·(T_ext - T_wall) dΓ   [W]  (positive = heat INTO fluid)

    This is the physically correct net wall heat flux.
    """
    # Extract rows of Rb corresponding to wall dofs
    Rb_wall = Rb[wall_pipe_dofs, :]
    Fb_wall = Fb[wall_pipe_dofs]
    q_nodal = Fb_wall - Rb_wall.dot(U)
    return float(q_nodal.sum())


def compute_Q_wall_direct(h, T_wall_mean, R_pipe):
    """Quick analytical cross-check: Q = h·A·(T_ext - T_wall)."""
    A_wall = 2.0 * np.pi * R_pipe * L_pipe
    return h * A_wall * (T_ext - T_wall_mean)


# ============================================================
# 2)  FEM simulation
# ============================================================

def run_simulation(R_pipe, verbose=False):
    """
    Steady-state FEM simulation for a tube of radius R_pipe.
    L_pipe is the global constant.
    Returns (Q_wall [W], Re, h [W/m²K], dP [Pa]).

    NOTE: caller is responsible for calling gmsh_finalize() if needed.
    """
    rho, cp, k_nak, mu, nu, Pr = nak_properties(0.5 * (T0 + T_ext))

    lc_pipe = min(R_pipe / 6, 0.002)   # finer mesh for small radii
    lc_res  = min(R_res  / 10, 0.010)

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        build_axi_reservoir_mesh(
            R_res=R_res, R_pipe=R_pipe,
            L_res=L_res, L_pipe=L_pipe,
            lc_res=lc_res, lc_pipe=lc_pipe,
            order=1,
        )

    # --- DOF mapping ---
    unique_tags         = np.unique(elemNodeTags)
    num_dofs            = len(unique_tags)
    max_tag             = int(np.max(nodeTags))
    all_coords          = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    tag_to_dof          = np.full(max_tag + 1, -1, dtype=int)
    node_index_from_tag = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(nodeTags):
        node_index_from_tag[int(tag)] = i
    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    for i, tag in enumerate(unique_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[node_index_from_tag[int(tag)]]

    # --- Quadrature / basis ---
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, 1)
    jac, det, coords = get_jacobians(elemType, xi)

    # --- Mass matrix ---
    M_lil = assemble_mass(
        elemTags=elemTags, conn=elemNodeTags,
        det=det, xphys=coords, w=w, N=N, tag_to_dof=tag_to_dof,
    )

    # --- Stiffness + RHS ---
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

    # --- Advection velocity (Poiseuille inside pipe) ---
    def beta(x):
        r, z = x[0], x[1]
        if (L_res < z < L_res + L_pipe) and (r <= R_pipe):
            U_mean = Qv / (np.pi * R_pipe**2)
            return np.array([0., 2.0 * U_mean * (1.0 - (r / R_pipe)**2), 0.])
        return np.zeros(3)

    C_lil = assemble_advection(
        elemTags=elemTags, conn=elemNodeTags,
        jac=jac, det=det, xphys=coords,
        w=w, N=N, gN=gN,
        beta_fun=beta, tag_to_dof=tag_to_dof,
    )
    C = rho * cp * C_lil.tocsr()

    # --- Boundaries ---
    boundary_tags  = {name: bnds_tags[i] for i, (name, _) in enumerate(bnds)}
    entree_dofs    = border_dofs_from_tags(boundary_tags["Entree"],    tag_to_dof)
    wall_pipe_dofs = border_dofs_from_tags(boundary_tags["Wall_pipe"], tag_to_dof)

    dir_dofs = entree_dofs
    dir_vals = np.full(len(entree_dofs), T0, dtype=float)
    U = np.full(num_dofs, T0, dtype=float)
    U[dir_dofs] = dir_vals

    Rb, Fb, h_last, Re_last = None, None, None, None
    T_max_prev = np.inf

    for step in range(NSTEPS):
        T_wall_mean     = float(np.mean(U[wall_pipe_dofs]))
        h_robin, Re, Pe = compute_h_robin(T_wall_mean, R_pipe)
        Rb, Fb          = assemble_robin_wall(wall_pipe_dofs, dof_coords, h_robin, T_ext)
        A_mat           = K + Rb + C
        F               = F0 + Fb
        U = theta_step(
            M=M_phys, K=A_mat, F_n=F, F_np1=F,
            U_n=U, dt=DT, theta=THETA,
            dirichlet_dofs=dir_dofs, dir_vals_np1=dir_vals,
        )
        h_last, Re_last = h_robin, Re

        # --- FIX 2: convergence check ---
        if step % CONV_CHECK_EVERY == 0 and step > 0:
            T_max = U.max()
            rel_change = abs(T_max - T_max_prev) / (abs(T_max_prev) + 1e-12)
            if verbose:
                print(f"    step {step:4d}  T_max={T_max-273.15:.2f}°C  "
                      f"rel_change={rel_change:.2e}")
            if rel_change < CONV_TOL:
                if verbose:
                    print(f"    -> Converged at step {step}")
                break
            T_max_prev = T_max

    # --- FIX 1: correct Q_wall ---
    # Convert Rb to CSR if needed so row-slicing works
    if not hasattr(Rb, 'tocsr'):
        Rb_csr = Rb.tocsr()
    else:
        Rb_csr = Rb

    Q_wall = compute_Q_wall_robin(Rb_csr, Fb, U, wall_pipe_dofs)

    # Cross-check with direct formula (printed in verbose mode)
    T_wall_mean_final = float(np.mean(U[wall_pipe_dofs]))
    Q_check = compute_Q_wall_direct(h_last, T_wall_mean_final, R_pipe)
    if verbose:
        print(f"    Q_wall (FEM Robin) = {Q_wall:.2f} W")
        print(f"    Q_wall (direct)    = {Q_check:.2f} W  (cross-check)")

    dP = compute_dP(R_pipe, mu)

    return Q_wall, Re_last, h_last, dP


# ============================================================
# 3)  Parametric scan
# ============================================================

def parametric_scan(R_pipe_values, verbose=False):
    rho_r, cp_r, k_r, mu_r, nu_r, Pr_r = nak_properties(0.5 * (T0 + T_ext))
    results = []

    for i, R_pipe in enumerate(R_pipe_values):
        Re = compute_Re_global(R_pipe, rho_r, mu_r)
        dP = compute_dP(R_pipe, mu_r)

        print(f"[{i+1}/{len(R_pipe_values)}] "
              f"R={R_pipe*1e3:.1f}mm  Re={Re:.0f}  dP={dP:.2f}Pa")

        # Physical constraints
        if Re > Re_max:
            print("  -> rejected: turbulent (Re > 2300)")
            continue
        if dP > dP_max:
            print("  -> rejected: pressure loss too high")
            continue
        if R_pipe >= R_res:
            print("  -> rejected: invalid geometry")
            continue

        try:
            Q_wall, Re_sim, h, dP_sim = run_simulation(R_pipe, verbose)
            gmsh_finalize() 

            A_wall = 2 * np.pi * R_pipe * L_pipe
            results.append({
                "R_pipe": R_pipe,
                "Re":     Re_sim,
                "h":      h,
                "dP":     dP_sim,
                "Q_wall": Q_wall,
                "A_wall": A_wall,
            })
            print(f"  -> OK: Q={Q_wall:.1f}W  dP={dP_sim:.2f}Pa  h={h:.0f}W/m2K")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            try:
                gmsh_finalize()
            except Exception:
                pass

    return results


# ============================================================
# 4)  Optimisation
# ============================================================

_rho_opt, _cp_opt, _k_opt, _mu_opt, _nu_opt, _Pr_opt = \
    nak_properties(0.5 * (T0 + T_ext))


def _is_feasible(R):
    Re = compute_Re_global(R, _rho_opt, _mu_opt)
    dP = compute_dP(R, _mu_opt)
    return (Re <= Re_max) and (dP <= dP_max) and (R < R_res)


def objective(R_array):
    R = float(R_array[0])
    if not _is_feasible(R):
        return 1e10   # large penalty for infeasible points

    try:
        Q_wall, Re, h, dP = run_simulation(R, verbose=False)
        gmsh_finalize()
    except Exception as e:
        print(f"  [OPT] simulation failed at R={R*1e3:.2f}mm: {e}")
        try:
            gmsh_finalize()
        except Exception:
            pass
        return 1e10

    print(f"  [OPT] R={R*1e3:.2f}mm | Q={Q_wall:.2f}W | Re={Re:.0f}")
    return -Q_wall   # minimise negative = maximise Q


def constraint_dP(R_array):
    R = float(R_array[0])
    dP = compute_dP(R, _mu_opt)
    return dP_max - dP   # >= 0


def constraint_Re(R_array):
    R = float(R_array[0])
    Re = compute_Re_global(R, _rho_opt, _mu_opt)
    return Re_max - Re   # >= 0


def run_optimization():
    """
    FIX 4 & 5: Two-stage optimisation.
    Stage 1 – Coarse scan of feasible R values using the FEM objective.
              (Differential evolution is overkill for 1D; we use a simple
               uniform sample as a warm start to avoid gradient noise issues.)
    Stage 2 – Local polish with SLSQP around the best candidate.
    """
    print("\n=== CONTINUOUS OPTIMISATION (2-stage) ===\n")

    # --- Stage 1: coarse feasibility-aware scan ---
    R_candidates = np.linspace(0.005, 0.019, 8)
    feasible = [(R, compute_Re_global(R, _rho_opt, _mu_opt),
                 compute_dP(R, _mu_opt))
                for R in R_candidates
                if _is_feasible(R)]

    if not feasible:
        print("No feasible candidate found in initial scan!")
        return

    print(f"Feasible candidates: {[f'{r*1e3:.1f}mm' for r,*_ in feasible]}\n")

    # Evaluate objective at each candidate
    best_R, best_Q = None, -np.inf
    for R, Re, dP in feasible:
        print(f"  Evaluating R={R*1e3:.1f}mm (Re={Re:.0f}, dP={dP:.2f}Pa)...")
        try:
            Q, _, _, _ = run_simulation(R, verbose=False)
            gmsh_finalize()
            print(f"    Q={Q:.2f}W")
            if Q > best_Q:
                best_Q, best_R = Q, R
        except Exception as e:
            print(f"    ERROR: {e}")
            try:
                gmsh_finalize()
            except Exception:
                pass

    if best_R is None:
        print("All simulations failed.")
        return

    print(f"\nStage 1 best: R={best_R*1e3:.2f}mm  Q={best_Q:.2f}W\n")

    # --- Stage 2: local SLSQP polish ---
    # FIX 5: use a finite-difference epsilon large enough to see past mesh noise
    # Rule of thumb: eps ~ lc_pipe ~ R/6 * 0.1 (10% of element size)
    eps = max(best_R * 0.05, 2e-4)   # at least 0.2 mm

    result = minimize(
        objective,
        x0=[best_R],
        method='SLSQP',
        bounds=[(0.004, 0.019)],
        constraints=[
            {"type": "ineq", "fun": constraint_dP},
            {"type": "ineq", "fun": constraint_Re},
        ],
        options={
            'maxiter': 15,
            'ftol': 5.0,        # [W] — meaningful given FEM noise level
            'eps': eps,         # FD step for gradient
            'disp': True,
        }
    )

    R_opt = result.x[0]

    print("\n=== FINAL RESULT ===")
    try:
        Q_wall, Re, h, dP = run_simulation(R_opt, verbose=True)
        gmsh_finalize()
        print(f"R_opt   = {R_opt*1e3:.2f} mm")
        print(f"Q_wall  = {Q_wall:.2f} W")
        print(f"dP      = {dP:.2f} Pa")
        print(f"Re      = {Re:.0f}")
        print(f"h       = {h:.0f} W/m²K")
        if Re > Re_max:
            print("  WARNING: Re > 2300, solution is outside laminar regime!")
    except Exception as e:
        print(f"Final evaluation failed: {e}")
        try:
            gmsh_finalize()
        except Exception:
            pass


# ============================================================
# 5)  Visualisation
# ============================================================

def plot_results(results):
    if not results:
        print("No results to plot.")
        return

    R  = np.array([r["R_pipe"]  for r in results]) * 1e3   # mm
    Q  = np.array([r["Q_wall"]  for r in results])          # W
    dP = np.array([r["dP"]      for r in results])          # Pa
    h  = np.array([r["h"]       for r in results])          # W/m²K
    Re = np.array([r["Re"]      for r in results])
    A  = np.array([r["A_wall"]  for r in results]) * 1e4    # cm²

    # Expected analytical trends (for sanity check overlay)
    R_an  = np.linspace(R.min(), R.max(), 200)
    rho_r, cp_r, k_r, mu_r, nu_r, Pr_r = nak_properties(0.5 * (T0 + T_ext))
    dP_an = compute_dP(R_an * 1e-3, mu_r)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"NaK bi-criteria optimisation | L={L_pipe*100:.0f}cm | "
        f"Qv={Qv:.1e} m³/s | T_ext={T_ext-273.15:.0f}°C",
        fontsize=12
    )

    # Q_wall vs R
    ax = axes[0, 0]
    ax.plot(R, Q, 'o-', color='crimson', lw=2, ms=7)
    ax.set_xlabel("R_pipe [mm]")
    ax.set_ylabel("Q_wall [W]")
    ax.set_title("Wall heat flux (FEM Robin, corrected)")
    ax.grid(True, alpha=0.3)

    # deltaP vs R  (+ analytical overlay)
    ax = axes[0, 1]
    ax.plot(R_an, dP_an, '--', color='lightblue', lw=1.5, label='Hagen-Poiseuille')
    ax.plot(R, dP, 's-', color='steelblue', lw=2, ms=6, label='FEM points')
    ax.set_xlabel("R_pipe [mm]")
    ax.set_ylabel("ΔP [Pa]")
    ax.set_title("Pressure losses  (∝ 1/R⁴)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Pareto front: Q vs deltaP
    ax = axes[0, 2]
    sc = ax.scatter(dP, Q, c=R, cmap='plasma', s=80, zorder=5)
    ax.plot(dP, Q, '-', color='gray', lw=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='R_pipe [mm]')
    for j in range(len(R)):
        ax.annotate(f"{R[j]:.1f}mm", (dP[j], Q[j]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("ΔP [Pa]  (minimise)")
    ax.set_ylabel("Q_wall [W]  (maximise)")
    ax.set_title("Pareto front  (top-left = best)")
    ax.grid(True, alpha=0.3)

    # h vs R
    ax = axes[1, 0]
    ax.plot(R, h, '^-', color='darkorange', lw=2, ms=6)
    ax.set_xlabel("R_pipe [mm]")
    ax.set_ylabel("h [W/m²K]")
    ax.set_title("Convection coefficient  (∝ 1/R)")
    ax.grid(True, alpha=0.3)

    # A_wall vs R
    ax = axes[1, 1]
    ax.plot(R, A, 'p-', color='purple', lw=2, ms=6)
    ax.set_xlabel("R_pipe [mm]")
    ax.set_ylabel("A_wall [cm²]")
    ax.set_title("Exchange surface  (∝ R)")
    ax.grid(True, alpha=0.3)

    # Re vs R
    ax = axes[1, 2]
    ax.plot(R, Re, 'x-', color='slategray', lw=2, ms=8)
    ax.axhline(2300, color='red', ls='--', alpha=0.7, label='Re = 2300')
    ax.set_xlabel("R_pipe [mm]")
    ax.set_ylabel("Re [–]")
    ax.set_title("Reynolds number  (∝ 1/R)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/parametric_study.png", dpi=150)
    print("Figure saved: figures/parametric_study.png")
    plt.show()

    # Summary table
    print("\n" + "=" * 62)
    print(f"{'R[mm]':>7} {'Q[W]':>8} {'ΔP[Pa]':>9} {'h[W/m²K]':>10} {'Re':>6}")
    print("-" * 50)
    for j in range(len(R)):
        print(f"{R[j]:>7.1f} {Q[j]:>8.1f} {dP[j]:>9.3f} {h[j]:>10.0f} {Re[j]:>6.0f}")
    print("=" * 62)

    print("\nPareto front reading:")
    print(f"  Max Q_wall -> R = {R[np.argmax(Q)]:.1f} mm  "
          f"(ΔP = {dP[np.argmax(Q)]:.3f} Pa)")
    print(f"  Min ΔP     -> R = {R[np.argmin(dP)]:.1f} mm  "
          f"(Q  = {Q[np.argmin(dP)]:.1f} W)")
    print("\nExpected physical trends:")
    print("  Q_wall should be NON-MONOTONIC in R:")
    print("    - small R: h large (good) but A small -> Q peaks at intermediate R")
    print("    - large R: A large (good) but h small -> Q peaks at intermediate R")
    print("  ΔP should be strictly DECREASING with R (1/R⁴ law)")


# ============================================================
# 6)  Entry point
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    R_pipe_values = np.linspace(0.005, 0.019, 10)

    rho_r, cp_r, k_r, mu_r, nu_r, Pr_r = nak_properties(0.5 * (T0 + T_ext))

    print("=" * 60)
    print(f"NaK bi-criteria study — {len(R_pipe_values)} geometries")
    print(f"L_pipe={L_pipe*100:.0f}cm | Qv={Qv:.2e}m³/s | Re_max={Re_max}")
    print("=" * 60)
    print(f"\n{'R[mm]':>7} {'Re':>6} {'ΔP[Pa]':>9}  status")
    print("-" * 38)
    for R in R_pipe_values:
        Re = compute_Re_global(R, rho_r, mu_r)
        dP = compute_dP(R, mu_r)
        tag = "ok" if Re < Re_max else "X turbulent"
        print(f"{R*1e3:>7.1f} {Re:>6.0f} {dP:>9.3f}  {tag}")

    print("\n--- FEM parametric scan ---\n")
    results = parametric_scan(R_pipe_values, verbose=False)

    if results:
        best = max(results, key=lambda r: r["Q_wall"])
        print("\n" + "=" * 60)
        print("CONSTRAINED OPTIMUM (parametric scan)")
        print("=" * 60)
        print(f"R_opt   = {best['R_pipe']*1e3:.2f} mm")
        print(f"Q_max   = {best['Q_wall']:.2f} W")
        print(f"ΔP      = {best['dP']:.2f} Pa")
        print(f"Re      = {best['Re']:.0f}")
        print("=" * 60)

    plot_results(results)
    run_optimization()