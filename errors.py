# errors.py
import numpy as np
import gmsh


def _numeric_grad_3d(u_exact, x, eps=1e-7):
    """
    Central finite-difference gradient in R^3 for a scalar function u_exact(x).
    x is a length-3 array-like.
    Returns grad as (3,) array.
    """
    x = np.asarray(x, dtype=float).reshape(3,)
    g = np.zeros(3, dtype=float)
    for i in range(3):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (float(u_exact(xp)) - float(u_exact(xm))) / (2.0 * eps)
    return g


def compute_L2_H1_errors(
    elemType,
    elemTags,
    elemNodeTags,
    U,
    xi, w, N, gN,
    jac, det, coords,
    u_exact,
    grad_exact=None,
    eps_grad=1e-7
):
    """
    Compute L2 and H1 errors against an analytical solution.

    Parameters
    ----------
    elemType : int
        Gmsh element type (line/triangle/tetra, etc.)
    elemTags : array-like, (ne,)
        Element tags from gmsh.model.mesh.getElementsByType(elemType)[0]
    elemNodeTags : array-like, flattened length ne*nloc
        Element connectivity from gmsh.model.mesh.getElementsByType(elemType)[1]
    U : ndarray, (nn,)
        FE solution vector aligned with gmsh compact node ordering (nodeTag-1 indexing).
    xi, w, N, gN :
        Outputs of prepare_quadrature_and_basis(elemType, order)
    jac, det, coords :
        Outputs of get_jacobians(elemType, xi)
        (flattened lists from gmsh.model.mesh.getJacobians)
    u_exact : callable
        u_exact(xyz) -> float, where xyz is length-3 array-like (x,y,z)
    grad_exact : callable or None
        grad_exact(xyz) -> array-like shape (3,) giving (du/dx, du/dy, du/dz).
        If None, gradient is approximated numerically from u_exact.
    eps_grad : float
        Step for numerical gradient if grad_exact is None.

    Returns
    -------
    err_L2 : float
        ||u_h - u||_{L2}
    err_H1_semi : float
        |u_h - u|_{H1} = ||grad(u_h - u)||_{L2}
    err_H1 : float
        ||u_h - u||_{H1} = sqrt(err_L2^2 + err_H1_semi^2)
    """
    U = np.asarray(U, dtype=float)

    # Element properties to infer nloc
    _, dim, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    ne = len(elemTags)
    ngp = len(w)

    # reshape gmsh data
    w = np.asarray(w, dtype=float)
    det = np.asarray(det, dtype=float).reshape(ne, ngp)
    jac = np.asarray(jac, dtype=float).reshape(ne, ngp, 3, 3)
    coords = np.asarray(coords, dtype=float).reshape(ne, ngp, 3)

    conn = np.asarray(elemNodeTags, dtype=np.int64).reshape(ne, nloc) - 1  # 0-based
    N = np.asarray(N, dtype=float).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=float).reshape(ngp, nloc, 3)

    # pick gradient function
    if grad_exact is None:
        def grad_fun(x):
            return _numeric_grad_3d(u_exact, x, eps=eps_grad)
    else:
        def grad_fun(x):
            return np.asarray(grad_exact(x), dtype=float).reshape(3,)

    I_L2 = 0.0
    I_H1 = 0.0

    # assembly-like integration over elements/gauss points
    for e in range(ne):
        nodes = conn[e, :]
        Ue = U[nodes]  # local nodal vector

        for g in range(ngp):
            xg = coords[e, g]              # (3,)
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            # FE value u_h(xg)
            uh = 0.0
            for a in range(nloc):
                uh += N[g, a] * Ue[a]

            # FE gradient grad u_h(xg)
            # grad(N_a) in physical coords = inv(J) * grad_ref(N_a)  (same as your assembly)
            grad_uh = np.zeros(3, dtype=float)
            for a in range(nloc):
                gradNa = invjacg @ gN[g, a]
                grad_uh += gradNa * Ue[a]

            # exact
            uex = float(u_exact(xg))
            gex = grad_fun(xg)

            du = uh - uex
            dg = grad_uh - gex

            I_L2 += wg * (du * du) * detg
            I_H1 += wg * float(np.dot(dg, dg)) * detg

    err_L2 = float(np.sqrt(max(I_L2, 0.0)))
    err_H1_semi = float(np.sqrt(max(I_H1, 0.0)))
    err_H1 = float(np.sqrt(max(I_L2 + I_H1, 0.0)))

    return err_L2, err_H1_semi, err_H1
