# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi, tag=-1):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords


def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right

def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Converts a list of GMSH node tags into the corresponding 
    compact matrix indices (DoFs).
    """
    # Ensure tags are integers
    l_tags = np.asarray(l_tags, dtype=int)
    
    # Filter out any tags that might not be in our DoF mapping (like geometry points)
    # then map them to our 0...N-1 indices
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    return l_dofs

def getPhysical(name):
    """
    Get the physical group elements and nodes for a given name and dimension.
    """
    
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dimTags[0][0], tag=dimTags[0][1])
    elemType = elemTypes[0]  # Assuming one element type per physical group
    elemTags = elemTags[0]
    elemNodeTags = elemNodeTags[0]
    entityTag = dimTags[0][1]
    return elemType, elemTags, elemNodeTags, entityTag
    

def open_2d_mesh(msh_filename, order=1):
    """
    Load a .msh file.

    Parameters
    ----------
    msh_filename : str
        Path to the .msh file
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(msh_filename)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    surf = gmsh.model.getEntities(2)[0][1]

    curve_tags = gmsh.model.getBoundary([(2, surf)], oriented=False)
    
    gmsh.model.addPhysicalGroup(1, [curve_tags[0][1]], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [curve_tags[1][1]], tag=2)
    gmsh.model.setPhysicalName(1, 2, "InnerBoundary")

    bnds = [('OuterBoundary', 1),('InnerBoundary', 1)]

    bnds_tags = []
    for name, dim in bnds:
        tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag = t[1]
                break
        if tag == -1:
            raise ValueError(f"Physical group '{name}' not found in mesh.")
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags


def build_conduit_mesh(R=1.0, L=5.0, lc=0.1, order=1):
    """
    Build a 2D axisymmetric conduit mesh in (r,z).

    Parameters
    ----------
    R : float
        Radius (max r)
    L : float
        Length (max z)
    lc : float
        Mesh size
    order : int
        Polynomial order

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
    """

    import gmsh
    import numpy as np

    gmsh.initialize()
    gmsh.model.add("conduit_axi")

    # -----------------------------
    # Points (r,z)
    # -----------------------------
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)   # axis, Entree
    p2 = gmsh.model.geo.addPoint(R, 0, 0, lc)   # wall, Entree
    p3 = gmsh.model.geo.addPoint(R, L, 0, lc)   # wall, Sortie
    p4 = gmsh.model.geo.addPoint(0, L, 0, lc)   # axis, Sortie

    # -----------------------------
    # Lines
    # -----------------------------
    l1 = gmsh.model.geo.addLine(p1, p2)  # Entree (z=0)
    l2 = gmsh.model.geo.addLine(p2, p3)  # Wall (r=R)
    l3 = gmsh.model.geo.addLine(p3, p4)  # Sortie (z=L)
    l4 = gmsh.model.geo.addLine(p4, p1)  # Axis (r=0)

    # -----------------------------
    # Surface
    # -----------------------------
    cloop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cloop])

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # -----------------------------
    # Physical groups (CRUCIAL)
    # -----------------------------
    gmsh.model.addPhysicalGroup(1, [l1], tag=1)
    gmsh.model.setPhysicalName(1, 1, "Entree")

    gmsh.model.addPhysicalGroup(1, [l2], tag=2)
    gmsh.model.setPhysicalName(1, 2, "Wall")

    gmsh.model.addPhysicalGroup(1, [l3], tag=3)
    gmsh.model.setPhysicalName(1, 3, "Sortie")

    gmsh.model.addPhysicalGroup(1, [l4], tag=4)
    gmsh.model.setPhysicalName(1, 4, "Axis r=0")

    gmsh.model.addPhysicalGroup(2, [surf], tag=10)
    gmsh.model.setPhysicalName(2, 10, "Fluid")

    # -----------------------------
    # Mesh
    # -----------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # -----------------------------
    # Extract mesh (comme ton code)
    # -----------------------------
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    # -----------------------------
    # Extract boundaries
    # -----------------------------
    boundary_names = ["Entree", "Wall", "Sortie", "Axis r=0"]

    phys_groups = gmsh.model.getPhysicalGroups(1)

    name_to_tag = {}
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if name:
            name_to_tag[name] = tag

    bnds = []
    bnds_tags = []

    for name in boundary_names:
        tag = name_to_tag[name]
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag)
        tags = np.unique(np.asarray(tags, dtype=int))

        bnds.append((name, 1))
        bnds_tags.append(tags)

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags

def _extract_boundaries(boundary_names):
    """
    Helper interne : extrait les tags de noeuds gmsh pour une liste de noms
    de groupes physiques 1D. Retourne (bnds, bnds_tags) dans le meme format
    que build_conduit_mesh.
    """
    import gmsh
    import numpy as np

    phys_groups = gmsh.model.getPhysicalGroups(1)
    name_to_tag = {}
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if name:
            name_to_tag[name] = tag

    bnds      = []
    bnds_tags = []
    for name in boundary_names:
        tag = name_to_tag[name]
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag)
        tags = np.unique(np.asarray(tags, dtype=int))
        bnds.append((name, 1))
        bnds_tags.append(tags)

    return bnds, bnds_tags


def build_axi_reservoir_mesh(
    R_res=0.04,
    R_pipe=0.01,
    L_res=0.08,
    L_pipe=0.16,
    lc_res=0.004,
    lc_pipe=0.001,
    order=1,
):
    """
    Build a 2D axisymmetric mesh for two cylindrical chambers connected
    by a narrow tube, in meridional coordinates (r, z).
 
    The domain is the meridional half-plane r >= 0. The axis r=0 is a
    natural Neumann condition (symmetry of revolution) — no assembly needed.
 
    Geometry (8 corners, same H-shape as build_two_reservoir_mesh but
    now interpreted as (r=y, z=x) in axisymmetric coordinates):
 
        r=R_res  p8──────p7          p4──────p3
                 │ CHAMB  │          │ CHAMB  │
        r=R_pipe │ GAUCHE p6────────p5 DROITE │
                 │                            │
        r=0      p1──────────────────────────p2
                 z=0    z=L1  z=L1+L2   z=L1+L2+L3
 
    Convention: x[0] = r (radial), x[1] = z (axial)
    This matches the existing assemblers (stiffness, mass, Robin, Neumann)
    which already include the axisymmetric factor r = x[0].
 
    Boundary conditions:
        "Entree"            z=0,          r in [0, R_res]  : Neumann inflow q_in
        "Sortie"            z=L_tot,      r in [0, R_res]  : Neumann outflow (homogeneous)
        "Wall_pipe"         r=R_pipe, z in [L_res, L_res+L_pipe] : Robin (heating)
        "Axis"              r=0                             : symmetry (natural)
        "Wall_res_left"     r=R_res,  z in [0, L_res]      : adiabatic (natural)
        "Wall_res_right"    r=R_res,  z in [L_res+L_pipe, L_tot] : adiabatic (natural)
        "Contraction_left"  z=L_res,  r in [R_pipe, R_res] : adiabatic (natural)
        "Contraction_right" z=L_res+L_pipe, r in [R_pipe, R_res] : adiabatic (natural)
 
    All adiabatic walls and the axis are natural Neumann → no assembly needed.
    Only "Entree", "Sortie", and "Wall_pipe" need explicit assembly.
 
    Parameters
    ----------
    R_res   : float  Radius of the cylindrical chambers
    R_pipe  : float  Radius of the connecting tube  (R_pipe < R_res)
    L_res   : float  Axial length of each chamber
    L_pipe  : float  Axial length of the tube
    lc_res  : float  Characteristic mesh size in the chambers
    lc_pipe : float  Characteristic mesh size in the tube
    order   : int    FE polynomial order (1 or 2)
 
    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
    (same format as build_conduit_mesh)
 
    Boundary names (in order):
        "Entree", "Sortie", "Wall_pipe", "Axis",
        "Wall_res_left", "Wall_res_right",
        "Contraction_left", "Contraction_right"
    """
    import gmsh
    import numpy as np
 
    gmsh.initialize()
    gmsh.model.add("two_chamber_axi")
 
    L_tot = 2.0 * L_res + L_pipe
 
    # ------------------------------------------------------------------
    # 8 corner points  (r, z) — counter-clockwise from bottom-left
    #
    #   "bottom" = axis r=0
    #   "top"    = outer wall r=R_res
    #   "left"   = inlet z=0
    #   "right"  = outlet z=L_tot
    #
    #   p1 = (r=0,      z=0     )   axis,  inlet
    #   p2 = (r=0,      z=L_tot )   axis,  outlet
    #   p3 = (r=R_res,  z=L_tot )   wall,  outlet
    #   p4 = (r=R_res,  z=L1+L2 )   wall,  right contraction top
    #   p5 = (r=R_pipe, z=L1+L2 )   tube wall, right
    #   p6 = (r=R_pipe, z=L1    )   tube wall, left
    #   p7 = (r=R_res,  z=L1    )   wall,  left contraction top
    #   p8 = (r=R_res,  z=0     )   wall,  inlet
    # ------------------------------------------------------------------
    p1 = gmsh.model.geo.addPoint(0.0,    0.0,            0.0, lc_res)
    p2 = gmsh.model.geo.addPoint(0.0,    L_tot,          0.0, lc_res) 
    p3 = gmsh.model.geo.addPoint(R_res,  L_tot,          0.0, lc_res) 
    p4 = gmsh.model.geo.addPoint(R_res,  L_res + L_pipe, 0.0, lc_res)   
    p5 = gmsh.model.geo.addPoint(R_pipe, L_res + L_pipe, 0.0, lc_pipe)  
    p6 = gmsh.model.geo.addPoint(R_pipe, L_res,          0.0, lc_pipe)  
    p7 = gmsh.model.geo.addPoint(R_res,  L_res,          0.0, lc_res)   
    p8 = gmsh.model.geo.addPoint(R_res,  0.0,            0.0, lc_res)   
 
    # ------------------------------------------------------------------
    # 8 boundary lines (counter-clockwise)
    # ------------------------------------------------------------------
    l_axis              = gmsh.model.geo.addLine(p1, p2)   # r=0      (axis)
    l_sortie            = gmsh.model.geo.addLine(p2, p3)   # z=L_tot  (outlet)
    l_wall_res_right    = gmsh.model.geo.addLine(p3, p4)   # r=R_res, right chamber
    l_contraction_right = gmsh.model.geo.addLine(p4, p5)   # z=L1+L2  (step wall)
    l_wall_pipe         = gmsh.model.geo.addLine(p5, p6)   # r=R_pipe (Robin wall)
    l_contraction_left  = gmsh.model.geo.addLine(p6, p7)   # z=L1     (step wall)
    l_wall_res_left     = gmsh.model.geo.addLine(p7, p8)   # r=R_res, left chamber
    l_entree            = gmsh.model.geo.addLine(p8, p1)   # z=0      (inlet)
 
    # ------------------------------------------------------------------
    # Single closed surface
    # ------------------------------------------------------------------
    cloop = gmsh.model.geo.addCurveLoop([
        l_axis,
        l_sortie,
        l_wall_res_right,
        l_contraction_right,
        l_wall_pipe,
        l_contraction_left,
        l_wall_res_left,
        l_entree,
    ])
    surf = gmsh.model.geo.addPlaneSurface([cloop])
    gmsh.model.geo.synchronize()
 
    # ------------------------------------------------------------------
    # Physical groups
    # ------------------------------------------------------------------
    groups = {
        "Entree":            ([l_entree],            1),
        "Sortie":            ([l_sortie],            2),
        "Wall_pipe":         ([l_wall_pipe],         3),
        "Axis":              ([l_axis],              4),
        "Wall_res_left":     ([l_wall_res_left],     5),
        "Wall_res_right":    ([l_wall_res_right],    6),
        "Contraction_left":  ([l_contraction_left],  7),
        "Contraction_right": ([l_contraction_right], 8),
    }
    for name, (lines, tag) in groups.items():
        gmsh.model.addPhysicalGroup(1, lines, tag=tag)
        gmsh.model.setPhysicalName(1, tag, name)
 
    gmsh.model.addPhysicalGroup(2, [surf], tag=10)
    gmsh.model.setPhysicalName(2, 10, "Fluid")
 
    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
 
    # ------------------------------------------------------------------
    # Extract mesh arrays
    # ------------------------------------------------------------------
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)
 
    boundary_names = [
        "Entree", "Sortie", "Wall_pipe", "Axis",
        "Wall_res_left", "Wall_res_right",
        "Contraction_left", "Contraction_right",
    ]
    bnds, bnds_tags = _extract_boundaries(boundary_names)
 
    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags

def mirror_axi_solution(nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof):
    """
    Construit le domaine complet r in [-R_res, R_res] par symétrie miroir
    du maillage axisymétrique r >= 0, pour la visualisation uniquement.

    Le maillage de calcul (r >= 0) est dupliqué en miroir (r <= 0).
    Les noeuds sur l'axe r=0 sont partagés. Le champ U est copié
    symétriquement (T(-r,z) = T(r,z) par axisymétrie).

    Parameters
    ----------
    nodeCoords   : ndarray flattened, coordonnées gmsh (x,y,z) de tous les noeuds
    elemNodeTags : ndarray flattened, connectivité des éléments
    nodeTags     : ndarray, tags gmsh des noeuds
    U            : ndarray (num_dofs,), solution FEM
    tag_to_dof   : ndarray, mapping tag gmsh -> indice compact

    Returns
    -------
    x_full  : ndarray (N_full,)  coordonnées r du domaine complet
    y_full  : ndarray (N_full,)  coordonnées z du domaine complet
    tri_full: ndarray (M_full,3) triangles du domaine complet (indices dans x_full)
    U_full  : ndarray (N_full,)  valeurs de T sur le domaine complet
    """
    import numpy as np

    all_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    nodeTags   = np.asarray(nodeTags,   dtype=int)

    # --- reconstruction coordonnées et valeurs sur r >= 0
    num_dofs = int(np.max(tag_to_dof[tag_to_dof >= 0]) + 1)
    r_axi = np.zeros(num_dofs)
    z_axi = np.zeros(num_dofs)

    for i, tag in enumerate(nodeTags):
        dof = tag_to_dof[int(tag)]
        if dof >= 0:
            r_axi[dof] = all_coords[i, 0]
            z_axi[dof] = all_coords[i, 1]

    # --- triangles du côté r >= 0
    ne   = len(elemNodeTags) // 3
    conn = np.asarray(elemNodeTags, dtype=int).reshape(ne, -1)
    # prendre les 3 premiers noeuds (coins) et convertir en indices compacts
    tri_axi = tag_to_dof[conn[:, :3]]

    # --- noeuds sur l'axe r=0
    on_axis = (np.abs(r_axi) < 1e-12)

    # --- côté miroir r <= 0
    # les noeuds hors axe sont dupliqués, les noeuds sur l'axe sont partagés
    # mapping : indice axi -> indice dans le domaine complet
    n_axi    = num_dofs
    # noeuds hors axe reçoivent un nouvel indice
    off_axis = ~on_axis
    n_off    = int(np.sum(off_axis))

    # indices dans x_full :
    #   0 .. n_axi-1          : côté r >= 0 (original)
    #   n_axi .. n_axi+n_off-1 : côté r <= 0 (miroir, hors axe seulement)

    mirror_idx = np.full(n_axi, -1, dtype=int)
    counter = n_axi
    for i in range(n_axi):
        if off_axis[i]:
            mirror_idx[i] = counter
            counter += 1
        else:
            mirror_idx[i] = i   # noeud sur l'axe : même indice

    n_full = counter

    # --- construction des tableaux complets
    x_full = np.zeros(n_full)
    y_full = np.zeros(n_full)
    U_full = np.zeros(n_full)

    # côté r >= 0
    x_full[:n_axi] = r_axi
    y_full[:n_axi] = z_axi
    U_full[:n_axi] = U

    # côté r <= 0 (miroir, hors axe)
    for i in range(n_axi):
        if off_axis[i]:
            j = mirror_idx[i]
            x_full[j] = -r_axi[i]   # symétrie r -> -r
            y_full[j] =  z_axi[i]
            U_full[j] =  U[i]        # T(-r,z) = T(r,z)

    # --- triangles côté r >= 0 (déjà en indices compacts)
    tri_right = tri_axi.copy()

    # --- triangles côté r <= 0 (miroir)
    # chaque triangle (a,b,c) devient (mirror(a), mirror(b), mirror(c))
    # et on inverse l'orientation pour que les normales pointent vers l'extérieur
    tri_left = np.zeros_like(tri_right)
    for k in range(3):
        tri_left[:, k] = mirror_idx[tri_right[:, k]]
    tri_left = tri_left[:, ::-1]   # inversion orientation

    tri_full = np.vstack([tri_right, tri_left])

    return x_full, y_full, tri_full, U_full


def plot_full_reservoir(ax, nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof,
                        vmin=None, vmax=None, cmap='hot_r', swap_axes=False):
    """
    Affiche la coupe complète du réservoir (r in [-R_res, R_res])
    par miroir du maillage axisymétrique.

    Parameters
    ----------
    ax          : matplotlib Axes
    nodeCoords, elemNodeTags, nodeTags, tag_to_dof : sorties de build_axi_reservoir_mesh
    U           : ndarray (num_dofs,), solution FEM
    vmin, vmax  : bornes colorbar (None = auto)
    cmap        : colormap matplotlib

    Returns
    -------
    contour : l'objet tricontourf (pour fig.colorbar)
    """
    x_full, y_full, tri_full, U_full = mirror_axi_solution(
        nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof
    )
 
    vmin_eff = vmin if vmin is not None else float(U_full.min())
    vmax_eff = vmax if vmax is not None else float(U_full.max())
 
    xa, ya = (y_full, x_full) if swap_axes else (x_full, y_full)
    xlabel  = "z [m]"         if swap_axes else "r [m]"
    ylabel  = "r [m]"         if swap_axes else "z [m]"
 
    contour = ax.tricontourf(xa, ya, tri_full, U_full,
                             levels=100, cmap=cmap,
                             vmin=vmin_eff, vmax=vmax_eff)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return contour
 

def revolve_axi_solution_3d(nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof,
                             n_sectors=60):
    """
    Build a 3D surface mesh by revolving the 2D axisymmetric solution
    around the z-axis, for 3D visualisation only.

    The 2D meridional mesh (r >= 0) is swept through n_sectors angular
    slices covering [0, 2π].  Each 2D triangle (a, b, c) becomes a
    triangular prism whose two triangular faces contribute to the surface.

    Parameters
    ----------
    nodeCoords   : ndarray flattened  – gmsh node coordinates
    elemNodeTags : ndarray flattened  – element connectivity
    nodeTags     : ndarray            – gmsh node tags
    U            : ndarray (num_dofs,) – FEM solution
    tag_to_dof   : ndarray            – gmsh tag -> compact DOF index
    n_sectors    : int                – number of angular divisions (default 60)

    Returns
    -------
    X, Y, Z  : ndarray (N_3d,)   – 3D Cartesian coordinates of all nodes
    tri3d    : ndarray (M_3d, 3) – triangle connectivity (indices into X/Y/Z)
    U3d      : ndarray (N_3d,)   – solution value at each 3D node
    """
    import numpy as np

    all_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    nodeTags   = np.asarray(nodeTags,   dtype=int)

    # ------------------------------------------------------------------ #
    # 1.  Recover (r, z) and U on the meridional half-plane              #
    # ------------------------------------------------------------------ #
    num_dofs = int(np.max(tag_to_dof[tag_to_dof >= 0]) + 1)
    r_axi = np.zeros(num_dofs)
    z_axi = np.zeros(num_dofs)

    for i, tag in enumerate(nodeTags):
        dof = tag_to_dof[int(tag)]
        if dof >= 0:
            r_axi[dof] = all_coords[i, 0]
            z_axi[dof] = all_coords[i, 1]

    # 2D triangles in compact DOF indices
    ne   = len(elemNodeTags) // 3
    conn = np.asarray(elemNodeTags, dtype=int).reshape(ne, -1)
    tri2d = tag_to_dof[conn[:, :3]]          # shape (ne, 3)

    # ------------------------------------------------------------------ #
    # 2.  Revolve: one layer of 3D nodes per angular sector              #
    # ------------------------------------------------------------------ #
    angles = np.linspace(0.0, 2.0 * np.pi, n_sectors, endpoint=False)

    # 3D node array: sector k, dof i  →  flat index k*num_dofs + i
    n3d = n_sectors * num_dofs
    X   = np.zeros(n3d)
    Y   = np.zeros(n3d)
    Z   = np.zeros(n3d)
    U3d = np.zeros(n3d)

    for k, theta in enumerate(angles):
        base = k * num_dofs
        X[base:base + num_dofs] = r_axi * np.cos(theta)
        Y[base:base + num_dofs] = r_axi * np.sin(theta)
        Z[base:base + num_dofs] = z_axi
        U3d[base:base + num_dofs] = U

    # ------------------------------------------------------------------ #
    # 3.  Build triangular faces from prisms between adjacent sectors    #
    #                                                                     #
    #  For each 2D triangle (a, b, c) and two consecutive sectors k, k+1 #
    #  we get a triangular prism with 6 nodes:                           #
    #    a0=k*n+a,  b0=k*n+b,  c0=k*n+c                                 #
    #    a1=k1*n+a, b1=k1*n+b, c1=k1*n+c                               #
    #  Split each quad face into 2 triangles → 2 tris per quad face,    #
    #  or simply output the two end-cap triangles (gives a closed solid). #
    #                                                                     #
    #  Here we keep ALL faces (2 end-caps + 3 side quads split in 2)    #
    #  so the surface is water-tight and shows interior colour variation. #
    # ------------------------------------------------------------------ #
    faces = []

    n = num_dofs   # shorthand

    for k in range(n_sectors):
        k1  = (k + 1) % n_sectors
        off = k  * n
        off1= k1 * n

        a0, b0, c0 = tri2d[:, 0] + off,  tri2d[:, 1] + off,  tri2d[:, 2] + off
        a1, b1, c1 = tri2d[:, 0] + off1, tri2d[:, 1] + off1, tri2d[:, 2] + off1

        # side quad a0-b0 / b0-b1 / b1-a1  → 2 triangles
        faces.append(np.stack([a0, b0, b1], axis=1))
        faces.append(np.stack([a0, b1, a1], axis=1))

        # side quad b0-c0 / c0-c1 / c1-b1
        faces.append(np.stack([b0, c0, c1], axis=1))
        faces.append(np.stack([b0, c1, b1], axis=1))

        # side quad c0-a0 / a0-a1 / a1-c1
        faces.append(np.stack([c0, a0, a1], axis=1))
        faces.append(np.stack([c0, a1, c1], axis=1))

    tri3d = np.vstack(faces)   # shape (6*n_sectors*ne, 3)

    return X, Y, Z, tri3d, U3d


def plot_full_reservoir_3d(ax, nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof,
                       n_sectors=60, cmap='plasma', vmin=None, vmax=None,
                       alpha=0.9):
    """
    Plot the 3D revolution surface of the axisymmetric FEM solution.

    Parameters
    ----------
    ax          : Axes3D  (created with projection='3d')
    nodeCoords, elemNodeTags, nodeTags, tag_to_dof : from build_*_mesh
    U           : ndarray (num_dofs,) – FEM solution
    n_sectors   : angular resolution (default 60)
    cmap        : matplotlib colormap
    vmin, vmax  : colorbar bounds (None = auto)
    alpha       : surface transparency

    Returns
    -------
    surf : the Poly3DCollection object (for fig.colorbar)

    Example usage
    -------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    surf = plot_3d_revolution(ax, nodeCoords, elemNodeTags, nodeTags,
                              U, tag_to_dof, n_sectors=60, cmap='plasma',
                              vmin=T_in, vmax=T_ext)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='T [K]')
    plt.show()
    """
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    X, Y, Z, tri3d, U3d = revolve_axi_solution_3d(
        nodeCoords, elemNodeTags, nodeTags, U, tag_to_dof, n_sectors=n_sectors
    )

    vmin_eff = vmin if vmin is not None else float(U3d.min())
    vmax_eff = vmax if vmax is not None else float(U3d.max())

    norm      = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    colormap  = cm.get_cmap(cmap)

    # Average U over each triangle face for face colour
    U_face = U3d[tri3d].mean(axis=1)
    face_colors = colormap(norm(U_face))

    verts = np.stack([X[tri3d], Y[tri3d], Z[tri3d]], axis=-1)  # (M, 3, 3)

    poly = Poly3DCollection(verts, facecolors=face_colors,
                            edgecolors='none', alpha=alpha)
    ax.add_collection3d(poly)

    # Axis limits
    R_max = np.max(np.sqrt(X**2 + Y**2))
    ax.set_xlim(-R_max, R_max)
    ax.set_ylim(-R_max, R_max)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # Return a ScalarMappable for colorbar compatibility
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    return sm