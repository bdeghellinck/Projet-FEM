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