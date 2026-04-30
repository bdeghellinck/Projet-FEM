import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Robin import assemble_robin_wall
import numpy as np
import pytest
from gmsh_utils import *

"""
Test 1: Rb et Fb comparé avec la formule 
"""
def test_2_points():

    wall_dofs = [0,1]

    dof_coords = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        ])

    h = 10000
    T_ext = 350

    Rb, Fb = assemble_robin_wall(wall_dofs, dof_coords, h, T_ext)

    ds = r0 = r1 = 1

    Rb_calc = h * 2*np.pi * ds / 12.0 * np.array([[3*r0 + r1, r0 + r1],[r0 + r1, r0 + 3*r1]])
    Fb_calc = h * T_ext * 2*np.pi * ds * np.array([(2*r0 + r1)/6,(r0 + 2*r1)/6])

    assert np.allclose(Rb.toarray(), Rb_calc)
    assert np.allclose(Fb, Fb_calc)

"""
Test 2: Rb sym et positif
"""
def test_sym_pos():

    wall_dofs = [0,1]

    dof_coords = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        ])

    h = 10000
    T_ext = 350

    Rb, Fb = assemble_robin_wall(wall_dofs, dof_coords, h, T_ext)

    assert np.allclose(Rb.toarray(), Rb.toarray().T)
    assert np.all(np.diag(Rb.toarray())>0) 


"""
Test 3: Sans échange thermique (h=0), les matrice sont nulle
"""
def test_h_nulle():

    wall_dofs = [0,1]

    dof_coords = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        ])

    h = 0
    T_ext = 350

    Rb, Fb = assemble_robin_wall(wall_dofs, dof_coords, h, T_ext)

    assert np.allclose(Rb.toarray(), 0)
    assert np.allclose(Fb, 0)

"""
Test 4: Test avec le maillage gmsh
"""
def test_maillage():
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = build_conduit_mesh(L=2.0, lc=0.5)

    dof_coords = np.array(nodeCoords).reshape(-1, 3)
    tag_to_dof = -np.ones(np.max(nodeTags)+1, dtype=int)
    for i, tag in enumerate(nodeTags):
        tag_to_dof[tag] = i

    for (name, _), tags in zip(bnds, bnds_tags):
        if name == "Wall":
            wall_tags = tags

    wall_dofs = tag_to_dof[wall_tags]


    h = 10000
    T_ext = 350

    Rb, Fb = assemble_robin_wall(wall_dofs, dof_coords, h, T_ext)

    assert Rb.shape[0] == len(dof_coords)
    assert np.linalg.norm(Fb) > 0

    gmsh_finalize()