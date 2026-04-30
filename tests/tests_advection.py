import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Advection import assemble_advection
from gmsh_utils import build_conduit_mesh
from gmsh_utils import *
import gmsh
from scipy.sparse import issparse
import numpy as np
import pytest


"""
Test 1: Retourne correctement, vérifie le type et la taille
"""
def test_taille_type():

    elemTags = [1,2]
    conn = [0, 1, 1, 2]
    jac = [1,0,0, 0,1,0, 0,0,1,1,0,0, 0,1,0, 0,0,1]
    det = [1.0, 1.0]
    xphys = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    w = [1.0]
    N = [0.5, 0.5]
    gN = [1,0,0,-1,0,0]
    def beta_fun(a):
        return [0.0, 0.0, 0.0]
    tag_to_dof = np.array([0, 1, 2])
    taille = len(tag_to_dof)

    matrice_C = assemble_advection(elemTags, conn, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof)

    assert matrice_C.shape == (taille,taille) #Vérifie la taille
    assert issparse(matrice_C) #Vérifie que C est bien creuse

""""
Test 2: Sans vitesse, la matrice doit etre nulle
"""
def test_sans_vitesse():

    elemTags = [1]
    conn = [0, 1]
    jac = np.eye(3).flatten()
    det = [1.0]
    xphys = [1.0, 0.0, 0.0]
    w = [1.0]
    N = [0.5, 0.5]
    gN = [1,0,0,-1,0,0]
    def beta_fun(a):
        return [0.0, 0.0, 0.0]
    tag_to_dof = np.array([0, 1])

    matrice_C = assemble_advection(elemTags, conn, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof)

    assert np.allclose(matrice_C.toarray(), 0.0)

""""
Test 3: Test avec le maillage du gmsh. 
"""
def test_maillage():
    gmsh_init()
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, _, _ = build_conduit_mesh()

    xi, w, N, gN = prepare_quadrature_and_basis(elemType,1)
    jac, det, xphys = get_jacobians(elemType, xi)

    tag_to_dof = -np.ones(max(nodeTags) + 1, dtype=int)
    for i, tag in enumerate(nodeTags):
        tag_to_dof[tag] = i

        def beta_fun(a):
            return [1.0, 0.0, 0.0]

    matrice_C = assemble_advection(elemTags, elemNodeTags, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof)

    assert np.linalg.norm(matrice_C.toarray()) > 0
    assert np.isfinite(matrice_C.toarray()).all()

    gmsh_finalize()


"""
Test 4: Champ constant, l'advection doit etre nulle (pas de transport). Avec gmsh.
"""
def test_champ_cst():
    gmsh.initialize()

    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, _, _ = build_conduit_mesh()

    xi, w, N, gN = prepare_quadrature_and_basis(elemType,1)
    jac, det, xphys = get_jacobians(elemType, xi)

    tag_to_dof = -np.ones(max(nodeTags) + 1, dtype=int)
    for i, tag in enumerate(nodeTags):
        tag_to_dof[tag] = i

    def beta_fun(a):
        return [1.0, 0.0, 0.0]

    matrice_C = assemble_advection(elemTags, elemNodeTags, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof)

    ones = np.ones(len(nodeTags))
    resulat = matrice_C @ ones

    assert np.allclose(resulat, 0.0, atol=1e-12)

    gmsh.finalize()

"""
Test 5: Avec une vitesse et gradient perpandiculaire, la matrice d'advection doit etre nulle
""" 
def test_beta_gradient_perpandiculaire():
    elemTags = [1]
    conn = [0, 1]
    jac = np.eye(3).flatten()
    det = [1.0]
    xphys = [1.0, 0.0, 0.0]
    w = [1.0]
    N = [0.5, 0.5]
    gN = [1,0,0,-1,0,0]
    def beta_fun(a):
        return [0.0, 1.0, 0.0]
    tag_to_dof = np.array([0, 1])
    
    matrice_C = assemble_advection(elemTags, conn, jac, det, xphys, w, N, gN, beta_fun, tag_to_dof)

    assert np.allclose(matrice_C.toarray(), 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])