import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import (
    gmsh_finalize,
    build_two_reservoir_mesh,
    prepare_quadrature_and_basis,
    get_jacobians,
    border_dofs_from_tags,
)
from plot_utils import plot_mesh_2d


def main():
    parser = argparse.ArgumentParser(description="Verification du maillage deux reservoirs")

    parser.add_argument("-order",    type=int,   default=1,    help="Ordre des elements finis")
    parser.add_argument("--H",       type=float, default=0.04, help="Demi-hauteur des reservoirs")
    parser.add_argument("--h_pipe",  type=float, default=0.01, help="Demi-hauteur du tuyeau")
    parser.add_argument("--L_res",   type=float, default=0.08, help="Longueur d un reservoir")
    parser.add_argument("--L_pipe",  type=float, default=0.16, help="Longueur du tuyeau")
    parser.add_argument("--lc_res",  type=float, default=0.004, help="Taille maille reservoirs")
    parser.add_argument("--lc_pipe", type=float, default=0.001, help="Taille maille tuyeau")

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

    print(f"\nMaillage genere :")
    print(f"  Noeuds   : {len(nodeTags)}")
    print(f"  Elements : {len(elemTags)}")
    print(f"\nBords :")
    for (name, _), tags in zip(bnds, bnds_tags):
        print(f"  {name:25s} : {len(tags):4d} noeuds")

    # ------------------------------------------------------------
    # 2) Mapping tag -> dof  (bloc standard, identique au main original)
    # ------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    max_tag = int(np.max(nodeTags))
    all_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)

    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    node_index_from_tag = np.full(max_tag + 1, -1, dtype=int)
    for i_node, tag in enumerate(nodeTags):
        node_index_from_tag[int(tag)] = i_node

    dof_coords = np.zeros((len(unique_dofs_tags), 3), dtype=float)
    for i_dof, tag in enumerate(unique_dofs_tags):
        tag_int = int(tag)
        tag_to_dof[tag_int] = i_dof
        dof_coords[i_dof] = all_coords[node_index_from_tag[tag_int]]

    # ------------------------------------------------------------
    # 3) Recuperer les dofs de chaque bord et verifier les coordonnees
    # ------------------------------------------------------------
    boundary_tags = {name: bnds_tags[i] for i, (name, _) in enumerate(bnds)}

    checks = {
        "Entree"            : ("x", 0.0),
        "Sortie"            : ("x", 2*args.L_res + args.L_pipe),
        "Wall_pipe"         : ("y", args.h_pipe),
        "Sym"               : ("y", 0.0),
        "Wall_res_left"     : ("y", args.H),
        "Wall_res_right"    : ("y", args.H),
        "Contraction_left"  : ("x", args.L_res),
        "Contraction_right" : ("x", args.L_res + args.L_pipe),
    }
    coord_idx = {"x": 0, "y": 1}

    print(f"\nVerification coordonnees des bords :")
    all_ok = True
    for name, (axis, expected) in checks.items():
        dofs = border_dofs_from_tags(boundary_tags[name], tag_to_dof)
        vals = dof_coords[dofs, coord_idx[axis]]
        ok = np.allclose(vals, expected, atol=1e-10)
        status = "OK" if ok else f"ERREUR (min={vals.min():.4f}, max={vals.max():.4f})"
        print(f"  {name:25s} {axis}={expected:.4f}  ->  {status}")
        if not ok:
            all_ok = False

    print(f"\n{'Tous les bords sont corrects !' if all_ok else 'Des erreurs ont ete detectees.'}")

    # ------------------------------------------------------------
    # 4) Visualisation du maillage avec les bords colories
    # ------------------------------------------------------------
    plot_mesh_2d(
        elemType=elemType,
        nodeTags=nodeTags,
        nodeCoords=nodeCoords,
        elemTags=elemTags,
        elemNodeTags=elemNodeTags,
        bnds=bnds,
        bnds_tags=bnds_tags,
    )

    # ------------------------------------------------------------
    # 5) Figure supplementaire : bords colories manuellement
    #    (pour voir exactement quels noeuds appartiennent a quoi)
    # ------------------------------------------------------------
    colors = {
        "Entree"            : "#2471a3",
        "Sortie"            : "#e74c3c",
        "Wall_pipe"         : "#c0392b",
        "Sym"               : "#117a65",
        "Wall_res_left"     : "#7f8c8d",
        "Wall_res_right"    : "#7f8c8d",
        "Contraction_left"  : "#8e44ad",
        "Contraction_right" : "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # Triangles en gris clair
    import matplotlib.tri as mtri
    ne = len(elemTags)
    conn = np.array(elemNodeTags, dtype=int).reshape(ne, -1)
    n2i = tag_to_dof  # reuse: tag -> compact dof index
    tris = np.vectorize(lambda t: node_index_from_tag[t])(conn[:, :3])
    # use raw node indices for triplot (nodeCoords order)
    raw_x = all_coords[:, 0]
    raw_y = all_coords[:, 1]
    ax.triplot(raw_x, raw_y, tris, 'k-', lw=0.3, alpha=0.4, zorder=1)

    # Noeuds de chaque bord
    for (name, _), tags in zip(bnds, bnds_tags):
        dofs = border_dofs_from_tags(tags, tag_to_dof)
        ax.scatter(dof_coords[dofs, 0], dof_coords[dofs, 1],
                   s=8, color=colors[name], label=name, zorder=5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Maillage deux reservoirs — verification des bords")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig("mesh_verification.png", dpi=150)
    plt.show()
    print("Figure sauvegardee : mesh_verification.png")

    gmsh_finalize()


if __name__ == "__main__":
    main()