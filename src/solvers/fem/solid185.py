"""The ANSYS SOLID185 hexahedron on a unit voxel in its B-bar form (KEYOPT(2) = 0): the strain-displacement matrix at the 8 Gauss points.

    Node order   the corners of a voxel from its lowest corner, as ANSYS I, J, K, L, M, N, O, P, with (ox, oy, oz) the offset in voxels along x, y, z:
                 0 (0,0,0), 1 (1,0,0), 2 (1,1,0), 3 (0,1,0), 4 (0,0,1), 5 (1,0,1), 6 (1,1,1), 7 (0,1,1). `NODE_OFFSETS`.
    DOF order    node-major: u[3 a + c], node a in the order above, c = 0, 1, 2 for ux, uy, uz. An element vector has 24 entries.
    Natural      (xi, eta, zeta) in [-1, 1] along x, y, z; node a sits at 2 (ox, oy, oz) - 1. On the unit voxel x = (1 + xi) / 2, so the Jacobian is I / 2
                 and det J = 1/8. Shape functions N_a = (1 + xi_a xi)(1 + eta_a eta)(1 + zeta_a zeta) / 8.
    Gauss        the 8 points (+-1, +-1, +-1) / sqrt(3), in the node order, each of weight 1, so each weighs det J = 1/8.
    Voigt        strain (exx, eyy, ezz, gxy, gyz, gxz) with engineering shear g = 2 e. The B matrix is Voigt: 6 x 24, rows in that order.
    B-bar        B_bar_g = B_g + (1/3) m (b_mean - b_g), m = (1, 1, 1, 0, 0, 0), b_g = m^T B_g the dilatation row at Gauss point g and b_mean its mean over
                 the element. The deviatoric strain stays that of B; the dilatation is the element mean at every point.
"""
import numpy as np

NODE_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
NODE_NATURAL = 2 * NODE_OFFSETS - 1
GAUSS = NODE_NATURAL / np.sqrt(3.0)


def _plain_B(points):
    t = 1.0 + NODE_NATURAL[None] * points[:, None, :]
    dN = np.empty(t.shape)
    for i, (j, k) in enumerate(((1, 2), (0, 2), (0, 1))):
        dN[..., i] = NODE_NATURAL[:, i] * t[..., j] * t[..., k] / 8.0
    dN *= 2.0
    B = np.zeros((len(points), 6, 24))
    for a in range(8):
        x, y, z = dN[:, a, 0], dN[:, a, 1], dN[:, a, 2]
        c = 3 * a
        B[:, 0, c], B[:, 1, c + 1], B[:, 2, c + 2] = x, y, z
        B[:, 3, c], B[:, 3, c + 1] = y, x
        B[:, 4, c + 1], B[:, 4, c + 2] = z, y
        B[:, 5, c], B[:, 5, c + 2] = z, x
    return B


def strain_matrix():
    """(8, 6, 24) Voigt strain-displacement matrix of the B-bar element at its 8 Gauss points: the dilatation is the mean over the element."""
    B = _plain_B(GAUSS)
    b_mean = _plain_B(GAUSS)[:, :3].sum(1).mean(0)
    B[:, :3] += (b_mean - B[:, :3].sum(1))[:, None, :] / 3.0
    return B
