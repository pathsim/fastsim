"""Symbolic @ symbolic matmul: a traced matrix times a traced vector/matrix
(e.g. a state-dependent mass matrix ``M(x) @ v``). The constant-matrix path
(``A @ x`` with concrete ``A``) is covered elsewhere; this pins the all-symbolic
1-D/2-D combinations against numpy."""
import numpy as np
from fastsim.jit import jit


def test_symbolic_matrix_vector():
    def f(x):
        M = np.array([[x[0], x[1]], [x[1], x[0]]])  # symbolic 2x2
        v = np.array([x[0], x[1]])
        return M @ v
    g = jit(f, n_x=2)
    for xv in ([2.0, 3.0], [1.0, -4.0], [0.5, 0.5]):
        exp = np.array([[xv[0], xv[1]], [xv[1], xv[0]]]) @ np.array(xv)
        assert np.allclose(np.ravel(g(xv)), exp)


def test_symbolic_dot():
    g = jit(lambda x: [np.dot(np.array([x[0], x[1], x[2]]),
                              np.array([x[2], x[0], x[1]]))], n_x=3)
    xv = [2.0, 3.0, 4.0]
    assert abs(float(np.ravel(g(xv))[0]) - (2*4 + 3*2 + 4*3)) < 1e-9


def test_symbolic_matrix_matrix():
    def h(x):
        A = np.array([[x[0], x[1]], [x[2], x[3]]])
        B = np.array([[x[3], x[2]], [x[1], x[0]]])
        return (A @ B).reshape(-1)
    g = jit(h, n_x=4)
    xv = [1.0, 2.0, 3.0, 4.0]
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[4.0, 3.0], [2.0, 1.0]])
    assert np.allclose(np.ravel(g(xv)), (A @ B).reshape(-1))


def test_symbolic_vector_matrix():
    def h(x):
        v = np.array([x[0], x[1]])
        B = np.array([[x[0], x[1]], [x[1], x[0]]])
        return v @ B
    g = jit(h, n_x=2)
    xv = [2.0, 5.0]
    v = np.array(xv)
    B = np.array([[xv[0], xv[1]], [xv[1], xv[0]]])
    assert np.allclose(np.ravel(g(xv)), v @ B)
