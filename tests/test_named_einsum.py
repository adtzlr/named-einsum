import numpy as np

from neinsum import named_einsum


def test_named_einsum():
    # input array
    x = np.eye(3)

    # output array
    y = np.zeros((3, 3, 3, 3))

    # test with unicode keys
    u = named_einsum("α_ij,β_kl")(α=x, β=x)

    # test with output destination and whitespace
    v = named_einsum("A_ij, B_kl")(A=x, B=x, out=y)

    # test with named output destination and whitespaces
    w = named_einsum("A_ij, B_kl ->  C_ijkl")(A=x, B=x, C=y)

    # test with default out-argument in explicit mode
    r = named_einsum("A_ij,B_kl->C_ijkl")(A=x, B=x, out=y)

    assert v is y
    assert w is y
    assert r is y
    assert np.allclose(u, v)


if __name__ == "__main__":
    test_named_einsum()
