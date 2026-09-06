import numpy as np

from reference.coev_reference_ops import directed_area, product_integral, second_level_signature


def test_reference_square_and_reverse():
    square=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]],dtype=float)
    assert np.isclose(directed_area(square),1) and np.isclose(directed_area(square[::-1]),-1)


def test_reference_subdivision_and_product_integral():
    coarse=np.array([[0.,0.],[1.,2.]])
    fine=np.array([[0.,0.],[.25,.5],[.5,1.],[1.,2.]])
    c=second_level_signature(np.diff(coarse,axis=0)); f=second_level_signature(np.diff(fine,axis=0))
    assert np.allclose(c[0],f[0]) and np.allclose(c[1],f[1])
    assert np.isclose(product_integral(np.array([0.,1.]),coarse[:,0],coarse[:,1]),2/3)
