import numpy as np
import pytest

from bnn_ref import dot_pm1, dot_xnor_popcount, pack_pm1


@pytest.mark.parametrize("n", [32, 64, 256, 1024])
def test_xnor_popcount_equals_pm1_dot(n):
    rng = np.random.default_rng(n)
    for _ in range(50):
        a, w = rng.choice([-1, 1], n), rng.choice([-1, 1], n)
        assert dot_xnor_popcount(pack_pm1(a), pack_pm1(w)) == dot_pm1(a, w)


def test_extremes():
    ones = np.ones(64)
    assert dot_xnor_popcount(pack_pm1(ones), pack_pm1(ones)) == 64
    assert dot_xnor_popcount(pack_pm1(ones), pack_pm1(-ones)) == -64
