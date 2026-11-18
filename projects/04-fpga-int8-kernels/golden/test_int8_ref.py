"""Unit tests for the golden model (run: pytest golden/)."""
import numpy as np
import pytest

from int8_ref import (dwconv1d_int8, fc_int8, fxp_mul, quantize_fc_layer,
                      requant_int8, rshift_round)


@pytest.mark.parametrize("x, s, expected", [
    (5, 1, 3),     # 2.5  -> 3   (half rounds up)
    (-5, 1, -2),   # -2.5 -> -2  (half rounds up, towards +inf)
    (-6, 1, -3),
    (127, 0, 127),
    (4096, 7, 32),   # 0.5*0.5 in Q1.7: 64*64 -> 32
])
def test_rshift_round(x, s, expected):
    assert int(rshift_round(x, s)) == expected


def test_requant_saturates():
    assert requant_int8(10**6, 4) == 127
    assert requant_int8(-(10**6), 4) == -128


def test_fxp_mul_q17():
    assert fxp_mul(64, 64, 7) == 32          # 0.5 * 0.5 = 0.25
    assert fxp_mul(-128, -128, 7) == 127     # (-1) * (-1) = 1 saturates to 127/128


def test_fc_matches_float_maths():
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 128, (50, 16)).astype(np.int8)
    w = rng.integers(-128, 128, (8, 16)).astype(np.int8)
    b = rng.integers(-4096, 4096, 8).astype(np.int32)
    shift = 8
    y = fc_int8(x, w, b, shift).astype(np.int64)
    exact = (x.astype(np.float64) @ w.T.astype(np.float64) + b) / 2**shift
    expected = np.clip(np.floor(exact + 0.5), -128, 127)
    assert np.array_equal(y, expected)


def test_dwconv_matches_numpy_correlate():
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 128, (3, 4, 20)).astype(np.int8)
    w = rng.integers(-128, 128, (4, 3)).astype(np.int8)
    b = rng.integers(-1000, 1000, 4).astype(np.int32)
    y = dwconv1d_int8(x, w, b, 7)
    for n in range(3):
        for c in range(4):
            acc = np.correlate(x[n, c].astype(np.int64), w[c].astype(np.int64), "valid") + b[c]
            assert np.array_equal(y[n, c], requant_int8(acc, 7))


def test_quantized_layer_is_close_to_float():
    rng = np.random.default_rng(3)
    w_f = rng.normal(0, 0.1, (8, 16))
    b_f = rng.normal(0, 0.05, 8)
    x_f = rng.uniform(-1, 1, (100, 16))
    x_scale = 1 / 127
    x_q = np.clip(np.round(x_f / x_scale), -127, 127).astype(np.int8)
    y_float = x_f @ w_f.T + b_f
    w_q, b_q, shift, y_scale = quantize_fc_layer(w_f, b_f, x_scale, np.abs(y_float).max())
    y_int = fc_int8(x_q, w_q, b_q, shift).astype(np.float64) * y_scale
    # Error stays within a couple of output steps (LSBs)
    assert np.max(np.abs(y_int - y_float)) < 2 * y_scale
