#!/usr/bin/env python3

"""
Test all Python examples from the cheatsheet
"""

import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve, minimize
import sympy as sp
import torch
import sys

def test_eigenvalues():
    """Test eigenvalue computation"""
    A = np.array([[1, 2], [3, 4]])
    eigvals = np.linalg.eigvals(A)
    expected = np.array([-0.37228132, 5.37228132])
    assert np.allclose(eigvals, expected), f"Eigenvalues failed: {eigvals}"
    print("✓ Eigenvalues test passed")

def test_ode_solving():
    """Test ODE solving with SciPy"""
    def f(t, y):
        return -2*y + np.sin(t)

    sol = solve_ivp(f, [0, 10], [1.0], dense_output=True)
    result = sol.sol(10.0)[0]
    expected = -0.04987190183538767
    assert np.isclose(result, expected, rtol=1e-5), f"ODE solving failed: {result}"
    print("✓ ODE solving test passed")

def test_ode_with_saveat():
    """Test ODE solving with specific time points"""
    def f(t, y):
        return -2*y + np.sin(t)

    t_eval = np.arange(0, 10.1, 0.1)
    sol = solve_ivp(f, [0, 10], [1.0], t_eval=t_eval)
    assert len(sol.t) == 101, f"Wrong number of time points: {len(sol.t)}"

    t_save = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    sol2 = solve_ivp(f, [0, 10], [1.0], t_eval=t_save)
    assert np.allclose(sol2.t, t_save), "Time points don't match"
    print("✓ ODE with saveat test passed")

def test_nonlinear_solving():
    """Test nonlinear equation solving"""
    def f(x):
        return x**3 - 2*x - 5

    result = fsolve(f, 2.0)[0]
    expected = 2.0945514815423265
    assert np.isclose(result, expected, rtol=1e-6), f"Nonlinear solving failed: {result}"
    print("✓ Nonlinear solving test passed")

def test_optimization():
    """Test optimization with SciPy"""
    def f(x):
        return x[0]**2 + x[1]**2

    result = minimize(f, [1.0, 1.0])
    assert np.allclose(result.x, [0.0, 0.0], atol=1e-6), f"Optimization failed: {result.x}"
    print("✓ Optimization test passed")

def test_automatic_differentiation():
    """Test automatic differentiation with PyTorch"""
    # Forward mode
    def f(x):
        return x**2 + torch.sin(x)

    x = torch.tensor(2.0)
    v = torch.tensor(1.0)
    with torch.enable_grad():
        _, jvp = torch.func.jvp(f, (x,), (v,))

    expected = 3.5838531634528574
    assert np.isclose(jvp.item(), expected, rtol=1e-6), f"Forward AD failed: {jvp.item()}"

    # Reverse mode (gradient)
    x = torch.tensor([2.0, 2.0], requires_grad=True)
    y = x[0]**2 + torch.sin(x[1])
    y.backward()
    grad = x.grad.numpy()

    assert np.isclose(grad[0], 4.0), f"Gradient[0] failed: {grad[0]}"
    assert np.isclose(grad[1], np.cos(2.0)), f"Gradient[1] failed: {grad[1]}"
    print("✓ Automatic differentiation test passed")

def test_symbolic_computing():
    """Test symbolic computing with SymPy"""
    x = sp.Symbol('x')
    expr = x**2 + sp.sin(x)
    derivative = sp.diff(expr, x)
    expected = "2*x + cos(x)"
    assert str(derivative) == expected, f"Symbolic diff failed: {derivative}"
    print("✓ Symbolic computing test passed")

def test_numerical_integration():
    """Test numerical integration"""
    def f(x):
        return np.exp(-x**2)

    result, error = quad(f, 0, 1)
    expected = 0.7468241328124271
    assert np.isclose(result, expected, rtol=1e-6), f"Integration failed: {result}"
    print("✓ Numerical integration test passed")

def test_rober_dae():
    """Test ROBER DAE problem (approximated as stiff ODE)"""
    def rober(t, y):
        y1, y2, y3 = y
        dy1 = -0.04*y1 + 1e4*y2*y3
        dy2 = 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2
        dy3 = -(dy1 + dy2)  # Enforce conservation
        return [dy1, dy2, dy3]

    sol = solve_ivp(rober, [0, 1e-5], [1, 0, 0], method='Radau', rtol=1e-4)
    final_state = sol.y[:, -1]

    # Check mass conservation
    assert np.isclose(sum(final_state), 1.0, atol=1e-6), f"Mass not conserved: {sum(final_state)}"
    assert final_state[0] > 0.99, f"Unexpected y1: {final_state[0]}"
    print("✓ ROBER DAE test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Python Examples from Scientific Modeling Cheatsheet")
    print("=" * 60)

    tests = [
        test_eigenvalues,
        test_ode_solving,
        test_ode_with_saveat,
        test_nonlinear_solving,
        test_optimization,
        test_automatic_differentiation,
        test_symbolic_computing,
        test_numerical_integration,
        test_rober_dae
    ]

    failed = 0
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("✅ All Python tests passed!")
    else:
        print(f"❌ {failed} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()