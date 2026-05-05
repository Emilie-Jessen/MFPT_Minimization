import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.integrate import solve_ivp, cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline


def pertubation_quad(s):
    return 4.0 * s * (1 - s)

def dpertubation_quad(s):
    return 4.0 - 8.0 * s

def U_quad(r, rn, A, alpha):
    """Calculate the potential at distance r."""
    return A / rn * r - alpha * pertubation_quad(r / rn)

def dU_quad(r, rn, A, alpha):
    """Calculate the derivative of the potential at distance r."""
    return (A - alpha * dpertubation_quad(r / rn)) / rn


def compute_for_constant_diffusion(r_target, rn, A, D0, alpha=0, ndim=3, N=4000):
    """
    Returns a callable T(r) evaluating
      T(r) = ∫_{r_target}^{r} (1/phi(y)) * [ ∫_{y}^{r_n} phi(z)/D(z) dz ] dy
    where phi solves phi'(r) = (mu(r)/D(r))*phi(r), phi(r_target)=1.
    """

    # definitions (vectorized-friendly)
    def pertubation(s):
        return 4.0 * s * (1.0 - s)

    def dpertubation(s):
        return 4.0 - 8.0 * s
    
    def U(rr):
        """Calculate the potential at distance r."""
        return A / rn * rr - alpha * pertubation(rr / rn) 

    def dU(rr):
        """Calculate the derivative of the potential at distance r."""
        return (A - alpha * dpertubation(rr / rn)) / rn

    def drift(rr):
        return (ndim - 1) * D0 / rr - D0 * dU(rr)

    # grid
    r_eval = np.linspace(r_target, rn, N)

    # ODE: phi' = (mu/D) * phi
    def rhs(x, phi):
        return (drift(x) / D0) * phi

    # solve_ivp(func, t_span, rhs(t_span[0]), t_eval=..., rtol=..., atol=...)
    # Instead of time, we use the spatial variable r here
    sol = solve_ivp(rhs, (r_target, rn), y0=[1.0], t_eval=r_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError("phi ODE failed: " + sol.message)
    phi = sol.y[0]

    # inner integral I(y) = ∫_{y}^{r_n} phi(z)/D(z) dz (right-to-left cumulative)
    inner_integrand = phi / D0
    reverse_inner_integral = cumulative_trapezoid(inner_integrand[::-1], r_eval[::-1], initial=0.0)  # ∫_{r_n}^{x} g
    inner_integral = -reverse_inner_integral[::-1]                                              # ∫_{x}^{r_n} g

    # outer integral F(r) = ∫_{r_target}^{r} [I(y)/phi(y)] dy
    outer_integrand = inner_integral / phi
    outer_integral = cumulative_trapezoid(outer_integrand, r_eval, initial=0.0)

    # spline for fast evaluation
    outer_spline = CubicSpline(r_eval, outer_integral, bc_type="natural", extrapolate=False)

    def outer_integral_func(r):
        r = np.asarray(r)
        if np.any((r < r_target) | (r > rn)):
            raise ValueError("r must lie within [r_target, rn].")
        return outer_spline(r)

    return outer_integral_func, {"r_grid": r_eval, "phi": phi, "inner_integral": inner_integral, "outer_integrand": outer_integrand, "outer_integral": outer_integral}


def normalized_p0_quad(r, r_target, rn, A, alpha, ndim=3):

    def pertubation(s):
        return 4.0 * s * (1 - s)
        
    def potential(r):
        """Calculate the potential at distance r."""
        return A / rn * r - alpha * pertubation(r / rn)

    def weighted_boltzmann(r):
        """Calculate unnormalized Boltzmann probability exp(-U(r)/k_b_T). 
        Including the r^2 term for the volume element in spherical coordinates."""
        return np.exp(-potential(r)) * r**(ndim-1)

    """Calculate normalized Boltzmann distribution."""
    Z, _ = quad(weighted_boltzmann, r_target, rn)
    return weighted_boltzmann(r) / Z


def compute_for_proportional_diffusion(r_target, rn, A, D0, Dmax, alpha=0, ndim=3, N=4000):
    """
    Returns a callable F(r) evaluating
      F(r) = ∫_{r_target}^{r} (1/phi(y)) * [ ∫_{y}^{r_n} phi(z)/D(z) dz ] dy
    where phi solves phi'(r) = (mu(r)/D(r))*phi(r), phi(r_target)=1.
    """

    # definitions (vectorized-friendly)
    def pertubation(s):
        return 4.0 * s * (1.0 - s)

    def dpertubation(s):
        return 4.0 - 8.0 * s
    
    def U(rr):
        """Calculate the potential at distance r."""
        return A / rn * rr - alpha * pertubation(rr / rn) 

    def dU(rr):
        """Calculate the derivative of the potential at distance r."""
        return (A - alpha * dpertubation(rr / rn)) / rn

    def D(rr):
        """Calculate the diffusion constant at distance r."""
        return D0 + U(rr) / A * Dmax

    def dD(rr):
        """Calculate the derivative of the diffusion constant at distance r."""
        return dU(rr) / A * Dmax

    def drift(rr):
        return (ndim - 1) * D(rr) / rr + dD(rr) - D(rr) * dU(rr)

    # grid
    r_eval = np.linspace(r_target, rn, N)

    # ODE: phi' = (mu/D) * phi
    def rhs(x, phi):
        return (drift(x) / D(x)) * phi

    # solve_ivp(func, t_span, rhs(t_span[0]), t_eval=..., rtol=..., atol=...)
    # Instead of time, we use the spatial variable r here
    sol = solve_ivp(rhs, (r_target, rn), y0=[1.0], t_eval=r_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError("phi ODE failed: " + sol.message)
    phi = sol.y[0]

    # inner integral I(y) = ∫_{y}^{r_n} phi(z)/D(z) dz (right-to-left cumulative)
    inner_integrand = phi / D(r_eval)
    reverse_inner_integral = cumulative_trapezoid(inner_integrand[::-1], r_eval[::-1], initial=0.0)  # ∫_{r_n}^{x} g
    inner_integral = -reverse_inner_integral[::-1]                                              # ∫_{x}^{r_n} g

    # outer integral F(r) = ∫_{r_target}^{r} [I(y)/phi(y)] dy
    outer_integrand = inner_integral / phi
    outer_integral = cumulative_trapezoid(outer_integrand, r_eval, initial=0.0)

    # spline for fast evaluation
    outer_spline = CubicSpline(r_eval, outer_integral, bc_type="natural", extrapolate=False)

    def outer_integral_func(r):
        r = np.asarray(r)
        if np.any((r < r_target) | (r > rn)):
            raise ValueError("r must lie within [r_target, rn].")
        return outer_spline(r)

    return outer_integral_func, {"r_grid": r_eval, "phi": phi, "inner_integral": inner_integral, "outer_integrand": outer_integrand, "outer_integral": outer_integral}


def compute_for_sigmoidal(r_target, rf, rn, A, b, D0, Dn, ndim=3, N=4000):
    """
    Returns a callable F(r) evaluating
      F(r) = ∫_{r_target}^{r} (1/phi(y)) * [ ∫_{y}^{r_n} phi(z)/D(z) dz ] dy
    where phi solves phi'(r) = (mu(r)/D(r))*phi(r), phi(r_target)=1.
    """

    # definitions (vectorized-friendly)
    def sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(x):
        e = np.exp(-x) 
        return e / (1.0 + e)**2

    def D(rr):
        return D0 + (Dn - D0) * sigmoid(b * (rr - rf))

    def dD(rr):
        return (Dn - D0) * sigmoid_derivative(b * (rr - rf)) * b

    def dU(rr):
        return A * sigmoid_derivative(b * (rr - rf)) * b

    def drift(rr):
        return (ndim - 1) * D(rr) / rr + dD(rr) - D(rr) * dU(rr)

    # grid
    r_eval = np.linspace(r_target, rn, N)

    # ODE: phi' = (mu/D) * phi
    def rhs(x, phi):
        return (drift(x) / D(x)) * phi

    # solve_ivp(func, t_span, rhs(t_span[0]), t_eval=..., rtol=..., atol=...)
    # Instead of time, we use the spatial variable r here
    sol = solve_ivp(rhs, (r_target, rn), y0=[1.0], t_eval=r_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError("phi ODE failed: " + sol.message)
    phi = sol.y[0]

    # inner integral I(y) = ∫_{y}^{r_n} phi(z)/D(z) dz (right-to-left cumulative)
    inner_integrand = phi / D(r_eval)
    reverse_inner_integral = cumulative_trapezoid(inner_integrand[::-1], r_eval[::-1], initial=0.0)  # ∫_{r_n}^{x} g
    inner_integral = -reverse_inner_integral[::-1]                                              # ∫_{x}^{r_n} g

    # outer integral F(r) = ∫_{r_target}^{r} [I(y)/phi(y)] dy
    outer_integrand = inner_integral / phi
    outer_integral = cumulative_trapezoid(outer_integrand, r_eval, initial=0.0)

    # spline for fast evaluation
    outer_spline = CubicSpline(r_eval, outer_integral, bc_type="natural", extrapolate=False)

    def outer_integral_func(r):
        r = np.asarray(r)
        if np.any((r < r_target) | (r > rn)):
            raise ValueError("r must lie within [r_target, r_n].")
        return outer_spline(r)

    return outer_integral_func, {"r_grid": r_eval, "phi": phi, "inner_integral": inner_integral, "outer_integrand": outer_integrand, "outer_integral": outer_integral}



def U_quad_variable_coef(r, rn, A, alpha, b):
    """Calculate the potential at distance r."""
    x = r / rn 
    
    return A / rn * r - alpha * pertubation_quad(r / rn)

def dU_quad(r, rn, A, alpha):
    """Calculate the derivative of the potential at distance r."""
    return (A - alpha * dpertubation_quad(r / rn)) / rn