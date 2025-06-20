import numpy as np


def gvc_limiter(a, b):
    """
    GVC (Group Velocity Control) limiter function.

    Parameters:
        a, b: Input values (scalar or array-like)
    
    Returns:
        Limiter value(s) with the same shape as input.
    
    Formula:
        phi(r) = 2r / (r^2 + 1), where r = a / b
    
    Notes:
        - Handles division by zero by returning 0 in such cases.
        - Supports both scalar and array inputs.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        result = np.zeros_like(a)
        for i in range(len(a)):
            if abs(b[i]) < 1e-10:
                result[i] = 0
            else:
                r = a[i] / b[i]
                result[i] = (2 * r) / (r ** 2 + 1)
        return result
    else:
        if abs(b) < 1e-10:
            return 0
        else:
            r = a / b
            return (2 * r) / (r ** 2 + 1)


def muscl_reconstruction_gvc(U):
    """
    MUSCL reconstruction using GVC limiter for group velocity control.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, nx)

    Returns:
        Reconstructed left and right states at interfaces: U_L, U_R of shape (3, nx)
    """
    nx = U.shape[1]
    U_L = np.zeros_like(U)
    U_R = np.zeros_like(U)

    for var in range(3):  # Loop over variables: density, momentum, energy
        grad_left = U[var, 1:-1] - U[var, 0:-2]   # Left gradient
        grad_right = U[var, 2:] - U[var, 1:-1]    # Right gradient
        grad_center = U[var, 2:] - U[var, 0:-2]   # Central gradient

        phi = gvc_limiter(grad_left, grad_center)

        U_L[var, 1:-1] = U[var, 1:-1] - 0.5 * phi * grad_left
        U_R[var, 1:-1] = U[var, 1:-1] + 0.5 * phi * grad_right

    # Boundary handling (first-order extrapolation)
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]

    return U_L, U_R


def gvc_flux(U, flux_func, **kwargs):
    """
    Compute flux using GVC scheme.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, nx)
        flux_func: Function to compute flux from left and right states
        kwargs: Additional arguments for flux function (e.g., dx, dt)

    Returns:
        Flux array F of shape (3, nx - 1)
    """
    U_L, U_R = muscl_reconstruction_gvc(U)
    nx = U.shape[1]
    F = np.zeros((3, nx - 1))

    for i in range(nx - 1):
        state_left = U_R[:, i]
        state_right = U_L[:, i + 1]
        states = np.array([state_left, state_right]).T
        F[:, i] = flux_func(states, **kwargs)

    return F


def minmod(a, b):
    """
    Minmod limiter function.

    Parameters:
        a, b: Input values (scalar or array-like)

    Returns:
        Result of the minmod function.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        result = np.zeros_like(a)
        for i in range(len(a)):
            if a[i] * b[i] <= 0:
                result[i] = 0
            else:
                result[i] = np.sign(a[i]) * min(abs(a[i]), abs(b[i]))
        return result
    else:
        if a * b <= 0:
            return 0
        else:
            return np.sign(a) * min(abs(a), abs(b))


def muscl_reconstruction(U, limiter='minmod'):
    """
    MUSCL reconstruction using selected limiter.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, nx)
        limiter: Type of limiter ('minmod', 'superbee', 'van_leer')

    Returns:
        Reconstructed left and right states at interfaces: U_L, U_R of shape (3, nx)
    """
    nx = U.shape[1]
    U_L = np.zeros_like(U)
    U_R = np.zeros_like(U)

    for var in range(3):
        left_grad = U[var, 1:] - U[var, :-1]
        right_grad = U[var, 2:] - U[var, 1:-1]

        phi = np.zeros(nx - 2)

        for i in range(nx - 2):
            if limiter == 'minmod':
                phi[i] = minmod(left_grad[i], right_grad[i])
            else:
                phi[i] = 0  # Placeholder for other limiters

        U_L[var, 1:-1] = U[var, 1:-1] + 0.5 * phi
        U_R[var, 1:-1] = U[var, 1:-1] - 0.5 * phi

    # Boundary handling
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]

    return U_L, U_R


def tvd_flux(U, flux_func, limiter='minmod', **kwargs):
    """
    TVD flux computation using MUSCL reconstruction.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, nx)
        flux_func: Function to compute flux from left and right states
        limiter: Type of limiter ('minmod', 'superbee', 'van_leer')
        kwargs: Additional parameters for flux function

    Returns:
        Flux array F of shape (3, nx - 1)
    """
    U_L, U_R = muscl_reconstruction(U, limiter=limiter)
    nx = U.shape[1]
    F = np.zeros((3, nx - 1))

    for i in range(nx - 1):
        state_left = U_R[:, i]
        state_right = U_L[:, i + 1]
        states = np.array([state_left, state_right]).T
        F[:, i] = flux_func(states, **kwargs)

    return F


def weno5_js_reconstruction(v):
    """
    Fifth-order WENO-JS reconstruction (Jiang & Shu, 1996).

    Parameters:
        v: Array of 5 cell averages [v_{i-2}, ..., v_{i+2}]

    Returns:
        Reconstructed value at interface i+1/2
    """
    if len(v) != 5:
        raise ValueError(f"WENO requires 5 points, got {len(v)}")

    gamma = np.array([0.1, 0.6, 0.3])

    # Sub-stencil reconstructions
    v0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6.0
    v1 = (-v[1] + 5*v[2] + 2*v[3]) / 6.0
    v2 = (2*v[2] + 5*v[3] - v[4]) / 6.0

    # Smoothness indicators
    beta0 = (13/12)*(v[0] - 2*v[1] + v[2])**2 + (1/4)*(v[0] - 4*v[1] + 3*v[2])**2
    beta1 = (13/12)*(v[1] - 2*v[2] + v[3])**2 + (1/4)*(v[1] - v[3])**2
    beta2 = (13/12)*(v[2] - 2*v[3] + v[4])**2 + (1/4)*(3*v[2] - 4*v[3] + v[4])**2

    epsilon = 1e-6

    # Weights
    alpha0 = gamma[0] / (epsilon + beta0)**2
    alpha1 = gamma[1] / (epsilon + beta1)**2
    alpha2 = gamma[2] / (epsilon + beta2)**2
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return w0*v0 + w1*v1 + w2*v2


def weno_reconstruction(U, num_ghost=3):
    """
    WENO reconstruction for conservative variables.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, n)
        num_ghost: Number of ghost cells (minimum 3)

    Returns:
        Reconstructed left and right states at interfaces: U_L, U_R of shape (3, n-1)
    """
    if num_ghost < 3:
        raise ValueError("WENO requires at least 3 ghost cells")

    n = U.shape[1]
    n_interface = n - 1
    U_L = np.zeros((3, n_interface))
    U_R = np.zeros((3, n_interface))

    for var in range(3):
        for i in range(num_ghost, n - num_ghost - 1):
            stencil_left = U[var, i-2:i+3]
            U_L[var, i] = weno5_js_reconstruction(stencil_left)

            stencil_right = U[var, i-1:i+4][::-1]
            U_R[var, i] = weno5_js_reconstruction(stencil_right)

    # Boundary handling
    for var in range(3):
        for i in range(num_ghost):
            U_L[var, i] = U[var, i + 1]
            U_R[var, i] = U[var, i + 1]

            j = n_interface - 1 - i
            U_L[var, j] = U[var, j]
            U_R[var, j] = U[var, j]

    return U_L, U_R


def weno_flux(U, flux_func, num_ghost=3, **kwargs):
    """
    WENO flux computation.

    Parameters:
        U: Conservative variables [ρ, ρu, E], shape (3, nx)
        flux_func: Function to compute flux from left and right states
        num_ghost: Number of ghost cells (default: 3)
        kwargs: Additional arguments for flux function

    Returns:
        Flux array F of shape (3, nx - 1)
    """
    U_L, U_R = weno_reconstruction(U, num_ghost)
    nx = U.shape[1]
    F = np.zeros((3, nx - 1))

    for i in range(nx - 1):
        state_left = U_L[:, i]
        state_right = U_R[:, i]
        states = np.array([state_left, state_right]).T
        F[:, i] = flux_func(states, **kwargs)

    return F