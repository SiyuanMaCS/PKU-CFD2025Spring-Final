import numpy as np

def lax_wendroff_flux(states, dx, dt, gamma=1.4):
    """
    Compute numerical flux using the Lax-Wendroff scheme for a single interface.

    Parameters:
        states (np.ndarray): Left and right conservative variables [ρ, ρu, E], shape (3, 2)
        dx (float): Spatial step size
        dt (float): Time step size
        gamma (float): Specific heat ratio (default: 1.4)

    Returns:
        np.ndarray: Computed flux vector of shape (3,)
    """

    # Unpack left and right states
    U_L = states[:, 0]
    U_R = states[:, 1]

    # Compute primitive variables from conservative variables (with safety limits)
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    e_L = max(U_L[2] - 0.5 * rho_L * u_L**2, 1e-10)
    p_L = (gamma - 1) * e_L

    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    e_R = max(U_R[2] - 0.5 * rho_R * u_R**2, 1e-10)
    p_R = (gamma - 1) * e_R

    # Compute fluxes at left and right
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (U_L[2] + p_L)
    ])

    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (U_R[2] + p_R)
    ])

    # Predictor step: compute intermediate state
    U_mid = 0.5 * (U_L + U_R) - 0.5 * (dt / dx) * (F_R - F_L)

    # Compute primitive variables at mid-point
    rho_mid = max(U_mid[0], 1e-10)
    u_mid = U_mid[1] / rho_mid
    e_mid = max(U_mid[2] - 0.5 * rho_mid * u_mid**2, 1e-10)
    p_mid = (gamma - 1) * e_mid

    # Compute flux at mid-point
    F_mid = np.array([
        rho_mid * u_mid,
        rho_mid * u_mid**2 + p_mid,
        u_mid * (U_mid[2] + p_mid)
    ])

    return F_mid


def steger_warming_flux(states, gamma=1.4):
    """
    Steger-Warming flux vector splitting method for Euler equations.

    Parameters:
        states (np.ndarray): Left and right conservative variables [ρ, ρu, E], shape (3, 2)
        gamma (float): Specific heat ratio (default: 1.4)

    Returns:
        np.ndarray: Computed flux vector of shape (3,)
    """

    # Extract left and right states
    U_L = states[:, 0]
    U_R = states[:, 1]

    # Compute primitive variables for left state
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    e_L = max(U_L[2] - 0.5 * rho_L * u_L**2, 1e-10)
    p_L = (gamma - 1) * e_L
    c_L = np.sqrt(gamma * p_L / rho_L)
    H_L = (U_L[2] + p_L) / rho_L

    # Compute primitive variables for right state
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    e_R = max(U_R[2] - 0.5 * rho_R * u_R**2, 1e-10)
    p_R = (gamma - 1) * e_R
    c_R = np.sqrt(gamma * p_R / rho_R)
    H_R = (U_R[2] + p_R) / rho_R

    # Characteristic speeds
    lambda1_L = u_L
    lambda2_L = u_L + c_L
    lambda3_L = u_L - c_L

    lambda1_R = u_R
    lambda2_R = u_R + c_R
    lambda3_R = u_R - c_R

    # Flux vector splitting
    def split_speeds(lam):
        lam_plus = 0.5 * (lam + abs(lam))
        lam_minus = 0.5 * (lam - abs(lam))
        return lam_plus, lam_minus

    # Left side positive flux
    l1p, _ = split_speeds(lambda1_L)
    l2p, _ = split_speeds(lambda2_L)
    l3p, _ = split_speeds(lambda3_L)

    F_plus = np.zeros(3)
    F_plus[0] = rho_L / (2 * gamma) * (2 * (gamma - 1) * l1p + l2p + l3p)
    F_plus[1] = rho_L / (2 * gamma) * (
        2 * (gamma - 1) * l1p * u_L +
        l2p * (u_L + c_L) +
        l3p * (u_L - c_L)
    )
    F_plus[2] = rho_L / (2 * gamma) * (
        (gamma - 1) * l1p * u_L**2 +
        0.5 * l2p * (u_L + c_L)**2 +
        0.5 * l3p * (u_L - c_L)**2 +
        ((3 - gamma) / (2 * (gamma - 1))) * (l2p + l3p - 2 * (gamma - 1) * l1p) * c_L**2
    )

    # Right side negative flux
    _, l1m = split_speeds(lambda1_R)
    _, l2m = split_speeds(lambda2_R)
    _, l3m = split_speeds(lambda3_R)

    F_minus = np.zeros(3)
    F_minus[0] = rho_R / (2 * gamma) * (2 * (gamma - 1) * l1m + l2m + l3m)
    F_minus[1] = rho_R / (2 * gamma) * (
        2 * (gamma - 1) * l1m * u_R +
        l2m * (u_R + c_R) +
        l3m * (u_R - c_R)
    )
    F_minus[2] = rho_R / (2 * gamma) * (
        (gamma - 1) * l1m * u_R**2 +
        0.5 * l2m * (u_R + c_R)**2 +
        0.5 * l3m * (u_R - c_R)**2 +
        ((3 - gamma) / (2 * (gamma - 1))) * (l2m + l3m - 2 * (gamma - 1) * l1m) * c_R**2
    )

    return F_plus + F_minus


def lax_friedrichs_flux(states, gamma=1.4):
    """
    Lax-Friedrichs numerical flux for Euler equations.

    Parameters:
        states (np.ndarray): Left and right conservative variables [ρ, ρu, E], shape (3, 2)
        gamma (float): Specific heat ratio (default: 1.4)

    Returns:
        np.ndarray: Computed flux vector of shape (3,)
    """

    # Extract left and right states
    U_L = states[:, 0]
    U_R = states[:, 1]

    # Primitive variables for left state
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    e_L = max(U_L[2] - 0.5 * rho_L * u_L**2, 1e-10)
    p_L = (gamma - 1) * e_L
    c_L = np.sqrt(gamma * p_L / rho_L)

    # Primitive variables for right state
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    e_R = max(U_R[2] - 0.5 * rho_R * u_R**2, 1e-10)
    p_R = (gamma - 1) * e_R
    c_R = np.sqrt(gamma * p_R / rho_R)

    # Compute fluxes
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (U_L[2] + p_L)
    ])

    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (U_R[2] + p_R)
    ])

    # Max wave speed
    max_speed = max(abs(u_L) + c_L, abs(u_R) + c_R)

    # Lax-Friedrichs flux
    F = 0.5 * (F_L + F_R) - 0.5 * max_speed * (U_R - U_L)

    return F


def get_flux_function(flux_type='steger_warming'):
    """
    Get the flux function based on the specified type.

    Parameters:
        flux_type (str): Type of flux function.
                         Supported values: 'steger_warming', 'lax_friedrichs'

    Returns:
        function: A flux computation function that takes (states, ...) as input
    """

    if flux_type == 'steger_warming':
        return steger_warming_flux
    elif flux_type == 'lax_friedrichs':
        return lax_friedrichs_flux
    else:
        raise ValueError(f"Unsupported flux type: {flux_type}")