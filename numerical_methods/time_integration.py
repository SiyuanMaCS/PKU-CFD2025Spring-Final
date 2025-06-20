import numpy as np


def rk3(U, RHS, dt, params):
    """
    Third-order Runge-Kutta time integration method.

    Parameters:
        U (np.ndarray): Conservative variables [ρ, ρu, E], shape (3, nx)
        RHS (function): Function to compute spatial discretization term dU/dt
        dt (float): Time step size
        params (dict): Dictionary of simulation parameters, includes boundary function

    Returns:
        np.ndarray: Updated conservative variables U_new with shape (3, nx)
    """
    # Get boundary condition function
    boundary_func = params.get('boundary_func', lambda U_val, p: U_val)

    # Stage 1
    U1 = U + dt * RHS(U)
    U1 = boundary_func(U1, params)  # Apply boundary conditions

    # Stage 2
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * RHS(U1)
    U2 = boundary_func(U2, params)  # Apply boundary conditions

    # Stage 3
    U_new = (1/3) * U + (2/3) * U2 + (2/3) * dt * RHS(U2)
    U_new = boundary_func(U_new, params)  # Apply boundary conditions

    return U_new


def compute_dt(U, dx, cfl, gamma):
    """
    Compute time step based on CFL condition.

    Parameters:
        U (np.ndarray): Conservative variables [ρ, ρu, E], shape (3, nx)
        dx (float): Spatial grid spacing
        cfl (float): CFL number (typically < 1 for stability)
        gamma (float): Specific heat ratio

    Returns:
        float: Computed time step dt
    """
    # Compute primitive variables with safety limits
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / np.maximum(rho, 1e-10)

    # Compute internal energy with bounds
    e = np.maximum(U[2] - 0.5 * rho * np.minimum(u**2, 1e10), 1e-10)
    p = np.maximum((gamma - 1) * e, 1e-10)

    # Compute sound speed
    c = np.sqrt(gamma * p / np.maximum(rho, 1e-10))

    # Compute maximum wave speed
    max_speed = np.max(np.abs(u) + c)

    # Compute time step
    if max_speed < 1e-10:
        # Avoid division by zero
        dt = cfl * dx
    else:
        dt = cfl * dx / max_speed

    # Apply minimum time step limit
    min_dt = 1e-6
    dt = max(dt, min_dt)

    return dt