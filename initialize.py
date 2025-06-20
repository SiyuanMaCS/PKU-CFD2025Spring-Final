import numpy as np


def initialize_sod(domain, gamma=1.4, cv=1.0):
    """
    Set initial conditions for the Sod shock tube problem.

    Parameters:
        domain (dict): Dictionary containing computational domain parameters
        gamma (float): Specific heat ratio (default: 1.4 for air)
        cv (float): Specific heat at constant volume (default: 1.0 for non-dimensionalization)

    Returns:
        dict: Dictionary containing all physical parameters
    """
    x = domain['x']

    # Initial conditions for Sod shock tube
    rho = np.where(x < 0, 1.0, 0.125)   # Density
    u = np.zeros_like(x)                # Velocity
    p = np.where(x < 0, 1.0, 0.1)       # Pressure

    # Compute conservative variables [ρ, ρu, E]
    E = p / (gamma - 1) + 0.5 * rho * u**2
    U = np.array([rho, rho * u, E])

    # Collect all physical parameters
    physics = {
        'gamma': gamma,
        'cv': cv,
        'U_init': U,
        'rho_init': rho,
        'u_init': u,
        'p_init': p,
        'bc_type': 'non-reflective',  # Default boundary condition
        'cfl': 0.5,                   # Default CFL number
    }

    return physics


def set_initial_conditions(physics, domain):
    """
    Set initial conditions and integrate into domain parameters.

    Parameters:
        physics (dict): Dictionary containing physical parameters
        domain (dict): Dictionary containing domain parameters

    Returns:
        dict: Combined dictionary with all simulation parameters
    """
    # Merge physics and domain dictionaries
    params = {**domain, **physics}

    # Add initial diagnostic information
    discontinuity_index = np.argmax(np.abs(np.diff(params['rho_init'])) > 0.5)
    if discontinuity_index < len(params['x']) - 1:
        params['discontinuity_position'] = params['x'][discontinuity_index]
        params['discontinuity_state'] = {
            'left': {
                'rho': params['rho_init'][discontinuity_index],
                'u': params['u_init'][discontinuity_index],
                'p': params['p_init'][discontinuity_index]
            },
            'right': {
                'rho': params['rho_init'][discontinuity_index + 1],
                'u': params['u_init'][discontinuity_index + 1],
                'p': params['p_init'][discontinuity_index + 1]
            }
        }

    return params


def create_domain(nx=200, x_min=-5.0, x_max=5.0, t_end=2.0):
    """
    Create computational domain and grid.

    Parameters:
        nx (int): Number of grid points (default: 200)
        x_min (float): Left boundary of domain (default: -5.0)
        x_max (float): Right boundary of domain (default: 5.0)
        t_end (float): End time of simulation (default: 2.0)

    Returns:
        dict: Dictionary containing all domain/grid parameters
    """
    # Compute grid spacing
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx)  # Coordinates at cell centers

    # Collect all domain parameters
    domain = {
        'nx': nx,
        'dx': dx,
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        't_end': t_end,
        'dt': 0.0,  # Will be computed based on CFL condition
    }

    return domain