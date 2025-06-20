import os
import time
import numpy as np
import argparse

# Import modules
from initialize import create_domain, initialize_sod, set_initial_conditions
from utils import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils import compute_exact_solution, interpolate_exact_to_grid, calculate_error, plot_solution_comparison
from numerical_methods.time_integration import rk3, compute_dt

# Numerical method imports
from numerical_methods.fluxes import get_flux_function
from numerical_methods.schemes import tvd_flux, gvc_flux, weno_flux


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="1D Euler Equation Solver")

    # Domain parameters
    parser.add_argument('--nx', type=int, default=200, help='Number of grid points')
    parser.add_argument('--x_min', type=float, default=-5.0, help='Left boundary of domain')
    parser.add_argument('--x_max', type=float, default=5.0, help='Right boundary of domain')
    parser.add_argument('--t_end', type=float, default=2.0, help='End simulation time')

    # Physical parameters
    parser.add_argument('--gamma', type=float, default=1.4, help='Specific heat ratio gamma')
    parser.add_argument('--cv', type=float, default=1.0, help='Specific heat at constant volume cv')

    # Numerical scheme options
    parser.add_argument('--scheme', type=str, default='weno',
                        choices=['tvd', 'gvc', 'weno'],
                        help='Shock capturing scheme: tvd, gvc, weno')
    parser.add_argument('--flux', type=str, default='steger_warming',
                        choices=['lax_friedrichs', 'roe', 'steger_warming'],
                        help='Flux splitting method: lax_friedrichs, roe, steger_warming')

    # Boundary conditions
    parser.add_argument('--bc_type', type=str, default='non-reflective',
                        choices=['non-reflective', 'periodic', 'fixed'],
                        help='Boundary condition type')
    parser.add_argument('--num_ghost', type=int, default=3, help='Number of ghost cells')

    # Time stepping parameters
    parser.add_argument('--cfl', type=float, default=0.5, help='CFL number')

    # Output control
    parser.add_argument('--output_interval', type=float, default=0.5, help='Output interval in time')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save solution plots')
    parser.add_argument('--plot_dir', type=str, default="results", help='Directory to save output plots')

    return parser.parse_args()


domain_args = {}  # Global variable to store domain info


def get_scheme_function(scheme, gamma=1.4):
    """Get shock-capturing function based on selected scheme"""
    if scheme == 'tvd':
        def tvd_wrapper(U, flux_func, **kwargs):
            return tvd_flux(U, flux_func, **kwargs)
        return tvd_wrapper
    elif scheme == 'gvc':
        def gvc_wrapper(U, flux_func, **kwargs):
            return gvc_flux(U, flux_func, **kwargs)
        return gvc_wrapper
    elif scheme == 'weno':
        def weno_wrapper(U, flux_func, **kwargs):
            return weno_flux(U, flux_func, **kwargs)
        return weno_wrapper
    else:
        raise ValueError(f"Unknown shock-capturing scheme: {scheme}")


def compute_rhs(U, flux_func, scheme_func, num_ghost, dx, dt):
    """Compute spatial discretization term (dU/dt)"""
    U_bc = apply_boundary_conditions(U, domain_args, num_ghost=num_ghost)
    U_ghost = add_ghost_cells(U_bc, num_ghost=num_ghost)

    F = scheme_func(U_ghost, flux_func)

    dFdx = np.zeros_like(U_ghost)
    n = U_ghost.shape[1]

    for i in range(num_ghost, n - num_ghost):
        dFdx[:, i] = (F[:, i] - F[:, i - 1]) / dx

    dFdx = remove_ghost_cells(dFdx, num_ghost=num_ghost)

    return -dFdx  # RHS is -dF/dx


def main(args):
    global domain_args

    print("Creating output directory...")
    os.makedirs(args.plot_dir, exist_ok=True)

    # Create computational domain
    print("Initializing computational domain...")
    domain_args = create_domain(nx=args.nx, x_min=args.x_min, x_max=args.x_max, t_end=args.t_end)

    # Set initial conditions
    physics = initialize_sod(domain_args, gamma=args.gamma, cv=args.cv)
    domain_args = set_initial_conditions(physics, domain_args)

    # Initialize conservative variables
    U = domain_args['U_init'].copy()

    # Get numerical method functions
    flux_func = get_flux_function(args.flux)
    scheme_func = get_scheme_function(args.scheme, args.gamma)

    # Main loop
    t = 0.0
    step = 0
    next_output_time = 0.0

    print(f"\nStarting Simulation: Scheme={args.scheme}, Flux={args.flux}")
    print(f"Grid Points: {args.nx}, End Time: {args.t_end}\n")

    start_time = time.time()

    while t < args.t_end:
        # Compute time step
        dt = compute_dt(U, domain_args['dx'], args.cfl, args.gamma)
        dt = min(dt, args.t_end - t)

        # Time integration using RK3
        rhs_func = lambda U_val: compute_rhs(U_val, flux_func, scheme_func, args.num_ghost, domain_args['dx'], dt)
        U = rk3(U, rhs_func, dt, domain_args)

        # Update time and step count
        t += dt
        step += 1

        # Output progress
        if t >= next_output_time:
            print(f"Time: {t:.4f}/{args.t_end:.2f}, Step: {step}, dt: {dt:.2e}")
            next_output_time += args.output_interval

            # # Save plot
            # if args.save_plots:
            #     exact_data = compute_exact_solution(domain_args, t)
            #     exact_on_grid = interpolate_exact_to_grid(domain_args, exact_data)

            #     plot_file = os.path.join(
            #         args.plot_dir,
            #         f"{args.scheme}_{args.flux}_t={t:.2f}.png"
            #     )
            #     plot_solution_comparison(
            #         domain_args, U, exact_on_grid, t,
            #         title=f"{args.scheme} + {args.flux} (nx={args.nx})",
            #         filename=plot_file
            #     )

    # Final time processing
    print("\nFinal time reached. Computing error and saving final result...")

    exact_data = compute_exact_solution(domain_args, args.t_end)
    exact_on_grid = interpolate_exact_to_grid(domain_args, exact_data)

    # Compute error
    errors = calculate_error(U, exact_on_grid, domain_args)
    print("\nSimulation Completed! Error Summary:")
    print(f"Density L2 Error: {errors['rho']:.4e}")
    print(f"Velocity L2 Error: {errors['u']:.4e}")
    print(f"Pressure L2 Error: {errors['p']:.4e}")

    # Save final plot
    final_plot = os.path.join(
        args.plot_dir,
        f"FINAL_{args.scheme}_{args.flux}_nx{args.nx}.png"
    )
    plot_solution_comparison(
        domain_args, U, exact_on_grid, args.t_end,
        title=f"{args.scheme} + {args.flux} (nx={args.nx}, t={args.t_end})",
        filename=final_plot
    )

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to directory: {args.plot_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)