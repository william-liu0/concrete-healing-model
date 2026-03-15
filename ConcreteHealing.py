#MPI imports
from mpi4py import MPI

#DOLFINX imports
from dolfinx import mesh
from dolfinx import fem
from dolfinx import plot
from dolfinx.io import VTKFile
from dolfinx.fem import functionspace
from dolfinx import io
from dolfinx.fem import assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem

# PETSc imports
from petsc4py import PETSc

# UFL imports
import ufl
from ufl import conditional, le, ge, And, dx

# Python imports
import os
import glob
import time
import csv
import shutil

# Math and plotting imports
import numpy as np
import pyvista
import pyvista as pv
import matplotlib.pyplot as plt


# =====================
# Global plotting config
# =====================
# Adjust these defaults as needed; users can override at runtime by
# calling set_plot_style(custom_font_sizes={...}, show_titles=bool)
DEFAULT_FONT_SIZES = {
    'title': 25,
    'xlabel': 20,
    'ylabel': 20,
    'legend': 20,
    'tick': 15,
    'annotation': 20,
}

SHOW_TITLES = True

def set_plot_style(font_sizes: dict | None = None, show_titles: bool | None = None) -> None:
    """Set matplotlib font sizes and title visibility globally for all plots.

    Parameters
    ----------
    font_sizes : dict | None
        Keys: 'title', 'xlabel', 'ylabel', 'legend', 'tick', 'annotation'.
    show_titles : bool | None
        If provided, override whether titles are shown (controls title size).
    """
    global DEFAULT_FONT_SIZES, SHOW_TITLES
    if font_sizes is None:
        font_sizes = DEFAULT_FONT_SIZES
    else:
        # Merge overrides into defaults so subsequent calls can build on them
        merged = DEFAULT_FONT_SIZES.copy()
        merged.update(font_sizes)
        font_sizes = merged
        DEFAULT_FONT_SIZES = merged

    if show_titles is not None:
        SHOW_TITLES = bool(show_titles)

    # Apply to rcParams
    plt.rcParams.update({
        'font.size': font_sizes['tick'],
        'axes.titlesize': font_sizes['title'] if SHOW_TITLES else 0,
        'axes.labelsize': font_sizes['xlabel'],
        'xtick.labelsize': font_sizes['tick'],
        'ytick.labelsize': font_sizes['tick'],
        'legend.fontsize': font_sizes['legend'],
        'figure.titlesize': font_sizes['title'] if SHOW_TITLES else 0,
    })

# Initialize plotting style once on import
set_plot_style()


def damage_profile(x):
    """
    Linear damage field: vertical crack centered at x = 0.5, gaussian shape
    
    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the points to evaluate the damage field at.

    Returns
    -------
    """

    return np.exp(-((x[0] - 0.5)**2) / 0.001)
    #return np.where(np.abs(x[0] - 0.5) < 0.025, 1.0, 0.0)

def tilted_damage_profile(x,beta,sigma):
    dx = x[0] - 0.5
    dy = x[1] - 0.5
    xi = dx * np.cos(beta) - dy * np.sin(beta)
    
    # Create sharp cutoff: damage = 0 outside crack, Gaussian inside
    # Use a threshold to ensure clean separation
    threshold = 3 * sigma  # 3-sigma rule for 99.7% of Gaussian
    
    # Calculate distance from crack center (vectorized)
    distance = np.abs(xi)
    
    # Create mask for crack region (vectorized)
    crack_mask = distance <= threshold
    
    # Initialize damage array
    damage = np.zeros_like(xi)
    
    # Apply Gaussian damage only within crack region
    damage[crack_mask] = np.exp(-(xi[crack_mask]**2) / (sigma ** 2))
    
    return damage
    #return np.exp(-((x[0] - 0.5)**2) / 0.001)
    # # Return array of zeros - no damage anywhere
    # return np.zeros_like(x[0])

def uD_expr(x):
    values = np.full(x.shape[1], 0.0)  # Default to Neumann-like condition
    tol = 1e-10  # Tolerance for boundary detection
    
    # Left boundary (x = 0) - Dirichlet BC: u = 1
    left_mask = np.abs(x[0]) < tol
    values[left_mask] = 1.0
    
    # Right, top, bottom boundaries - Natural Neumann BC (zero flux)
    # No Dirichlet BC applied, so natural Neumann BC is automatically satisfied
    
    return values

#Smoothing function for the damage field that is used to heal empty space from adjacent unhydrated cement
def smooth_chi(damage: fem.Function, p: float, gamma: float) -> fem.Function:
    V = damage.function_space
    chi_expr = (1.0 - damage) ** p  # Use ** instead of ufl.pow()

    # Define trial and test functions
    chi_eff = fem.Function(V)
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    # Variational form of Helmholtz smoothing
    a = u * v * ufl.dx + gamma * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = chi_expr * v * ufl.dx

    # Assemble and solve
    problem = fem.petsc.LinearProblem(a, L, u=chi_eff,
                                        bcs=[],
                                        petsc_options={"ksp_type": "cg",
                                                        "pc_type": "hypre"})
    chi_eff = problem.solve()
    
    # Clip chi_eff to ensure it stays between 0 and 1
    chi_eff.x.array[:] = np.clip(chi_eff.x.array, 0.0, 1.0)
    chi_eff.name = "chi_eff"

    return chi_eff

# Allow a minimum background diffusion across the crack
def diffusion_coefficient_update(damage: fem.Function, p: float, D_concrete: float, D_air: float):
    log_interpolated_diffusivity = D_concrete ** ((1-damage.x.array[:]) ** p) * (D_air ** (1-(1-damage.x.array[:]) ** p))
    return log_interpolated_diffusivity

def total_damage_area_integral(damage: fem.Function):
    # Define bounds
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    # Get the mesh coordinates
    mesh = damage.function_space.mesh
    x = ufl.SpatialCoordinate(mesh)

    # Indicator function (1 inside the box, 0 outside)
    in_box = And(And(ge(x[0], x_min), le(x[0], x_max)),
                And(ge(x[1], y_min), le(x[1], y_max)))

    # Restricted integrand
    integrand = conditional(in_box, damage, 0.0)

    # Compute integral
    partial_damage = assemble_scalar(form(integrand * dx(mesh)))
    # print(f"Partial damage: {partial_damage}")
    return partial_damage
            

def run_model(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma):
    print("Running model...")

    # Create 1x1 square mesh with NxN cells
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create function space of linear Lagrange elements
    V = functionspace(domain, ("Lagrange", 1))

    # Interpolate damage field into function space
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))

    # Get topology dimensions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Create connectivity for boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Create VTK mesh for visualization
    domain.topology.create_connectivity(tdim,tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)


    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)

    # Time-stepping parameters
    
    dt = (T_final - T_init) / num_steps   # time step size

    # Initialize time
    t = T_init

    # Initial condition: initial moisture U0 = 0.0 inside
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))

    # Project initial condition (L2 projection)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx

    # Store the solution in u_0
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()

    # Time-dependent unknown
    u_n = fem.Function(V)

    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx

    # Dirichlet BCs: U = 1 on left boundary only
    # Natural Neumann BCs (zero flux) on right, top, and bottom boundaries
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Find left boundary facets (x = 0)
    left_facets = []
    for facet in boundary_facets:
        # Get the coordinates of the facet
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        # Check if this facet is on the left boundary (x = 0)
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)

    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Create directory if it doesn't exist
    os.makedirs("healing_diffusion", exist_ok=True)

    # Create VTK files ONCE before the loop
    vtk = VTKFile(MPI.COMM_WORLD, "healing_diffusion/solution.pvd", "w")
    vtk_damage = VTKFile(MPI.COMM_WORLD, "healing_diffusion/damage.pvd", "w")
    vtk_diffusivity = VTKFile(MPI.COMM_WORLD, "healing_diffusion/diffusivity.pvd", "w")

    # Write initial conditions
    vtk.write_function(u_n, t)
    vtk_damage.write_function(damage, t)
    vtk_diffusivity.write_function(D_expr, t)

    
    # Time-stepping loop    
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        solution.name = "saturation"  # set scalar name
        vtk.write_function(solution, t)  # Use existing vtk object
        u_n.x.array[:] = solution.x.array  # update previous solution
        u_n.x.array[:] = np.clip(u_n.x.array, 0.0, 1.0)
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        chi_smooth.x.array[:] = np.clip(chi_smooth.x.array, 0.0, 1.0)

        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt  # Fixed: added dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        damage.name = "damage"  # set scalar name
        vtk_damage.write_function(damage, t)  # Use existing vtk_damage object
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        D_expr.name = "diffusivity"  # set scalar name
        vtk_diffusivity.write_function(D_expr, t)  # Use existing vtk_diffusivity object


    return grid, damage, V

def run_model_with_angle_find_time(N, angle, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, sigma):
    print("Running different angles find time model...")

    # Create 1x1 square mesh with NxN cells
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create function space of linear Lagrange elements
    V = functionspace(domain, ("Lagrange", 1))

    # Interpolate damage field into function space
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=angle, sigma=sigma))

    # Get topology dimensions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Create connectivity for boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Create VTK mesh for visualization
    domain.topology.create_connectivity(tdim,tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    

    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)

    # Time-stepping parameters
    dt = (T_final - T_init) / num_steps   # time step size

    # Initialize time
    t = T_init

    # Initial condition: initial moisture U0 = 0.0 inside
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))

    # Project initial condition (L2 projection)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx

    # Store the solution in u_0
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()

    # Time-dependent unknown
    u_n = fem.Function(V)

    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx

    # Dirichlet BCs: U = 1 on left boundary only
    # Natural Neumann BCs (zero flux) on right, top, and bottom boundaries
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Find left boundary facets (x = 0)
    left_facets = []
    for facet in boundary_facets:
        # Get the coordinates of the facet
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        # Check if this facet is on the left boundary (x = 0)
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)

    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Create directory if it doesn't exist
    os.makedirs("healing_diffusion", exist_ok=True)

    # Create VTK files ONCE before the loop
    vtk = VTKFile(MPI.COMM_WORLD, "healing_diffusion/solution.pvd", "w")
    vtk_damage = VTKFile(MPI.COMM_WORLD, "healing_diffusion/damage.pvd", "w")
    vtk_diffusivity = VTKFile(MPI.COMM_WORLD, "healing_diffusion/diffusivity.pvd", "w")

    # Write initial conditions
    vtk.write_function(u_n, t)
    vtk_damage.write_function(damage, t)
    vtk_diffusivity.write_function(D_expr, t)

    # Calculate initial total damage area
    total_area_damage = total_damage_area_integral(damage)

    # Time-stepping loop    
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        solution.name = "saturation"  # set scalar name
        vtk.write_function(solution, t)  # Use existing vtk object
        u_n.x.array[:] = solution.x.array  # update previous solution
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt  # Fixed: added dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        damage.name = "damage"  # set scalar name
        vtk_damage.write_function(damage, t)  # Use existing vtk_damage object
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        D_expr.name = "diffusivity"  # set scalar name
        vtk_diffusivity.write_function(D_expr, t)  # Use existing vtk_diffusivity object
        new_total_area_damage = total_damage_area_integral(damage)
        #print(f"New total damage: {new_total_area_damage}")
        if new_total_area_damage < 0.05 *total_area_damage:
            #print(f"Time: {t}")
            time_to_heal_95_percent = t
            break
    else:
        # If 95% healing not reached within time limit
        time_to_heal_95_percent = 0
    
    return grid, damage, V, time_to_heal_95_percent

def run_model_with_all_angles_find_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, angle_step, sigma, results_dir="."):
    # Lists to store data
    angles_deg = []
    times_to_heal = []
    
    for deg in range(0, 181, angle_step):
        angle_rad = np.deg2rad(deg)  # convert degrees to radians
        grid, damage, V, time_to_heal_95_percent = run_model_with_angle_find_time(N, angle_rad, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, sigma)
        print(f"Time to heal 95% of damage for angle {deg} degrees: {time_to_heal_95_percent}")
        
        # Store data
        angles_deg.append(deg)
        times_to_heal.append(time_to_heal_95_percent)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(angles_deg, times_to_heal, alpha=0.7, s=50)
    plt.xlabel('Angle $\\beta$ (degrees)', fontsize=20)
    plt.ylabel('Time to heal 95% (seconds)', fontsize=20)
    plt.title('Healing Time vs Crack Angle', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'healing_time_vs_angle.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data to CSV file
    csv_filename = os.path.join(results_dir, 'healing_time_vs_angle_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Angle (degrees)', 'Time to heal 95% (seconds)'])
        # Write data
        for angle, time_to_heal in zip(angles_deg, times_to_heal):
            writer.writerow([angle, time_to_heal])
    
    print(f"Data saved to {csv_filename}")
    
    return angles_deg, times_to_heal

def plot_damage_and_diffusivity(results_dir="."):
    import pyvista as pv
    import numpy as np
    import matplotlib.pyplot as plt

    # File paths
    damage_file = "healing_diffusion/damage_p0_000000.vtu"
    diffusivity_file = "healing_diffusion/diffusivity_p0_000000.vtu"  # adjust name if needed

    # Cross section location
    y0 = 0.5
    tol = 5e-2

    # --- Load damage field ---
    mesh_dmg = pv.read(damage_file)
    points_dmg = mesh_dmg.points
    damage_vals = mesh_dmg.point_data["f"]  # Replace 'f' if needed
    mask_dmg = np.abs(points_dmg[:, 1] - y0) < tol
    x_dmg = points_dmg[mask_dmg][:, 0]
    damage_slice = damage_vals[mask_dmg]

    # --- Load diffusivity field ---
    mesh_diff = pv.read(diffusivity_file)
    points_diff = mesh_diff.points
    diffusivity_vals = mesh_diff.point_data["f"]  # Replace 'D' if needed
    mask_diff = np.abs(points_diff[:, 1] - y0) < tol
    x_diff = points_diff[mask_diff][:, 0]
    diffusivity_slice = diffusivity_vals[mask_diff]

    # --- Sort both by x for clean plots ---
    idx_dmg = np.argsort(x_dmg)
    x_dmg = x_dmg[idx_dmg]
    damage_slice = damage_slice[idx_dmg]

    idx_diff = np.argsort(x_diff)
    x_diff = x_diff[idx_diff]
    diffusivity_slice = diffusivity_slice[idx_diff]

    # --- Calculate dynamic axis limits based on actual data ---
    damage_min, damage_max = np.min(damage_slice), np.max(damage_slice)
    diffusivity_min, diffusivity_max = np.min(diffusivity_slice), np.max(diffusivity_slice)
    
    # Calculate dynamic buffers (5% of data range)
    damage_range = damage_max - damage_min
    diffusivity_range = diffusivity_max - diffusivity_min
    
    damage_buffer = max(0.01, damage_range * 0.05)  # At least 0.01, or 5% of range
    diffusivity_buffer = max(diffusivity_max * 0.01, diffusivity_range * 0.05)  # At least 1% of max, or 5% of range
    
    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(x_dmg, damage_slice, label="damage", color="black")
    axs[0].set_ylabel("Damage")
    axs[0].set_title(f"Cross Section at y = {y0}")
    axs[0].grid(True)
    
    # Set dynamic y-limits for damage with buffer
    axs[0].set_ylim(damage_min - damage_buffer, damage_max + damage_buffer)

    axs[1].plot(x_diff, diffusivity_slice, label="Diffusivity", color="blue")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Diffusivity")
    axs[1].grid(True)
    
    # Set dynamic y-limits for diffusivity with buffer
    axs[1].set_ylim(max(0, diffusivity_min - diffusivity_buffer), diffusivity_max + diffusivity_buffer)
    
    # Print the calculated limits for reference
    print(f"Damage range: [{damage_min:.6f}, {damage_max:.6f}] with buffer ±{damage_buffer:.6f}")
    print(f"Diffusivity range: [{diffusivity_min:.2e}, {diffusivity_max:.2e}] with buffer ±{diffusivity_buffer:.2e}")

    # Save cross-section data to CSV
    csv_filename = os.path.join(results_dir, 'damage_vs_diffusivity_cross_section.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['X Position', 'Damage', 'Diffusivity'])
        # Write data (use the shorter array length)
        min_length = min(len(x_dmg), len(x_diff))
        for i in range(min_length):
            writer.writerow([x_dmg[i], damage_slice[i], diffusivity_slice[i]])
    
    print(f"Cross-section data saved to {csv_filename}")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'damage_vs_diffusivity_cross_section.png'), dpi=300, bbox_inches='tight')
    plt.show()

def clear_output_directory():
    """
    Clear the crack_membrane_model_output directory before each run
    """
    output_dir = "crack_membrane_model_output"
    if os.path.exists(output_dir):
        print(f"Clearing {output_dir} directory...")
        shutil.rmtree(output_dir)
    print(f"Creating fresh {output_dir} directory...")

def print_mesh_info(domain, V):
    """
    Print detailed information about the mesh and function space
    """
    print("\n" + "="*60)
    print("MESH INFORMATION")
    print("="*60)
    
    # Mesh topology information
    tdim = domain.topology.dim
    print(f"Mesh dimension: {tdim}D")
    print(f"Number of cells: {domain.topology.index_map(tdim).size_local}")
    print(f"Number of vertices: {domain.topology.index_map(0).size_local}")
    
    # Function space information
    print(f"\nFunction space type: {V.element}")
    print(f"Number of degrees of freedom: {V.dofmap.index_map.size_local}")
    
    # Mesh geometry information
    print(f"\nMesh bounds:")
    print(f"  X: [{domain.geometry.x[:, 0].min():.6f}, {domain.geometry.x[:, 0].max():.6f}]")
    print(f"  Y: [{domain.geometry.x[:, 1].min():.6f}, {domain.geometry.x[:, 1].max():.6f}]")
    
    # Cell information
    print(f"\nCell information:")
    print(f"  Cell type: {domain.topology.cell_type}")
    # Get number of vertices per cell - for triangles it's always 3
    print(f"  Number of vertices per cell: 3")
    
    # Boundary information
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    print(f"  Number of boundary facets: {len(boundary_facets)}")
    
    print("="*60 + "\n")

def run_plotting(grid, damage, V, plot_every=50, dt=1.0, frame_delay=0.001, results_dir=".", use_logarithmic_saturation=False):
    """
    Create animated visualizations for the normal model with configurable plot coarseness
    
    Parameters:
    -----------
    grid, damage, V : PyVista objects and FEniCS objects
        Model results to visualize
    plot_every : int, optional
        Plot every N time steps (default: 50)
    dt : float, optional
        Time step size in seconds (default: 1.0)
    frame_delay : float, optional
        Delay between frames in seconds (default: 0.001)
    results_dir : str, optional
        Directory to save results (default: ".")
    use_logarithmic_saturation : bool, optional
        Whether to create both linear and logarithmic saturation animations (default: False)
    """
    print("Creating normal model animations...")
    print(f"Plot coarseness: every {plot_every} time steps")
    print(f"Frame delay: {frame_delay} seconds")
    
    # Turn off off-screen mode if you want to display interactively
    pv.OFF_SCREEN = False

    # Collect all timestep files
    files_damage = sorted(glob.glob("healing_diffusion/damage_p0_*.vtu"))
    files_diffusivity = sorted(glob.glob("healing_diffusion/diffusivity_p0_*.vtu"))
    files_water_content = sorted(glob.glob("healing_diffusion/solution_p0_*.vtu"))
    
    
    # Apply plot coarseness by filtering files
    if plot_every > 1:
        files_damage = files_damage[::plot_every]
        files_diffusivity = files_diffusivity[::plot_every]
        files_water_content = files_water_content[::plot_every]
        print(f"Reduced from {len(files_damage)*plot_every} to {len(files_damage)} frames for faster animation")

    # Create the plotter and GIF for damage evolution
    plotter = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter.window_size = [1200, 1000]  # [width, height]
    plotter.open_gif(os.path.join(results_dir, "damage_evolution.gif"))
    
    print(f"Creating damage evolution animation with {len(files_damage)} frames...")
    
    for i, file in enumerate(files_damage):
        mesh = pv.read(file)
        # Check for "damage" field name first, then fallback to "f"
        if "damage" in mesh.point_data:
            scalar_field = "damage"
        elif "f" in mesh.point_data:
            scalar_field = "f"
        else:
            print(f"Warning: No damage field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        actual_time = i * plot_every * dt
        
        plotter.clear()
        plotter.add_mesh(mesh, scalars=scalar_field, clim=[0, 1], cmap="coolwarm", show_edges=False)
        plotter.view_xy()
        plotter.add_title(f"Crack Diffusion Model - Damage Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter.render()
        plotter.write_frame()
        time.sleep(frame_delay)

    plotter.close()

    # Create a separate GIF for water content evolution (linear)
    plotter2 = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter2.window_size = [1200, 1000]  # [width, height]
    plotter2.open_gif(os.path.join(results_dir, "water_content_evolution.gif"))
    
    print(f"Creating water content evolution animation (linear) with {len(files_water_content)} frames...")
    
    for i, file in enumerate(files_water_content):
        mesh = pv.read(file)
        # Check for "saturation" field name first, then fallback to "f"
        if "saturation" in mesh.point_data:
            scalar_field = "saturation"
        elif "f" in mesh.point_data:
            scalar_field = "f"
        else:
            print(f"Warning: No saturation field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        actual_time = i * plot_every * dt
        
        plotter2.clear()
        plotter2.add_mesh(mesh, scalars=scalar_field, clim=[0, 1], cmap="viridis", show_edges=False)
        plotter2.view_xy()
        plotter2.add_title(f"Crack Diffusion Model - Saturation Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter2.render()
        plotter2.write_frame()
        time.sleep(frame_delay)

    plotter2.close()
    
    # Create logarithmic version if requested
    if use_logarithmic_saturation:
        print(f"Creating water content evolution animation (logarithmic) with {len(files_water_content)} frames...")
        
        plotter2_log = pv.Plotter()
        # Set window size to ensure full frame capture
        plotter2_log.window_size = [1200, 1000]  # [width, height]
        plotter2_log.open_gif(os.path.join(results_dir, "water_content_evolution_logarithmic.gif"))
        
        for i, file in enumerate(files_water_content):
            mesh = pv.read(file)
            # Check for "saturation" field name first, then fallback to "f"
            if "saturation" in mesh.point_data:
                scalar_field = "saturation"
            elif "f" in mesh.point_data:
                scalar_field = "f"
            else:
                print(f"Warning: No saturation field found in {file}. Available fields: {list(mesh.point_data.keys())}")
                continue
                
            # Calculate actual time based on step number and dt
            actual_time = i * plot_every * dt
            
            # Logarithmic scaling
            epsilon = 1e-6  # Small value to avoid log(0) in color mapping
            clim = [epsilon, 1]
            plotter2_log.clear()
            plotter2_log.add_mesh(mesh, scalars=scalar_field, clim=clim, cmap="viridis", 
                                show_edges=False, log_scale=True)
            plotter2_log.view_xy()
            plotter2_log.add_title(f"Crack Diffusion Model - Saturation Field - t={actual_time:.2e}s")
            # Force render to ensure title is included in frame
            plotter2_log.render()
            plotter2_log.write_frame()
            time.sleep(frame_delay)
        
        plotter2_log.close()
    
    # Create diffusivity evolution animation
    plotter3 = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter3.window_size = [1200, 1000]  # [width, height]
    plotter3.open_gif(os.path.join(results_dir, "diffusivity_evolution.gif"))
    
    print(f"Creating diffusivity evolution animation with {len(files_diffusivity)} frames...")
    
    # First pass: calculate global min/max diffusivity across all time steps
    print("Calculating global diffusivity range across all time steps...")
    global_diffusivity_min = float('inf')
    global_diffusivity_max = float('-inf')
    
    for file in files_diffusivity:
        mesh = pv.read(file)
        # Check for "diffusivity" field name first, then fallback to "f"
        if "diffusivity" in mesh.point_data:
            diffusivity_vals = mesh.point_data["diffusivity"]
        elif "f" in mesh.point_data:
            diffusivity_vals = mesh.point_data["f"]
        else:
            continue  # Skip files without diffusivity data
        file_min = np.min(diffusivity_vals)
        file_max = np.max(diffusivity_vals)
        global_diffusivity_min = min(global_diffusivity_min, file_min)
        global_diffusivity_max = max(global_diffusivity_max, file_max)
    
    # Set global color scale with padding
    global_diffusivity_clim_min = max(0, global_diffusivity_min * 0.9)  # Don't go below 0
    global_diffusivity_clim_max = global_diffusivity_max * 1.1  # Add 10% padding above max
    
    print(f"Global diffusivity range: [{global_diffusivity_min:.2e}, {global_diffusivity_max:.2e}] m²/s")
    print(f"Color scale limits: [{global_diffusivity_clim_min:.2e}, {global_diffusivity_clim_max:.2e}] m²/s")
    
    # Second pass: create animation with consistent color scale
    for i, file in enumerate(files_diffusivity):
        mesh = pv.read(file)
        # Check for "diffusivity" field name first, then fallback to "f"
        if "diffusivity" in mesh.point_data:
            scalar_field = "diffusivity"
        elif "f" in mesh.point_data:
            scalar_field = "f"
        else:
            print(f"Warning: No diffusivity field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        actual_time = i * plot_every * dt
        
        plotter3.clear()
        plotter3.add_mesh(mesh, scalars=scalar_field, clim=[global_diffusivity_clim_min, global_diffusivity_clim_max], cmap="plasma", show_edges=False)
        plotter3.view_xy()
        plotter3.add_title(f"Crack Diffusino Model - Diffusivity Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter3.render()
        plotter3.write_frame()
        time.sleep(frame_delay)

    plotter3.close()

    # Visualize final mesh with damage field
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["damage"] = damage.x.array.real
    u_grid.set_active_scalars("damage")
    
    final_plotter = pyvista.Plotter()
    final_plotter.add_mesh(u_grid, show_edges=True, show_scalar_bar=True)
    final_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        final_plotter.show()
    else:
        final_plotter.screenshot(os.path.join(results_dir, "final_damage_field.png"))
    
    #Warped plot
    warped = u_grid.warp_by_scalar()
    plotter3 = pyvista.Plotter()
    plotter3.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    if not pyvista.OFF_SCREEN:
        plotter3.show()
    else:
        plotter3.screenshot(os.path.join(results_dir, "warped_damage_field.png"))
    
def plot_initial_damage(domain, V, beta, sigma, results_dir="."):
    """
    Plot the initial damage field before healing begins
    
    Parameters:
    -----------
    results_dir : str, optional
        Directory to save results (default: current directory)
    """
    print("Creating initial damage field visualization...")
    
    # Create initial damage field
    initial_damage = fem.Function(V)
    initial_damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
    initial_damage.name = "damage"
    
    # Create VTK mesh for visualization
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add damage data to grid
    grid.point_data["damage"] = initial_damage.x.array.real
    grid.set_active_scalars("damage")
    
    # Create plotter for initial damage
    initial_plotter = pyvista.Plotter()
    initial_plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    initial_plotter.view_xy()
    
    if not pyvista.OFF_SCREEN:
        initial_plotter.show()
    else:
        initial_plotter.screenshot(os.path.join(results_dir, "initial_damage_field.png"))
    
    # Create warped version of initial damage
    warped_initial = grid.warp_by_scalar()
    warped_plotter = pyvista.Plotter()
    warped_plotter.add_mesh(warped_initial, show_edges=True, show_scalar_bar=True)
    warped_plotter.view_xy()
    
    if not pyvista.OFF_SCREEN:
        warped_plotter.show()
    else:
        warped_plotter.screenshot(os.path.join(results_dir, "initial_damage_field_warped.png"))
    
    print("Initial damage field visualization completed!")


def plot_unit_square_mesh(domain, results_dir="."):
    """
    Plot the unit square mesh structure
    
    Parameters:
    -----------
    domain : dolfinx.mesh.Mesh
        The mesh domain to plot
    results_dir : str, optional
        Directory to save results (default: current directory)
    """
    print("Creating unit square mesh visualization...")
    
    # Create VTK mesh for visualization
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Create plotter for mesh
    mesh_plotter = pyvista.Plotter()
    mesh_plotter.add_mesh(grid, show_edges=True)
    mesh_plotter.view_xy()
    mesh_plotter.add_title(f"Unit Square Mesh ({N}x{N})")
    
    if not pyvista.OFF_SCREEN:
        mesh_plotter.show()
    else:
        mesh_plotter.screenshot(os.path.join(results_dir, "unit_square_mesh.png"))
    
    print("Unit square mesh visualization completed!")


def test_healing_percentage_over_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma, plot_every=50, results_dir=".", create_plot=True, use_logarithmic_saturation=False):
    """
    Test current model parameters and return healing percentage over time
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p, gamma : float
        Model parameters
    T_init, T_final, num_steps : float, float, int
        Time parameters
    beta : float
        Crack angle parameter
    sigma : float, optional
        Crack width parameter (default: 0.001**0.5)
    plot_every : int, optional
        Save data points every N time steps for plotting (default: 50)
        - Lower values = more data points = smoother plots
        - Higher values = fewer data points = faster execution
    create_plot : bool, optional
        Whether to create and show a plot (default: True)
        - Set to False when calling from other functions that create their own plots
    """
    print("Testing healing percentage over time...")
    print(f"Plot coarseness: saving data every {plot_every} time steps")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = functionspace(domain, ("Lagrange", 1))

    # Initial damage field
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
    
    # Calculate initial total damage
    initial_damage = total_damage_area_integral(damage)
    print(f"Initial total damage: {initial_damage:.6f}")
    
    # Time-stepping setup
    dt = (T_final - T_init) / num_steps
    t = T_init
    
    # Storage for results
    times = []
    healing_percentages = []
    
    # Tracking for 100% healing
    hundred_percent_reached = False
    hundred_percent_time = None
    
    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
    
    # Initial condition
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))
    
    # Project initial condition
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()
    
    # Time-dependent unknown
    u_n = fem.Function(V)
    
    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Find left boundary facets
    left_facets = []
    for facet in boundary_facets:
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)
    
    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    # Record initial state
    current_damage = total_damage_area_integral(damage)
    healing_percentage = (initial_damage - current_damage) / initial_damage * 100
    times.append(t)
    healing_percentages.append(healing_percentage)
    
    # Time-stepping loop
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        u_n.x.array[:] = solution.x.array
        # Ensure water content is non-negative
        u_n.x.array[:] = np.clip(u_n.x.array, 0.0, np.inf)
        
        # Update damage field
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        
        # Update diffusivity
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        
        # Calculate healing percentage
        current_damage = total_damage_area_integral(damage)
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        
        # Check if 100% healing is reached
        if healing_percentage >= 100.0 and not hundred_percent_reached:
            hundred_percent_time = t
            hundred_percent_reached = True
            print(f"🎉 100% HEALING ACHIEVED at time: {hundred_percent_time:.2f} seconds!")
        
        # Debug information
        if step % 10 == 0:
            max_chi = chi_smooth.x.array.max()
            min_chi = chi_smooth.x.array.min()
            max_u = u_n.x.array.max()
            min_u = u_n.x.array.min()
            max_damage = damage.x.array.max()
            min_damage = damage.x.array.min()
            healing_rate = alpha * u_n.x.array.max() * chi_smooth.x.array.max() * dt
            print(f"Step {step}/{num_steps}, Time: {t:.2f}, Healing: {healing_percentage:.2f}%")
            print(f"  chi_smooth range: [{min_chi:.6f}, {max_chi:.6f}]")
            print(f"  u_n range: [{min_u:.6f}, {max_u:.6f}]")
            print(f"  damage range: [{min_damage:.6f}, {max_damage:.6f}]")
            print(f"  max healing rate per step: {healing_rate:.8f}")
        
        # Store results based on plot_every parameter
        if step % plot_every == 0:  # Save data points for plotting
            times.append(t)
            healing_percentages.append(healing_percentage)
    
    # Create line plot (better for time series data) - only if requested
    if create_plot:
        plt.figure(figsize=(12, 8))
        plt.plot(times, healing_percentages, 'o-', color='#1f77b4', linewidth=2, markersize=6, alpha=0.8)
        plt.xlabel('Time (seconds)', fontsize=20)
        plt.ylabel('Healed Damage Percentage (%)', fontsize=20)
        plt.title('Crack Diffusion Model - Healing Progress Over Time', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)  # Set y-axis from 0 to 105% for better visualization
        plt.tick_params(labelsize=15)
        
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'healing_percentage_over_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print plotting statistics
    print(f"\nPlotting Statistics:")
    print(f"  Total time steps: {num_steps}")
    print(f"  Data points saved: {len(times)}")
    print(f"  Plot coarseness: every {plot_every} steps")
    print(f"  Data collection frequency: {len(times)/num_steps*100:.1f}% of steps")
    
    # Print final results safely
    print(f"\nFinal healing percentage: {healing_percentages[-1]:.2f}%")
    
    # Safely find milestone times
    healing_array = np.array(healing_percentages)
    times_array = np.array(times)
    
    idx_50 = np.where(healing_array >= 50)[0]
    if len(idx_50) > 0:
        time_50 = times_array[idx_50[0]]
        print(f"Time to reach 50% healing: {time_50:.2f} seconds")
    else:
        print("50% healing not reached within simulation time")
    
    idx_90 = np.where(healing_array >= 90)[0]
    if len(idx_90) > 0:
        time_90 = times_array[idx_90[0]]
        print(f"Time to reach 90% healing: {time_90:.2f} seconds")
    else:
        print("90% healing not reached within simulation time")
    
    # Report 100% healing time if achieved
    if hundred_percent_reached:
        print(f"🎉 Time to reach 100% healing: {hundred_percent_time:.2f} seconds")
    else:
        print("100% healing not reached within simulation time")
    
    # Save healing progress data to CSV
    csv_filename = os.path.join(results_dir, 'healing_percentage_over_time_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Time (seconds)', 'Healing Percentage (%)'])
        # Write data
        for time_val, healing_pct in zip(times, healing_percentages):
            writer.writerow([time_val, healing_pct])
    
    print(f"Healing progress data saved to {csv_filename}")
    
    return times, healing_percentages

def run_model_with_sigma_find_time(N, sigma, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta):
    """
    Run model with specific sigma (crack width) and find time to heal 95% of damage
    """
    print(f"Running model with sigma = {sigma:.6f}...")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = functionspace(domain, ("Lagrange", 1))
    
    # Initial damage field with specific sigma
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
    
    # Calculate initial total damage
    initial_damage = total_damage_area_integral(damage)
    
    # Time-stepping setup
    dt = (T_final - T_init) / num_steps
    t = T_init
    
    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
    
    # Initial condition
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))
    
    # Project initial condition
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()
    
    # Time-dependent unknown
    u_n = fem.Function(V)
    
    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Find left boundary facets
    left_facets = []
    for facet in boundary_facets:
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)
    
    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    # Tracking for 100% healing
    hundred_percent_reached = False
    hundred_percent_time = None
    
    # Time-stepping loop
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        u_n.x.array[:] = solution.x.array
        # Ensure water content is non-negative
        u_n.x.array[:] = np.clip(u_n.x.array, 0.0, np.inf)
        
        # Update damage field
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        
        # Update diffusivity
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        
        # Calculate healing percentage
        current_damage = total_damage_area_integral(damage)
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        
        # Check if 100% healing is reached
        if healing_percentage >= 100.0 and not hundred_percent_reached:
            hundred_percent_time = t
            hundred_percent_reached = True
            print(f"🎉 100% HEALING ACHIEVED for sigma {sigma:.6f} at time: {hundred_percent_time:.2f} seconds!")
        
        # Check if 95% healing is reached
        if healing_percentage >= 95.0:
            time_to_heal_95_percent = t
            print(f"Time to heal 95% for sigma {sigma:.6f}: {time_to_heal_95_percent:.2f} seconds")
            # If we also reached 100%, return both times
            if hundred_percent_reached:
                return time_to_heal_95_percent, hundred_percent_time
            else:
                return time_to_heal_95_percent
    
    # If 95% healing not reached within time limit
    final_healing_percentage = (initial_damage - current_damage) / initial_damage * 100
    print(f"95% healing not reached for sigma {sigma:.6f} within {T_final} seconds. Final healing: {final_healing_percentage:.2f}%")
    
    # Check if we reached 100% healing even if 95% wasn't reached
    if hundred_percent_reached:
        print(f"🎉 100% HEALING ACHIEVED for sigma {sigma:.6f} at time: {hundred_percent_time:.2f} seconds!")
        return T_final, hundred_percent_time
    else:
        return T_final

def run_model_with_all_sigmas_find_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma_min, sigma_max, sigma_step, results_dir="."):
    """
    Run model for all sigma values and plot healing time vs crack width
    """
    # Lists to store data
    sigmas = []
    times_to_heal = []
    
    # Range of sigma values (crack widths)
    sigma_values = np.linspace(sigma_min, sigma_max, int((sigma_max - sigma_min) / sigma_step) + 1)
    
    for sigma in sigma_values:
        time_to_heal_95_percent = run_model_with_sigma_find_time(N, sigma, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta)
        
        # Store data
        sigmas.append(sigma)
        times_to_heal.append(time_to_heal_95_percent)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sigmas, times_to_heal, alpha=0.7, s=50)
    plt.xlabel('Crack Width $\\sigma$', fontsize=20)
    plt.ylabel('Time to heal 95% (seconds)', fontsize=20)
    plt.title('Healing Time vs Crack Width', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'healing_time_vs_crack_width.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data to CSV file
    csv_filename = os.path.join(results_dir, 'healing_time_vs_crack_width_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Crack Width (sigma)', 'Time to heal 95% (seconds)'])
        # Write data
        for sigma, time_to_heal in zip(sigmas, times_to_heal):
            writer.writerow([sigma, time_to_heal])
    
    print(f"Data saved to {csv_filename}")
    
    return sigmas, times_to_heal


def run_model_with_sigma_and_alpha_find_100_percent_time(N, sigma, alpha, D_concrete, D_air, p, gamma, T_init, T_final, num_steps, beta):
    """
    Run model with specific sigma (crack width) and alpha (healing rate) to find time to heal 100% of damage
    This function is designed for creating 3D surface plots of healing time vs crack width and alpha
    """
    print(f"Running model with sigma = {sigma:.6f}, alpha = {alpha:.6f}...")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = functionspace(domain, ("Lagrange", 1))
    
    # Initial damage field with specific sigma
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
    
    # Calculate initial total damage
    initial_damage = total_damage_area_integral(damage)
    
    # Time-stepping setup
    dt = (T_final - T_init) / num_steps
    t = T_init
    
    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
    
    # Initial condition
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))
    
    # Project initial condition
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()
    
    # Time-dependent unknown
    u_n = fem.Function(V)
    
    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Find left boundary facets
    left_facets = []
    for facet in boundary_facets:
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)
    
    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    # Time-stepping loop
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        u_n.x.array[:] = solution.x.array
        # Ensure water content is non-negative
        u_n.x.array[:] = np.clip(u_n.x.array, 0.0, np.inf)
        
        # Update damage field
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        
        # Update diffusivity
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        
        # Calculate healing percentage
        current_damage = total_damage_area_integral(damage)
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        
        # Check if 100% healing is reached
        if healing_percentage >= 100.0:
            print(f"🎉 100% HEALING ACHIEVED for sigma {sigma:.6f}, alpha {alpha:.6f} at time: {t:.2f} seconds!")
            return t
    
    # If 100% healing not reached within time limit
    final_healing_percentage = (initial_damage - current_damage) / initial_damage * 100
    print(f"100% healing not reached for sigma {sigma:.6f}, alpha {alpha:.6f} within {T_final} seconds. Final healing: {final_healing_percentage:.2f}%")
    return T_final


def analyze_healing_progress_important_angles(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, sigma, results_dir="."):
    """
    Analyze healing progress across important crack angles: β = 0, π/8, π/4, π/2
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p, gamma : float
        Model parameters
    T_init, T_final, num_steps : float, float, int
        Time parameters
    sigma : float
        Crack width parameter
    results_dir : str
        Directory to save results
    """
    print("Analyzing healing progress across important crack angles...")
    
    # Define the important angles
    angles = [0, np.pi/8, np.pi/4, np.pi/2]
    angle_names = ["0", "π/8", "π/4", "π/2"]
    
    # Create output directory and clear existing files
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data storage
    all_times = []
    all_healing_percentages = []
    all_angles = []
    
    # High contrast standard colors for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Analyze each angle
    for i, (angle, angle_name) in enumerate(zip(angles, angle_names)):
        print(f"\nAnalyzing angle β = {angle_name} ({np.degrees(angle):.1f}°)...")
        
        # Run model for this angle
        times, healing_percentages = test_healing_percentage_over_time(
            N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, angle, sigma, plot_every=1, results_dir=results_dir
        )
        
        # Store data
        all_times.append(times)
        all_healing_percentages.append(healing_percentages)
        all_angles.append(angle)
        
        # Save individual CSV for this angle
        csv_filename = os.path.join(results_dir, f'angle_{angle_name.replace("/", "_")}_healing_progress.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (seconds)', 'Healing Percentage (%)'])
            for time_val, healing_pct in zip(times, healing_percentages):
                writer.writerow([time_val, healing_pct])
        
        print(f"  ✓ Data saved to {csv_filename}")
        print(f"  ✓ Final healing: {healing_percentages[-1]:.2f}%")
        
        # Create and save individual plot for this angle
        fig_individual, ax_individual = plt.subplots(figsize=(10, 6))
        ax_individual.plot(times, healing_percentages, 
                          color='blue', linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # Customize individual plot
        ax_individual.set_xlabel('Time (seconds)', fontsize=12)
        ax_individual.set_ylabel('Healed Damage Percentage (%)', fontsize=12)
        ax_individual.set_title(f'Healing Progress: β = {angle_name} ({np.degrees(angle):.1f}°)', fontsize=14)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_ylim(0, 105)
        
        # Save individual plot
        individual_plot_filename = os.path.join(results_dir, f'angle_{angle_name.replace("/", "_")}_healing_progress.png')
        plt.savefig(individual_plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)  # Close to free memory
        
        print(f"  ✓ Individual plot saved to {individual_plot_filename}")
    
    # Create combined plot
    print("\nCreating combined healing progress plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each angle with sequential blue colors
    for i, (times, healing_percentages, angle, angle_name) in enumerate(zip(all_times, all_healing_percentages, all_angles, angle_names)):
        ax.plot(times, healing_percentages, 
                color=colors[i], linewidth=2, marker='o', markersize=4, alpha=0.8,
                label=f'β = {angle_name}')
    
    # Customize plot
    ax.set_xlabel('Time (seconds)', fontsize=20)
    ax.set_ylabel('Healed Damage Percentage (%)', fontsize=20)
    ax.set_title('Healing Progress of Various Angles', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 105)  # Set y-axis from 0 to 105% for better visualization
    
    # Save combined plot
    plot_filename = os.path.join(results_dir, 'combined_healing_progress_angles.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined plot saved to {plot_filename}")
    
    # Save combined data to CSV
    combined_csv_filename = os.path.join(results_dir, 'all_angles_combined_data.csv')
    with open(combined_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['Time (seconds)']
        for angle_name in angle_names:
            header.append(f'β = {angle_name} (%)')
        writer.writerow(header)
        
        # Find the maximum time length
        max_length = max(len(times) for times in all_times)
        
        # Write data rows
        for i in range(max_length):
            row = []
            # Add time (use the first angle's time if available, otherwise interpolate)
            if i < len(all_times[0]):
                row.append(all_times[0][i])
            else:
                row.append(all_times[0][-1] + (all_times[0][-1] - all_times[0][-2]) * (i - len(all_times[0]) + 1))
            
            # Add healing percentage for each angle
            for times, healing_percentages in zip(all_times, all_healing_percentages):
                if i < len(healing_percentages):
                    row.append(healing_percentages[i])
                else:
                    row.append(healing_percentages[-1])  # Use final value if beyond range
            
            writer.writerow(row)
    
    print(f"Combined data saved to {combined_csv_filename}")
    print("Healing progress analysis across important angles completed!")
    
    return all_times, all_healing_percentages, all_angles


def analyze_healing_progress_important_sigma(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, results_dir="."):
    """
    Analyze healing progress across important sigma values: σ = 0.005, 0.01, 0.015, 0.02, 0.025
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p, gamma : float
        Model parameters
    T_init, T_final, num_steps : float, float, int
        Time parameters
    beta : float
        Crack angle parameter
    results_dir : str
        Directory to save results
    """
    print("Analyzing healing progress across important sigma values...")
    
    # Define the important sigma values
    sigma_values = [0.01, 0.015, 0.02, 0.025]
    sigma_names = ["0.01", "0.015", "0.02", "0.025"]
    
    # Create output directory and clear existing files
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data storage
    all_times = []
    all_healing_percentages = []
    all_sigma_values = []
    
    # High contrast standard colors for better visibility (matching all angles plot)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Analyze each sigma value
    for i, (sigma, sigma_name) in enumerate(zip(sigma_values, sigma_names)):
        print(f"\nAnalyzing sigma σ = {sigma_name}...")
        
        # Run model for this sigma value
        times, healing_percentages = test_healing_percentage_over_time(
            N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma, plot_every=1, results_dir=results_dir, create_plot=False
        )
        
        # Store data
        all_times.append(times)
        all_healing_percentages.append(healing_percentages)
        all_sigma_values.append(sigma)
        
        # Save individual CSV for this sigma value
        csv_filename = os.path.join(results_dir, f'sigma_{sigma_name}_healing_progress.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (seconds)', 'Healing Percentage (%)'])
            for time_val, healing_pct in zip(times, healing_percentages):
                writer.writerow([time_val, healing_pct])
        
        print(f"  ✓ Data saved to {csv_filename}")
        print(f"  ✓ Final healing: {healing_percentages[-1]:.2f}%")
        
        # Create and save individual plot for this sigma value
        fig_individual, ax_individual = plt.subplots(figsize=(10, 6))
        ax_individual.plot(times, healing_percentages, 
                          color='blue', linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # Customize individual plot
        ax_individual.set_xlabel('Time (seconds)', fontsize=12)
        ax_individual.set_ylabel('Healed Damage Percentage (%)', fontsize=12)
        ax_individual.set_title(f'Healing Progress: sigma={sigma_name}', fontsize=14)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_ylim(0, 105)
        
        # Save individual plot
        individual_plot_filename = os.path.join(results_dir, f'sigma_{sigma_name}_healing_progress.png')
        plt.savefig(individual_plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)  # Close to free memory
        
        print(f"  ✓ Individual plot saved to {individual_plot_filename}")
    
    # Create combined plot
    print("\nCreating combined healing progress plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each sigma value with sequential blue colors
    for i, (times, healing_percentages, sigma, sigma_name) in enumerate(zip(all_times, all_healing_percentages, all_sigma_values, sigma_names)):
        ax.plot(times, healing_percentages, 
                color=colors[i], linewidth=2, marker='o', markersize=4, alpha=0.8,
                label=f'σ = {sigma_name}')
    
    # Customize plot (matching all angles combined plot styling)
    ax.set_xlabel('Time (seconds)', fontsize=20)
    ax.set_ylabel('Healed Damage Percentage (%)', fontsize=20)
    ax.set_title('Healing Progress of Various Sigma Values', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, 105)  # Set y-axis from 0 to 105% for better visualization
    
    # Save combined plot
    plot_filename = os.path.join(results_dir, 'combined_healing_progress_sigma.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined plot saved to {plot_filename}")
    
    # Save combined data to CSV
    combined_csv_filename = os.path.join(results_dir, 'all_sigma_combined_data.csv')
    with open(combined_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['Time (seconds)']
        for sigma_name in sigma_names:
            header.append(f'σ = {sigma_name} (%)')
        writer.writerow(header)
        
        # Find the maximum time length
        max_length = max(len(times) for times in all_times)
        
        # Write data rows
        for i in range(max_length):
            row = []
            # Add time (use the first sigma's time if available, otherwise interpolate)
            if i < len(all_times[0]):
                row.append(all_times[0][i])
            else:
                row.append(all_times[0][-1] + (all_times[0][-1] - all_times[0][-2]) * (i - len(all_times[0]) + 1))
            
            # Add healing percentage for each sigma value
            for times, healing_percentages in zip(all_times, all_healing_percentages):
                if i < len(healing_percentages):
                    row.append(healing_percentages[i])
                else:
                    row.append(healing_percentages[-1])  # Use final value if beyond range
            
            writer.writerow(row)
    
    print(f"Combined data saved to {combined_csv_filename}")
    print("Healing progress analysis across important sigma values completed!")
    
    return all_times, all_healing_percentages, all_sigma_values


def _calculate_single_sgb_healing_time(N, D_concrete, D_air, alpha, p, sigma, gamma, beta,
                                       T_init=0.0, T_final=25000000.0, num_steps=10000,
                                       healing_threshold=95.0):
    """
    Helper function: Calculate healing time for a single (sigma, gamma, beta) combination
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p : float
        Model parameters
    sigma : float
        Crack width parameter
    gamma : float
        Smoothing parameter
    beta : float
        Crack angle parameter (in radians)
    T_init : float
        Initial time (default: 0.0)
    T_final : float
        Final time in seconds (default: 25000000.0)
    num_steps : int
        Number of time steps (default: 10000)
    healing_threshold : float
        Healing percentage threshold (default: 95.0 for 95% healing)
    
    Returns:
    --------
    healing_time : float or None
        Time in seconds to reach the healing threshold, or None if threshold not reached
    final_healing_percentage : float
        Final healing percentage achieved
    """
    try:
        # Create mesh and function space
        domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        V = functionspace(domain, ("Lagrange", 1))
        
        # Initial damage field
        damage = fem.Function(V)
        damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
        
        # Calculate initial total damage
        initial_damage = total_damage_area_integral(damage)
        
        # Time-stepping setup
        dt = (T_final - T_init) / num_steps
        t = T_init
        
        # Create diffusion field function
        D_expr = fem.Function(V)
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        
        # Initial condition
        u_0 = fem.Function(V)
        u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))
        
        # Project initial condition
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a_proj = ufl.inner(u, v) * ufl.dx
        L_proj = ufl.inner(u_0, v) * ufl.dx
        problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
        problem_proj.solve()
        
        # Time-dependent unknown
        u_n = fem.Function(V)
        
        # Variational form for backward Euler
        a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        
        # Find left boundary facets
        left_facets = []
        for facet in boundary_facets:
            facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
            facet_coords = domain.geometry.x[facet_vertices]
            if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
                left_facets.append(facet)
        
        u_D = fem.Function(V)
        u_D.interpolate(uD_expr)
        left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
        bc = fem.dirichletbc(u_D, left_boundary_dofs)
        
        # Prepare solver
        problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
        # Track healing time
        healing_time = None
        
        # Time-stepping loop
        for step in range(num_steps):
            t += dt
            
            # Solve diffusion equation
            solution = problem.solve()
            u_n.x.array[:] = solution.x.array
            u_n.x.array[:] = np.clip(u_n.x.array, 0.0, np.inf)
            
            # Update damage field
            chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
            damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt
            damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
            
            # Update diffusivity
            D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
            
            # Calculate healing percentage
            current_damage = total_damage_area_integral(damage)
            healing_percentage = (initial_damage - current_damage) / initial_damage * 100
            
            # Check if healing threshold is reached
            if healing_percentage >= healing_threshold and healing_time is None:
                healing_time = t
                break
        
        # Calculate final healing percentage
        final_damage = total_damage_area_integral(damage)
        final_healing_percentage = (initial_damage - final_damage) / initial_damage * 100
        
        return healing_time, final_healing_percentage
        
    except Exception as e:
        print(f"  ✗ Error during simulation for σ={sigma:.6f}, γ={gamma:.6f}, β={beta:.6f}: {e}")
        return None, 0.0


def generate_sgb_healing_time_data(N, D_concrete, D_air, alpha, p,
                                    sigma_start, sigma_end, sigma_step,
                                    gamma_start, gamma_end, gamma_step,
                                    beta_start, beta_end, beta_step,
                                    T_init=0.0, T_final=25000000.0, num_steps=10000,
                                    healing_threshold=95.0, results_dir="."):
    """
    Calculate healing time for all combinations of sigma, gamma, and beta parameters
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p : float
        Model parameters
    sigma_start, sigma_end, sigma_step : float
        Sigma parameter range and step size
    gamma_start, gamma_end, gamma_step : float
        Gamma parameter range and step size
    beta_start, beta_end, beta_step : float
        Beta parameter range and step size (in radians)
    T_init : float
        Initial time (default: 0.0)
    T_final : float
        Final time in seconds (default: 25000000.0)
    num_steps : int
        Number of time steps (default: 10000)
    healing_threshold : float
        Healing percentage threshold (default: 95.0 for 95% healing)
    results_dir : str
        Directory to save results (default: ".")
    
    Returns:
    --------
    results : list
        List of tuples (sigma, gamma, beta, healing_time, final_healing_percentage)
    """
    print("Generating SGB (Sigma-Gamma-Beta) healing time data...")
    print("=" * 60)
    
    # Create parameter arrays
    sigma_values = np.arange(sigma_start, sigma_end + sigma_step, sigma_step)
    gamma_values = np.arange(gamma_start, gamma_end + gamma_step, gamma_step)
    beta_values = np.arange(beta_start, beta_end + beta_step, beta_step)
    
    print(f"Parameter ranges:")
    print(f"  Sigma: {sigma_start:.6f} to {sigma_end:.6f}, step {sigma_step:.6f} ({len(sigma_values)} values)")
    print(f"  Gamma: {gamma_start:.6f} to {gamma_end:.6f}, step {gamma_step:.6f} ({len(gamma_values)} values)")
    print(f"  Beta: {beta_start:.6f} to {beta_end:.6f}, step {beta_step:.6f} ({len(beta_values)} values)")
    print(f"  Simulation time: {T_init:.1f} to {T_final:.1f} seconds, {num_steps} steps")
    
    total_combinations = len(sigma_values) * len(gamma_values) * len(beta_values)
    print(f"Total combinations: {total_combinations:,}")
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(results_dir)}")
    
    # Initialize results storage
    results = []
    start_time = time.time()
    successful_simulations = 0
    failed_simulations = 0
    combinations_reached_threshold = 0
    
    # Generate data for all combinations
    current_combination = 0
    for i, sigma in enumerate(sigma_values):
        for j, gamma_val in enumerate(gamma_values):
            for k, beta_val in enumerate(beta_values):
                current_combination += 1
                print(f"\n[{current_combination:,}/{total_combinations:,}] Testing σ={sigma:.6f}, γ={gamma_val:.6f}, β={beta_val:.6f}...")
                
                # Calculate healing time for this combination
                healing_time, final_healing = _calculate_single_sgb_healing_time(
                    N, D_concrete, D_air, alpha, p, sigma, gamma_val, beta_val,
                    T_init, T_final, num_steps, healing_threshold
                )
                
                # Store result
                results.append((sigma, gamma_val, beta_val, healing_time, final_healing))
                
                if healing_time is not None:
                    successful_simulations += 1
                    combinations_reached_threshold += 1
                    print(f"  ✅ {healing_threshold}% healing reached at time: {healing_time:.2f} seconds")
                else:
                    successful_simulations += 1
                    print(f"  ⚠️  {healing_threshold}% healing not reached. Final healing: {final_healing:.2f}%")
                
                # Progress update
                if current_combination % 10 == 0 or current_combination == total_combinations:
                    elapsed_time = time.time() - start_time
                    avg_time_per_combination = elapsed_time / current_combination
                    remaining_combinations = total_combinations - current_combination
                    estimated_remaining_time = remaining_combinations * avg_time_per_combination
                    
                    print(f"\n  Progress: {current_combination:,}/{total_combinations:,} ({current_combination/total_combinations*100:.1f}%)")
                    print(f"  Threshold reached: {combinations_reached_threshold:,} ({combinations_reached_threshold/current_combination*100:.1f}%)")
                    print(f"  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
    
    # Save results to CSV
    csv_filename = os.path.join(results_dir, 'sgb_healing_time_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sigma', 'Gamma', 'Beta', 'Healing_Time', 'Final_Healing_Percentage'])
        for row in results:
            writer.writerow(row)
    
    print(f"\n✅ Data saved to {csv_filename}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✅ Total combinations tested: {total_combinations:,}")
    print(f"✅ Successful simulations: {successful_simulations}")
    print(f"❌ Failed simulations: {failed_simulations}")
    print(f"🎯 Combinations reaching {healing_threshold}% threshold: {combinations_reached_threshold:,} ({combinations_reached_threshold/total_combinations*100:.1f}%)")
    print(f"⏱️  Total execution time: {(time.time() - start_time)/3600:.2f} hours")
    print("="*60)
    
    return results


def create_sigma_gamma_healing_time_3d_plot(N, D_concrete, D_air, alpha, p, T_init, T_final, num_steps, beta, 
                                           sigma_min=0.005, sigma_max=0.025, sigma_step=0.002,
                                           gamma_min=0.005, gamma_max=0.025, gamma_step=0.002,
                                           results_dir="."):
    """
    Create a 3D surface plot of healing time vs crack width (sigma) and smoothing parameter (gamma)
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p : float
        Model parameters
    T_init, T_final, num_steps : float, float, int
        Time parameters
    beta : float
        Crack angle parameter
    sigma_min, sigma_max, sigma_step : float
        Crack width parameter range and step size
    gamma_min, gamma_max, gamma_step : float
        Smoothing parameter range and step size
    results_dir : str
        Directory to save results
    """
    print("Creating 3D surface plot of healing time vs sigma and gamma...")
    
    # Parameter ranges
    sigma_values = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
    gamma_values = np.arange(gamma_min, gamma_max + gamma_step, gamma_step)
    
    print(f"Sigma range: {sigma_values[0]:.3f} to {sigma_values[-1]:.3f}, {len(sigma_values)} values")
    print(f"Gamma range: {gamma_values[0]:.3f} to {gamma_values[-1]:.3f}, {len(gamma_values)} values")
    print(f"Total combinations: {len(sigma_values) * len(gamma_values)}")
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data storage
    data_rows = []
    total_combinations = len(sigma_values) * len(gamma_values)
    current_combination = 0
    
    # Collect data for all combinations
    for i, sigma in enumerate(sigma_values):
        for j, gamma in enumerate(gamma_values):
            current_combination += 1
            print(f"Progress: {current_combination}/{total_combinations} - Testing sigma={sigma:.3f}, gamma={gamma:.3f}")
            
            # Run model with current parameters
            try:
                healing_time = run_model_with_sigma_and_gamma_find_95_percent_time(
                    N, sigma, gamma, D_concrete, D_air, alpha, p, T_init, T_final, num_steps, beta
                )
                
                # Store data
                data_rows.append([sigma, gamma, healing_time])
                print(f"  ✓ Healing time: {healing_time:.2f} seconds")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                # Store error case with a large time value
                data_rows.append([sigma, gamma, T_final])
    
    # Save raw data to CSV
    csv_filename = os.path.join(results_dir, 'sigma_gamma_healing_time_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sigma', 'Gamma', 'Time_to_95_percent_healing'])
        for row in data_rows:
            writer.writerow(row)
    
    print(f"Raw data saved to {csv_filename}")
    
    # Create 3D surface plot
    print("Creating 3D surface plot...")
    
    # Reshape data for plotting
    sigma_mesh, gamma_mesh = np.meshgrid(sigma_values, gamma_values)
    time_mesh = np.zeros_like(sigma_mesh)
    
    # Fill time_mesh with data
    for i, sigma in enumerate(sigma_values):
        for j, gamma in enumerate(gamma_values):
            # Find the corresponding data row
            for row in data_rows:
                if abs(row[0] - sigma) < 1e-6 and abs(row[1] - gamma) < 1e-6:
                    time_mesh[j, i] = row[2]  # Note: j, i for correct orientation
                    break
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(sigma_mesh, gamma_mesh, time_mesh, 
                          cmap='viridis', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Crack Width $\\sigma$', fontsize=20, labelpad=15)
    ax.set_ylabel('Smoothing Parameter γ', fontsize=20, labelpad=15)
    # Keep Z-label empty (we'll use the colorbar for the label text)
    try:
        ax.zaxis.set_rotate_label(True)
    except Exception:
        pass
    ax.set_zlabel('', fontsize=20, labelpad=10)
    try:
        ax.zaxis.labelpad = 10
        ax.zaxis.label.set_rotation(90)
    except Exception:
        pass
    ax.set_title('3D Surface Plot: Healing Time vs Crack Width and Smoothing Parameter', fontsize=20)
    
    # Increase distance between tick labels and axes, set 3D tick label font size to 15
    ax.tick_params(pad=8, labelsize=15)
    ax.tick_params(axis='z', pad=6, labelsize=15)
    
    # Restore default scientific offset (e.g., 1e6) behavior on z-axis
    try:
        ax.zaxis.get_offset_text().set_visible(True)
        # Nudge the scientific offset (e.g., 1e6) slightly closer to the axis
        try:
            off_text = ax.zaxis.get_offset_text()
            ox, oy = off_text.get_position()
            off_text.set_position((ox, oy - 0.02))
        except Exception:
            pass
    except Exception:
        pass
    
    # Add colorbar with the Z-axis label text on the bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.03)
    try:
        cbar.set_label('Time to 95% Healing (seconds)', fontsize=20, labelpad=12)
    except Exception:
        pass
    
    # Set view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Slightly relax layout to reduce overlaps for 3D
    plt.tight_layout()
    try:
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92)
    except Exception:
        pass
    
    # Save plot with extended left border
    plot_filename = os.path.join(results_dir, 'sigma_gamma_healing_time_3d_surface.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.show()
    
    print(f"3D surface plot saved to {plot_filename}")
    print("3D surface plot creation completed!")
    
    return sigma_values, gamma_values, time_mesh


def run_model_with_sigma_and_gamma_find_95_percent_time(N, sigma, gamma, D_concrete, D_air, alpha, p, T_init, T_final, num_steps, beta):
    """
    Run model with specific sigma (crack width) and gamma (smoothing parameter) to find time to heal 95% of damage
    This function is designed for creating 3D surface plots of healing time vs crack width and gamma
    """
    print(f"  Running model with sigma = {sigma:.6f}, gamma = {gamma:.6f}...")
    
    # Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = functionspace(domain, ("Lagrange", 1))
    
    # Initial damage field with specific sigma
    damage = fem.Function(V)
    damage.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
    
    # Calculate initial total damage
    initial_damage = total_damage_area_integral(damage)
    
    # Time-stepping setup
    dt = (T_final - T_init) / num_steps
    t = T_init
    
    # Create diffusion field function
    D_expr = fem.Function(V)
    D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
    
    # Initial condition
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: 0.0 * np.ones(x.shape[1]))
    
    # Project initial condition
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_proj = ufl.inner(u, v) * ufl.dx
    L_proj = ufl.inner(u_0, v) * ufl.dx
    problem_proj = LinearProblem(a_proj, L_proj, u=u_0)
    problem_proj.solve()
    
    # Time-dependent unknown
    u_n = fem.Function(V)
    
    # Variational form for backward Euler
    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(D_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Find left boundary facets
    left_facets = []
    for facet in boundary_facets:
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    u_D = fem.Function(V)
    u_D.interpolate(uD_expr)
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(u_D, left_boundary_dofs)
    
    # Prepare solver
    problem = LinearProblem(a, L, bcs=[bc], u=u_n, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    # Time-stepping loop
    for step in range(num_steps):
        t += dt
        solution = problem.solve()
        u_n.x.array[:] = solution.x.array
        # Ensure water content is non-negative
        u_n.x.array[:] = np.clip(u_n.x.array, 0.0, np.inf)
        
        # Update damage field
        chi_smooth = smooth_chi(damage, p=p, gamma=gamma)
        damage.x.array[:] = damage.x.array[:] - alpha * u_n.x.array[:] * chi_smooth.x.array[:] * dt
        damage.x.array[:] = np.clip(damage.x.array, 0.0, 1.0)
        
        # Update diffusivity
        D_expr.x.array[:] = diffusion_coefficient_update(damage, p=p, D_concrete=D_concrete, D_air=D_air)
        
        # Calculate healing percentage
        current_damage = total_damage_area_integral(damage)
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        
        # Check if 95% healing is reached
        if healing_percentage >= 95.0:
            return t
    
    # If 95% healing not reached within time limit
    return T_final


def run_crack_membrane_model(N, dt, t_end, d_thr, m_open, p_log, 
                            D_conc, D_mild, S_crit, dS_gate, eps_gate,
                            alpha_heal, p_heal, gamma_heal,
                            beta=0, sigma=0):
    """
    Run the FEniCSx crack membrane model with configurable parameters
    
    Parameters:
    -----------
    N : int
        Mesh refinement level (NxN cells)
    dt : float
        Time step size in seconds
    t_end : float
        End time in seconds
    d_thr : float
        Damage threshold for "band" cells
    m_open : float
        Openness exponent
    p_log : float
        Log-interp sharpness in matrix
    D_conc : float
        Intact concrete diffusivity (m^2/s)
    D_mild : float
        Slightly damaged diffusivity (m^2/s)
    S_crit : float
        Wetting threshold in saturation
    dS_gate : float
        Smoothing for logistic gate
    eps_gate : float
        Tiny floor to avoid front pinning

    alpha_heal : float
        Healing rate parameter (default: 0.1 for visible healing)
    p_heal : float
        Healing power parameter for chi smoothing
    gamma_heal : float
        Healing smoothing parameter
    # Note: crack_baseline parameter removed - smart membrane diffusivity control
    # now naturally regulates water flow through cracks based on moisture gate
    beta : float
        Angle of the crack in radians (0 = vertical crack)
    sigma : float
        Width parameter for the crack (controls how wide the damage band is)

    """
    print("Running crack membrane model...")
    
    # Print initial setup summary
    print(f"Setup: {N}x{N} mesh, dt={dt:.2e}s, t_end={t_end:.2e}s")

    # Print initial parameters summary
    print(f"Parameters:")
    print(f"  d_thr={d_thr}, S_crit={S_crit}, alpha_heal={alpha_heal:.2e}, p_heal={p_heal}, gamma_heal={gamma_heal}")
    print(f"  m_open={m_open}, p_log={p_log}")
    print(f"  D_conc={D_conc:.2e}, D_mild={D_mild:.2e}")
    print(f"  beta={beta}, sigma={sigma}")
    
    # Create output directory and clear previous output files
    crack_output_dir = "crack_membrane_model_output"
    if os.path.exists(crack_output_dir):
        print(f"Clearing previous crack membrane output from {crack_output_dir}...")
        for file in os.listdir(crack_output_dir):
            file_path = os.path.join(crack_output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    #print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Could not remove {file}: {e}")
        print(f"Cleared {crack_output_dir} directory")
    else:
        print(f"Creating {crack_output_dir} directory")
        os.makedirs(crack_output_dir, exist_ok=True)
    
    # Create unit square mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    
    # Create function spaces
    V = functionspace(domain, ("Lagrange", 1))      # for saturation U
    Vd = functionspace(domain, ("Lagrange", 1))     # for damage d
    
    # Create damage field using tilted_damage_profile
    d_fun = fem.Function(Vd, name="damage")
    d_fun.interpolate(lambda x: tilted_damage_profile(x, beta=beta, sigma=sigma))
     
    # Create trial and test functions
    U = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain)
    
    # Create diffusivity coefficient function
    D_coeff = fem.Function(Vd, name="diffusivity")
    
    # Initialize with initial diffusivity values; D_mat_initial is the initial diffusivity matrix based on logarithmic interpolation with sharpness p_log
    damage_array = d_fun.x.array
    w_array = (1.0 - damage_array)**p_log
    D_mat_initial = np.exp(w_array * np.log(D_conc) + (1.0 - w_array) * np.log(D_mild))


    # # Calculate initial chi ("openness" normalized between 0 and 1) and apply crack impermeability
    # d_hat_array = np.maximum((damage_array - d_thr) / (1.0 - d_thr), 0.0)
    # d_hat_array = np.minimum(d_hat_array, 1.0)
    # psi_array = d_hat_array**m_open
    # D_crack_initial = D_mat_initial * (1.0 - psi_array)
    
    D_coeff.x.array[:] = D_mat_initial
    
    # --- Time discretization ---
    num_steps = int(t_end / dt)
    
    U_n = fem.Function(V, name="saturation")  # previous time step
    U_n.x.array[:] = 0.0 # initial saturation is 0
    
    # --- Boundary conditions ---
    # Prepare boundary topology for left boundary Dirichlet BC
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Find left boundary facets with tolerance 1e-10
    left_facets = []
    for facet in boundary_facets:
        facet_vertices = domain.topology.connectivity(fdim, 0).links(facet)
        facet_coords = domain.geometry.x[facet_vertices]
        if np.allclose(facet_coords[:, 0], 0.0, atol=1e-10):
            left_facets.append(facet)
    
    # Left boundary x=0 is wet reservoir: U=1
    left_boundary_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc_left = fem.dirichletbc(PETSc.ScalarType(1.0), left_boundary_dofs, V)
    bcs = [bc_left]  # Neumann elsewhere (natural)
    
    # --- Variational form: Backward Euler with CG --- 
    # bilinear form
    a_form = ( ufl.inner(U, v) * dx + dt * ufl.inner(D_coeff * ufl.grad(U), ufl.grad(v)) * dx )
    
    # linear form
    L_form = ( ufl.inner(U_n, v) * dx )

    # --- Output ---
    xdmf = io.XDMFFile(domain.comm, "crack_membrane_model_initial_conditions.xdmf", "w")
    xdmf.write_mesh(domain)  # Write mesh first
    xdmf.write_function(d_fun, 0.0)
    
    # Create VTK files for animation (compatible with your existing infrastructure)
    os.makedirs("crack_membrane_model_output", exist_ok=True)
    vtk_solution = VTKFile(MPI.COMM_WORLD, "crack_membrane_model_output/solution.pvd", "w")
    vtk_damage = VTKFile(MPI.COMM_WORLD, "crack_membrane_model_output/damage.pvd", "w")
    vtk_diffusivity = VTKFile(MPI.COMM_WORLD, "crack_membrane_model_output/diffusivity.pvd", "w")
    
    # Write initial conditions
    vtk_solution.write_function(U_n, 0.0)
    vtk_damage.write_function(d_fun, 0.0)
    
    # Write initial diffusivity (use same function space as D_coeff)
    D_output_initial = fem.Function(Vd, name="diffusivity")
    D_output_initial.x.array[:] = D_coeff.x.array
    vtk_diffusivity.write_function(D_output_initial, 0.0)
    
    # Calculate initial damage for healing percentage tracking
    initial_damage = np.sum(d_fun.x.array)
    print(f"Initial total damage: {initial_damage:.6f}")
    d_fun.name = "damage"  # Set initial name for VTK output
    
    # --- Time loop with backward Euler time stepping and healing ---
    prev_healing = 0.0  # Track previous healing percentage
    for step in range(1, num_steps+1):
        t = step * dt
        
        # --- Simple Backward Euler Time Stepping ---
        # Solve diffusion equation for current time step
        Uh = fem.Function(V, name="U")
        problem_temp = LinearProblem(a_form, L_form, bcs=bcs, u=Uh, petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
        Uh = problem_temp.solve()
        
        # Smart membrane: Diffusivity automatically controls water flow based on moisture gate
        # No manual water blocking needed - the physics handles it naturally
        
        # Define crack mask for smart membrane diffusivity control
        crack_mask = d_fun.x.array > d_thr
        
        # --- HEALING PROCESS ---
        chi_smooth = smooth_chi(d_fun, p=p_heal, gamma=gamma_heal)
        
        # Ensure water content is non-negative before healing
        Uh.x.array[:] = np.clip(Uh.x.array, 0.0, 1.0)
        
        # Apply healing (reduce damage)
        healing_rate = alpha_heal * Uh.x.array[:] * chi_smooth.x.array[:] * dt
        d_fun.x.array[:] = d_fun.x.array[:] - healing_rate
        d_fun.x.array[:] = np.clip(d_fun.x.array, 0.0, 1.0)
        d_fun.name = "damage"  # Set name for VTK output
        
        # --- SMART MEMBRANE: Update diffusivity based on damage field AND moisture gate ---
        damage_array = d_fun.x.array
        w_array = (1.0 - damage_array)**p_log
        D_mat_new = np.exp(w_array * np.log(D_conc) + (1.0 - w_array) * np.log(D_mild))
        
        # Apply moisture gate to diffusivity in crack regions (smart membrane effect)
        if np.any(crack_mask):
            # Get indices of crack regions
            crack_indices = np.where(crack_mask)[0]
            num_crack_points = len(crack_indices)
            
            if num_crack_points > 0:
                # Calculate moisture gate values for crack regions
                crack_water = Uh.x.array[crack_indices]
                
                # GLOBAL GATE SYSTEM: If ANY crack element reaches threshold, ALL crack elements open
                # Find the maximum water content across ALL crack elements
                max_crack_water = np.max(crack_water)
                
                # Calculate a SINGLE global gate value based on the maximum water content
                global_gate_value = eps_gate + (1.0 - eps_gate) / (1.0 + np.exp(-(max_crack_water - S_crit) / dS_gate))
                
                # BINARY GATE: Convert to 0 (closed) or 1 (open) based on 0.5 threshold
                if global_gate_value < 0.5:
                    binary_gate_value = 0.0  # Gate closed - no flow
                else:
                    binary_gate_value = 1.0  # Gate open - full flow
                
                # Apply the BINARY global gate value to ALL crack elements
                D_crack = D_mat_new[crack_indices]
                D_crack_gated = D_crack * binary_gate_value
                
                # Update diffusivity in crack regions
                D_mat_new[crack_indices] = D_crack_gated
                
                # COMPREHENSIVE BLOCKING: When gate is closed, also block water content in crack regions
                if binary_gate_value == 0.0:
                    # Get current water content in crack regions
                    crack_water_current = Uh.x.array[crack_indices]
                    
                    # Set water content to 0 in crack regions when gate is closed
                    Uh.x.array[crack_indices] = 0.0
                    
                                    # Create a barrier zone: also reduce diffusivity in matrix regions near cracks
                # This prevents water from "flowing around" the crack
                barrier_thickness = 0.001  # 5% of domain size
                barrier_mask = d_fun.x.array > (d_thr - barrier_thickness)
                barrier_indices = np.where(barrier_mask)[0]
                
                # Reduce diffusivity in barrier zone when gate is closed
                if len(barrier_indices) > 0:
                    D_barrier = D_mat_new[barrier_indices]
                    D_barrier_reduced = D_barrier * 0.1  # 10% of normal diffusivity
                    D_mat_new[barrier_indices] = D_barrier_reduced
                
                if step % 10 == 0:
                    print(f"    💧 Water content in cracks reset to 0 (gate closed)")
                    print(f"    🚧 Barrier zone created: {len(barrier_indices)} points with reduced diffusivity")
                
                if step % 10 == 0:
                    max_water_in_crack = np.max(crack_water)
                    min_water_in_crack = np.min(crack_water)
                    
                    print(f"    Global smart membrane: Raw gate = {global_gate_value:.3f}, Binary = {binary_gate_value:.0f}")
                    print(f"    Crack water range: [{min_water_in_crack:.3f}, {max_water_in_crack:.3f}]")
                    print(f"    Diffusivity in crack: {np.min(D_crack_gated):.2e} to {np.max(D_crack_gated):.2e} m²/s")
                    
                    # Binary gate status messages
                    if binary_gate_value == 1.0:
                        print(f"    🟢 BINARY GATE OPEN - All crack elements flowing at full diffusivity!")
                    else:
                        print(f"    🔴 BINARY GATE CLOSED - All crack elements blocked (zero diffusivity)")
        
        # Update D_coeff with smart membrane diffusivity
        D_coeff.x.array[:] = D_mat_new
        
        # Create diffusivity function for output
        D_output = fem.Function(V, name="diffusivity")
        D_output.x.array[:] = D_coeff.x.array
        
        # Accept solution as U_{n+1}
        U_n.x.array[:] = Uh.x.array
        U_n.name = "saturation"  # Set name for VTK output
                
        # Progress update every 10 steps
        if step % 10 == 0:
            print(f"  Step {step}: Time = {step * dt:.2e}s")
        
        # Write output every step for smooth animation
        vtk_solution.write_function(U_n, t)
        vtk_damage.write_function(d_fun, t)
        vtk_diffusivity.write_function(D_output, t)
        
        # Calculate healing percentage
        current_damage = np.sum(d_fun.x.array)
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        
        # Print progress every 10 steps
        if step % 10 == 0 and domain.comm.rank == 0:
            print(f"[t={t:.2e}] Step {step}, Healing: {healing_percentage:.2f}%")
    
    # Close all output files
    xdmf.close()
    vtk_solution.close()
    vtk_damage.close()
    vtk_diffusivity.close()
    
    print("Crack membrane model with healing completed!")
    return domain, V, d_fun, U_n


def visualize_crack_membrane_results(domain, V, damage, solution, results_dir="."):
    """
    Visualize the crack membrane model results using existing plotting infrastructure
    """
    print("Creating crack membrane visualization...")
    
    # Create VTK mesh for visualization
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add damage data to grid
    grid.point_data["damage"] = damage.x.array.real
    grid.set_active_scalars("damage")
    
    # Create plotter for damage field
    damage_plotter = pyvista.Plotter()
    damage_plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, 
                           scalars="damage", clim=[0, 1], cmap="coolwarm")
    damage_plotter.view_xy()
    damage_plotter.add_title("Crack Membrane Model - Damage Field")
    
    if not pyvista.OFF_SCREEN:
        damage_plotter.show()
    else:
        damage_plotter.screenshot(os.path.join(results_dir, "crack_membrane_model_damage.png"))
    
    # Create plotter for solution field
    solution_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    solution_grid.point_data["saturation"] = solution.x.array.real
    solution_grid.set_active_scalars("saturation")
    
    solution_plotter = pyvista.Plotter()
    solution_plotter.add_mesh(solution_grid, show_edges=True, show_scalar_bar=True,
                             scalars="saturation", clim=[0, 1], cmap="viridis")
    solution_plotter.view_xy()
    solution_plotter.add_title("Crack Membrane Model - Saturation Field")
    
    if not pyvista.OFF_SCREEN:
        solution_plotter.show()
    else:
        solution_plotter.screenshot(os.path.join(results_dir, "crack_membrane_model_saturation.png"))
    
    # Create warped version of damage field
    warped_damage = grid.warp_by_scalar()
    warped_plotter = pyvista.Plotter()
    warped_plotter.add_mesh(warped_damage, show_edges=True, show_scalar_bar=True)
    warped_plotter.view_xy()
    warped_plotter.add_title("Crack Membrane Model - Warped Damage Field")
    
    if not pyvista.OFF_SCREEN:
        warped_plotter.show()
    else:
        warped_plotter.screenshot(os.path.join(results_dir, "crack_membrane_model_damage_warped.png"))
    
    print("Crack membrane visualization completed!")


def save_crack_membrane_results(domain, V, damage, solution, results_dir="."):
    """
    Save the crack membrane model results in VTK format compatible with existing infrastructure
    """
    print(f"Saving crack membrane results to {results_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create VTK files
    vtk_damage = VTKFile(MPI.COMM_WORLD, f"{results_dir}/damage.pvd", "w")
    vtk_solution = VTKFile(MPI.COMM_WORLD, f"{results_dir}/solution.pvd", "w")
    
    # Write initial conditions at time t=0
    vtk_damage.write_function(damage, 0.0)
    vtk_solution.write_function(solution, 0.0)
    
    # Close files
    vtk_damage.close()
    vtk_solution.close()
    
    print(f"Convenient final crack membrane model results in .pvd format saved to {results_dir}/")
    
    # Also save final state as individual VTU files for easy loading
    damage.name = "damage"
    solution.name = "saturation"
    
    # Create individual VTU files
    damage_vtk = VTKFile(MPI.COMM_WORLD, f"{results_dir}/damage_final.vtu", "w")
    solution_vtk = VTKFile(MPI.COMM_WORLD, f"{results_dir}/solution_final.vtu", "w")
    
    damage_vtk.write_function(damage, 0.0)
    solution_vtk.write_function(solution, 0.0)
    
    damage_vtk.close()
    solution_vtk.close()
    
    print(f"Convenient final crack membrane model results in .vtu format saved to {results_dir}/")


def create_crack_membrane_animation(dt, frame_delay, results_dir=".", use_logarithmic_saturation=False):
    """
    Create animated visualizations of the crack membrane healing process
    
    Parameters:
    -----------
    dt : float
        Time step size in seconds.
    frame_delay : float, optional
        Delay between frames in seconds (default: 0.1).
    use_logarithmic_saturation : bool, optional
        Whether to use logarithmic scaling for saturation field (default: False). 
        Lower values = faster animation creation, higher values = slower but more stable.
    """
    print("Creating crack membrane model healing animation...")
    
    print(f"Using time step: dt = {dt} s")
    print(f"Expected time range: 0 to {len(glob.glob('crack_membrane_model_output/solution_p0_*.vtu')) * dt} seconds")
    
    # Collect all timestep files
    files_solution = sorted(glob.glob("crack_membrane_model_output/solution_p0_*.vtu"))
    files_damage = sorted(glob.glob("crack_membrane_model_output/damage_p0_*.vtu"))
    files_diffusivity = sorted(glob.glob("crack_membrane_model_output/diffusivity_p0_*.vtu"))
    
    print(f"Found {len(files_solution)} solution files: {[os.path.basename(f) for f in files_solution[:5]]}")
    print(f"Found {len(files_damage)} damage files: {[os.path.basename(f) for f in files_damage[:5]]}")
    print(f"Found {len(files_diffusivity)} diffusivity files: {[os.path.basename(f) for f in files_diffusivity[:5]]}")
    
    if not files_solution or not files_damage or not files_diffusivity:
        print("No crack membrane model animation output files found.")
        return
    
    # Create the plotter and GIF for solution evolution (linear)
    plotter_solution = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter_solution.window_size = [1200, 1000]  # [width, height]
    plotter_solution.open_gif(os.path.join(results_dir, "saturation_evolution.gif"))
    
    print(f"Creating solution animation (linear) with {len(files_solution)} frames...")
    
    for i, file in enumerate(files_solution):
        mesh = pv.read(file)
        # Robust field detection with fallbacks
        if "saturation" in mesh.point_data:
            scalar_field = "saturation"
        elif "f" in mesh.point_data:  # Fallback to default name
            scalar_field = "f"
        else:
            print(f"Warning: No usable field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        # The first file (i=0) represents the initial condition at t=0
        actual_time = i * dt       
        
        # Linear scaling
        clim = [0, 1]
        plotter_solution.clear()
        plotter_solution.add_mesh(mesh, scalars=scalar_field, clim=clim, cmap="viridis", 
                                show_edges=False)
        plotter_solution.view_xy()
        plotter_solution.add_title(f"Crack Membrane Model - Saturation Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter_solution.render()
        plotter_solution.write_frame()
        time.sleep(frame_delay)
    
    plotter_solution.close()
    
    # Create logarithmic version if requested
    if use_logarithmic_saturation:
        print(f"Creating solution animation (logarithmic) with {len(files_solution)} frames...")
        
        plotter_solution_log = pv.Plotter()
        # Set window size to ensure full frame capture
        plotter_solution_log.window_size = [1200, 1000]  # [width, height]
        plotter_solution_log.open_gif(os.path.join(results_dir, "saturation_evolution_logarithmic.gif"))
        
        for i, file in enumerate(files_solution):
            mesh = pv.read(file)
            # Robust field detection with fallbacks
            if "saturation" in mesh.point_data:
                scalar_field = "saturation"
            elif "f" in mesh.point_data:  # Fallback to default name
                scalar_field = "f"
            else:
                print(f"Warning: No usable field found in {file}. Available fields: {list(mesh.point_data.keys())}")
                continue
                
            # Calculate actual time based on step number and dt
            actual_time = i * dt           
            
            # Logarithmic scaling
            epsilon = 1e-6  # Small value to avoid log(0) in color mapping
            clim = [epsilon, 1]
            plotter_solution_log.clear()
            plotter_solution_log.add_mesh(mesh, scalars=scalar_field, clim=clim, cmap="viridis", 
                                        show_edges=False, log_scale=True)
            plotter_solution_log.view_xy()
            plotter_solution_log.add_title(f"Crack Membrane Model - Saturation Field - t={actual_time:.2e}s")
            # Force render to ensure title is included in frame
            plotter_solution_log.render()
            plotter_solution_log.write_frame()
            time.sleep(frame_delay)
        
        plotter_solution_log.close()
    
    # Create the plotter and GIF for damage evolution
    plotter_damage = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter_damage.window_size = [1200, 1000]  # [width, height]
    plotter_damage.open_gif(os.path.join(results_dir, "damage_evolution.gif"))
    
    for i, file in enumerate(files_damage):
        mesh = pv.read(file)
        # Robust field detection with fallbacks
        if "damage" in mesh.point_data:
            scalar_field = "damage"
        elif "f" in mesh.point_data:  # Fallback to default name
            scalar_field = "f"
        else:
            print(f"Warning: No usable field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        actual_time = i * dt
        plotter_damage.clear()
        plotter_damage.add_mesh(mesh, scalars=scalar_field, clim=[0, 1], cmap="coolwarm", show_edges=False)
        plotter_damage.view_xy()
        plotter_damage.add_title(f"Crack Membrane Model - Damage Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter_damage.render()
        plotter_damage.write_frame()
        time.sleep(frame_delay)
    
    plotter_damage.close()
    
    # Create the plotter and GIF for diffusivity evolution
    plotter_diffusivity = pv.Plotter()
    # Set window size to ensure full frame capture
    plotter_diffusivity.window_size = [1200, 1000]  # [width, height]
    plotter_diffusivity.open_gif(os.path.join(results_dir, "diffusivity_evolution.gif"))
    
    for i, file in enumerate(files_diffusivity):
        mesh = pv.read(file)
        # Robust field detection with fallbacks
        if "diffusivity" in mesh.point_data:
            scalar_field = "diffusivity"
        elif "f" in mesh.point_data:  # Fallback to default name
            scalar_field = "f"
        else:
            print(f"Warning: No usable field found in {file}. Available fields: {list(mesh.point_data.keys())}")
            continue
            
        # Calculate actual time based on step number and dt
        actual_time = i * dt
        
        plotter_diffusivity.clear()
        plotter_diffusivity.add_mesh(mesh, scalars=scalar_field, cmap="plasma", show_edges=False)
        plotter_diffusivity.view_xy()
        plotter_diffusivity.add_title(f"Crack Membrane Model - Diffusivity Field - t={actual_time:.2e}s")
        # Force render to ensure title is included in frame
        plotter_diffusivity.render()
        plotter_diffusivity.write_frame()
        time.sleep(frame_delay)
    
    plotter_diffusivity.close()
    
    print("All 3 (saturation, damage, diffusivity) crack membrane healing animations completed!")


def plot_crack_membrane_healing_progress(dt, results_dir="."):
    """
    Plot the healing progress over time for the crack membrane model
    
    Parameters:
    -----------
    dt : float
        Time step size in seconds.
    """
    # Clear any existing plots and delete previous screenshot before starting
    plt.clf()
    plt.close('all')
    
    # Delete any existing screenshot file to keep things clean
    try:
        import os
        if os.path.exists('crack_membrane_model_healing_progress_over_time.png'):
            os.remove('crack_membrane_model_healing_progress_over_time.png')
            print("  Previous screenshot file deleted")
    except Exception as e:
        print(f"  Warning: Could not delete previous screenshot file: {e}")
    
    print("Creating healing percentage over time plot...")
    
    print(f"Using time step: dt = {dt} s")
    
    # Read the damage files to extract healing data
    files_damage = sorted(glob.glob("crack_membrane_model_output/damage_p0_*.vtu"))
    
    if not files_damage:
        print("No damage files found for healing percentage over time analysis.")
        return
    
    # Extract time information from filenames and calculate healing percentages
    times = []
    healing_percentages = []
    
    # Read initial damage
    initial_mesh = pv.read(files_damage[0])
    if "damage" in initial_mesh.point_data:
        initial_damage = np.sum(initial_mesh.point_data["damage"])
    elif "f" in initial_mesh.point_data:  # Fallback to default name
        initial_damage = np.sum(initial_mesh.point_data["f"])
    else:
        print(f"No initial damage field found. Available fields: {list(initial_mesh.point_data.keys())}")
        return
    
    for i, file in enumerate(files_damage):
        mesh = pv.read(file)
        if "damage" in mesh.point_data:
            current_damage = np.sum(mesh.point_data["damage"])
        elif "f" in mesh.point_data:  # Fallback to default name
            current_damage = np.sum(mesh.point_data["f"])
        else:
            continue
        
        # Calculate time using auto-detected dt
        t = i * dt
        times.append(t)
        
        # Calculate healing percentage
        healing_percentage = (initial_damage - current_damage) / initial_damage * 100
        healing_percentages.append(healing_percentage)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, healing_percentages, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Healed Damage Percentage (%)')
    plt.title('Crack Membrane Model - Healing Progress Over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'crack_membrane_model_healing_progress_over_time.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save healing progress data to CSV
    csv_filename = os.path.join(results_dir, 'crack_membrane_healing_progress_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Time (seconds)', 'Healing Percentage (%)'])
        # Write data
        for time_val, healing_pct in zip(times, healing_percentages):
            writer.writerow([time_val, healing_pct])
    
    print(f"Healing progress data saved to {csv_filename}")
    print("Healing percentage over time plot completed!")


def analyze_crack_membrane_results(domain, V, damage, solution, crack_S_crit, crack_dS_gate, results_dir="."):
    """
    Analyze the crack membrane model results and provide insights
    """
    print("\n" + "="*60)
    print("CRACK MEMBRANE RESULTS ANALYSIS")
    print("="*60)
    
    # Get mesh information
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    num_vertices = domain.topology.index_map(0).size_local
    
    print(f"Mesh: {num_cells} cells, {num_vertices} vertices")
    
    # Analyze damage field
    damage_array = damage.x.array
    max_damage = np.max(damage_array)
    min_damage = np.min(damage_array)
    mean_damage = np.mean(damage_array)
    
    print(f"\nFinal Damage Field Analysis:")
    print(f"  Final Max damage: {max_damage:.6f}")
    print(f"  Final Min damage: {min_damage:.6f}")
    print(f"  Final Mean damage: {mean_damage:.6f}")
    
    # Count high-damage cells (crack band)
    high_damage_threshold = 0.5
    high_damage_cells = np.sum(damage_array > high_damage_threshold)
    high_damage_percentage = (high_damage_cells / len(damage_array)) * 100
    
    print(f"  High damage cells (>0.5): {high_damage_cells} ({high_damage_percentage:.2f}%)")
    
    # Analyze solution field
    solution_array = solution.x.array
    max_solution = np.max(solution_array)
    min_solution = np.min(solution_array)
    mean_solution = np.mean(solution_array)
    
    print(f"\nFinal Saturation Field Analysis:")
    print(f"  Final Max saturation: {max_solution:.6f}")
    print(f"  Final Min saturation: {min_solution:.6f}")
    print(f"  Final Mean saturation: {mean_solution:.6f}")
    
    # Analyze saturation in high-damage regions
    high_damage_mask = damage_array > high_damage_threshold
    mean_crack_saturation = 0.0  # Initialize with default value
    
    if np.any(high_damage_mask):
        crack_saturation = solution_array[high_damage_mask]
        mean_crack_saturation = np.mean(crack_saturation)
        print(f"  Mean saturation in high damage regions (crack band): {mean_crack_saturation:.6f}")
    else:
        print(f"  No high damage regions found for crack saturation analysis")
    
    # Note: Water flow is now controlled by smart membrane diffusivity
    # The diffusivity field actively controls water transport based on moisture gate
    print(f"\nNote: Water flow through cracks is controlled by smart membrane diffusivity")
    print(f"  Moisture gate threshold: S_crit = {crack_S_crit}")
    print(f"  Gate sharpness: dS_gate = {crack_dS_gate}")
    
    # Provide insights
    print(f"\nInsights:")
    if mean_solution > 0.5:
        print(f"  - High overall saturation suggests good water penetration (mean saturation > 0.5)")
    else:
        print(f"  - Moderate saturation indicates controlled water flow (mean saturation < 0.5)")
    
    if mean_crack_saturation > mean_solution:
        print(f"  - Crack band shows higher saturation than matrix (expected for preferential flow)")
    else:
        print(f"  - Matrix shows higher saturation than crack band (smart membrane is controlling crack flow)")
    
    print(f"  - Smart membrane diffusivity control: Water flow through cracks is now naturally")
    print(f"    controlled by the moisture gate, creating a self-regulating barrier")
    
    # Save analysis results to CSV
    csv_filename = os.path.join(results_dir, 'crack_membrane_analysis_results.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Metric', 'Value', 'Description'])
        # Write damage analysis
        writer.writerow(['Max Damage', f'{max_damage:.6f}', 'Final maximum damage value'])
        writer.writerow(['Min Damage', f'{min_damage:.6f}', 'Final minimum damage value'])
        writer.writerow(['Mean Damage', f'{mean_damage:.6f}', 'Final mean damage value'])
        writer.writerow(['High Damage Cells', f'{high_damage_cells}', f'Cells with damage > {high_damage_threshold}'])
        writer.writerow(['High Damage Percentage', f'{high_damage_percentage:.2f}%', 'Percentage of high damage cells'])
        # Write saturation analysis
        writer.writerow(['Max Saturation', f'{max_solution:.6f}', 'Final maximum saturation value'])
        writer.writerow(['Min Saturation', f'{min_solution:.6f}', 'Final minimum saturation value'])
        writer.writerow(['Mean Saturation', f'{mean_solution:.6f}', 'Final mean saturation value'])
        writer.writerow(['Crack Saturation', f'{mean_crack_saturation:.6f}', 'Mean saturation in high damage regions'])
        # Write parameters
        writer.writerow(['S_crit', f'{crack_S_crit}', 'Moisture gate threshold'])
        writer.writerow(['dS_gate', f'{crack_dS_gate}', 'Gate sharpness parameter'])
    
    print(f"Analysis results saved to {csv_filename}")
    
def full_model(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, 
               beta, sigma, angle_step, sigma_min, sigma_max, sigma_step, plot_every, frame_delay=0.001, use_logarithmic_saturation=False):
    """
    Run the complete normal model with all outputs and post-processing
    
    Parameters:
    -----------
    N : int
        Mesh refinement parameter (NxN elements)
    D_concrete, D_air, alpha, p, gamma : float
        Model parameters
    T_init, T_final, num_steps : float, float, int
        Time parameters
    beta, sigma : float
        Crack parameters
    angle_step, sigma_min, sigma_max, sigma_step : float
        Parameter sweep settings
    plot_every : int
        Plot every N time steps
    frame_delay : float, optional
        Animation frame delay (default: 0.001)
    use_logarithmic_saturation : bool, optional
        Whether to create both linear and logarithmic saturation animations (default: False)
    """
    print("Running complete normal model...")
    
    # Create and clear Normal Model Results directory
    results_dir = "Normal Model Results"
    if os.path.exists(results_dir):
        print(f"Clearing {results_dir} directory...")
        shutil.rmtree(results_dir)
    print(f"Creating fresh {results_dir} directory...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear healing_diffusion directory before starting
    healing_dir = "healing_diffusion"
    if os.path.exists(healing_dir):
        print(f"Clearing {healing_dir} directory...")
        shutil.rmtree(healing_dir)
    print(f"Creating fresh {healing_dir} directory...")
    
    # Create sample mesh for initial damage visualization
    sample_domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    sample_V = functionspace(sample_domain, ("Lagrange", 1))
    
    # Plot the unit square mesh
    #plot_unit_square_mesh(sample_domain, results_dir)
    
    # Plot initial damage field before healing and save to results directory
    #plot_initial_damage(sample_domain, sample_V, beta, sigma, results_dir)
    
    # Run the main model
    #grid, damage, V = run_model(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma)
    
    # Test healing percentage over time with current parameters
    #times, healing_percentages = test_healing_percentage_over_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma, plot_every, results_dir, use_logarithmic_saturation=use_logarithmic_saturation)
    
    # Plot damage vs diffusivity and save to results directory
    #plot_damage_and_diffusivity(results_dir)
    
    # Run plotting and postprocessing with configurable plot coarseness
    #run_plotting(grid, damage, V, plot_every=plot_every, dt=(T_final - T_init) / num_steps, frame_delay=frame_delay, results_dir=results_dir, use_logarithmic_saturation=use_logarithmic_saturation)
    
    # Test healing time vs angle
    #angles, times = run_model_with_all_angles_find_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, angle_step, sigma)
    
    # Test healing time vs crack width
    sigmas, times = run_model_with_all_sigmas_find_time(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, sigma_min, sigma_max, sigma_step, results_dir)
    
    print("Normal model completed!")
    print(f"All results saved to '{results_dir}' directory")


def full_crack_membrane_model(N, dt, t_end, d_thr, m_open, p_log, D_conc, D_mild,
                             S_crit, dS_gate, eps_gate, alpha_heal, p_heal, gamma_heal,
                             beta, sigma, frame_delay, use_logarithmic_saturation=False):
    """
    Run the complete crack membrane model with all outputs and post-processing
    
    Parameters:
    -----------
    N, dt, t_end, d_thr, m_open, p_log : int, float, float, float, float, float
        Model setup parameters
    D_conc, D_mild : float
        Diffusivity parameters
    S_crit, dS_gate, eps_gate : float
        Moisture gate parameters
    alpha_heal, p_heal, gamma_heal : float
        Healing parameters
    beta, sigma : float
        Crack geometry parameters
    frame_delay : float
        Animation frame delay
    use_logarithmic_saturation : bool, optional
        Whether to use logarithmic scaling for saturation field animation (default: False)
    """
    print("Running complete crack membrane model...")
    
    # Create and clear Crack Membrane Model Results directory
    results_dir = "Crack Membrane Model Results"
    if os.path.exists(results_dir):
        print(f"Clearing {results_dir} directory...")
        shutil.rmtree(results_dir)
    print(f"Creating fresh {results_dir} directory...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create sample mesh for initial damage visualization
    sample_domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    sample_V = functionspace(sample_domain, ("Lagrange", 1))
    
    # Plot the unit square mesh
    plot_unit_square_mesh(sample_domain, results_dir)
    
    # Plot initial damage field before healing and save to results directory
    plot_initial_damage(sample_domain, sample_V, beta, sigma, results_dir)
    
    # Run the crack membrane model
    domain, V, damage, solution = run_crack_membrane_model(
        N=N, dt=dt, t_end=t_end, d_thr=d_thr, m_open=m_open, p_log=p_log, 
        D_conc=D_conc, D_mild=D_mild, S_crit=S_crit, dS_gate=dS_gate, 
        eps_gate=eps_gate, alpha_heal=alpha_heal, p_heal=p_heal, gamma_heal=gamma_heal,
        beta=beta, sigma=sigma
    )
    
    # Visualize crack membrane results and save to results directory
    visualize_crack_membrane_results(domain, V, damage, solution, results_dir)
    
    # Save crack membrane results to results directory
    save_crack_membrane_results(domain, V, damage, solution, results_dir)
    
    # Create animated visualizations and save to results directory
    create_crack_membrane_animation(dt=dt, frame_delay=frame_delay, results_dir=results_dir, 
                                   use_logarithmic_saturation=use_logarithmic_saturation)
    
    # Create healing progress plot and save to results directory
    plot_crack_membrane_healing_progress(dt=dt, results_dir=results_dir)
    
    # Analyze crack membrane results
    analyze_crack_membrane_results(domain, V, damage, solution, S_crit, dS_gate, results_dir)
    
    print("Crack membrane model completed!")
    print(f"All results saved to '{results_dir}' directory")


def compare_healing_progress(show_50_percent_indicators=True):
    """
    Compare healing progress between normal model and crack membrane model
    by overlaying their healing percentage vs time plots
    
    Parameters:
    -----------
    show_50_percent_indicators : bool, optional
        Whether to show 50% healing time indicators on the plot (default: True)
    """
    print("Creating comparison of healing progress between models...")
    
    # Create and clear Comparison Results directory
    comparison_dir = "Comparison Results"
    if os.path.exists(comparison_dir):
        print(f"Clearing {comparison_dir} directory...")
        shutil.rmtree(comparison_dir)
    print(f"Creating fresh {comparison_dir} directory...")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define file paths for data
    normal_csv = "Normal Model Results/healing_percentage_over_time_data.csv"
    crack_membrane_csv = "Crack Membrane Model Results/crack_membrane_healing_progress_data.csv"
    
    # Check if both data files exist
    if not os.path.exists(normal_csv):
        print(f"Error: Normal model data not found at {normal_csv}")
        print("Please run the normal model first to generate comparison data.")
        return
    
    if not os.path.exists(crack_membrane_csv):
        print(f"Error: Crack membrane model data not found at {crack_membrane_csv}")
        print("Please run the crack membrane model first to generate comparison data.")
        return
    
    # Read data from both CSV files
    try:
        # Read normal model data
        normal_data = []
        with open(normal_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                normal_data.append([float(row[0]), float(row[1])])
        
        # Read crack membrane model data
        crack_membrane_data = []
        with open(crack_membrane_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                crack_membrane_data.append([float(row[0]), float(row[1])])
        
        # Convert to numpy arrays for easier handling
        normal_times = np.array([row[0] for row in normal_data])
        normal_healing = np.array([row[1] for row in normal_data])
        crack_membrane_times = np.array([row[0] for row in crack_membrane_data])
        crack_membrane_healing = np.array([row[1] for row in crack_membrane_data])
        
        print(f"Normal model: {len(normal_data)} data points")
        print(f"Crack membrane model: {len(crack_membrane_data)} data points")
        
    except Exception as e:
        print(f"Error reading data files: {e}")
        return
    
    # Create the comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot both datasets with different colors
    plt.plot(normal_times, normal_healing, 'o-', color='#1f77b4', linewidth=2, 
             markersize=6, alpha=0.8, label='Crack Diffusion Model')
    plt.plot(crack_membrane_times, crack_membrane_healing, 'o-', color='#17becf', 
             linewidth=2, markersize=6, alpha=0.8, label='Crack Membrane Model')
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=20)
    plt.ylabel('Healing Percentage (%)', fontsize=20)
    plt.title('Crack Diffusion vs. Crack Membrane Model Healing Progress', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tick_params(labelsize=15)
    
    # Add legend (matplotlib default position)
    plt.legend(fontsize=20)
    
    # Add some statistics to the plot
    normal_final = normal_healing[-1] if len(normal_healing) > 0 else 0
    crack_membrane_final = crack_membrane_healing[-1] if len(crack_membrane_healing) > 0 else 0
    
    # Find milestone times for both models
    normal_50_idx = np.where(normal_healing >= 50)[0]
    normal_90_idx = np.where(normal_healing >= 90)[0]
    crack_membrane_50_idx = np.where(crack_membrane_healing >= 50)[0]
    crack_membrane_90_idx = np.where(crack_membrane_healing >= 90)[0]
    
    # Add milestone annotations (if enabled)
    if show_50_percent_indicators:
        if len(normal_50_idx) > 0:
            time_50_normal = normal_times[normal_50_idx[0]]
            plt.annotate(f'50% at {time_50_normal:.2e}s', 
                        xy=(time_50_normal, 50), xytext=(time_50_normal*0.3, 60),
                        arrowprops=dict(arrowstyle='->', color='#1f77b4', alpha=0.7),
                        fontsize=10, color='#1f77b4')
        
        if len(crack_membrane_50_idx) > 0:
            time_50_crack = crack_membrane_times[crack_membrane_50_idx[0]]
            plt.annotate(f'50% at {time_50_crack:.2e}s', 
                        xy=(time_50_crack, 50), xytext=(time_50_crack*1.1, 40),
                        arrowprops=dict(arrowstyle='->', color='#17becf', alpha=0.7),
                        fontsize=10, color='#17becf')
    
    plt.tight_layout()
    
    # Save the comparison plot
    plot_filename = os.path.join(comparison_dir, 'healing_progress_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_filename}")
    
    # Save combined data to CSV
    csv_filename = os.path.join(comparison_dir, 'combined_healing_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Time (seconds)', 'Normal Model Healing (%)', 'Crack Membrane Model Healing (%)'])
        
        # Find the maximum time range to interpolate both datasets
        max_time = max(np.max(normal_times), np.max(crack_membrane_times))
        min_time = min(np.min(normal_times), np.min(crack_membrane_times))
        
        # Create a common time grid for comparison
        common_times = np.linspace(min_time, max_time, 100)
        
        # Simple interpolation using numpy (no scipy dependency)
        def simple_interpolate(x_known, y_known, x_new):
            """Simple linear interpolation without scipy"""
            if len(x_known) == 1:
                return np.full_like(x_new, y_known[0])
            
            y_interp = np.zeros_like(x_new)
            for i, x in enumerate(x_new):
                # Find the two closest known points
                if x <= x_known[0]:
                    y_interp[i] = y_known[0]
                elif x >= x_known[-1]:
                    y_interp[i] = y_known[-1]
                else:
                    # Find the two surrounding points
                    for j in range(len(x_known) - 1):
                        if x_known[j] <= x <= x_known[j + 1]:
                            # Linear interpolation between these two points
                            x0, x1 = x_known[j], x_known[j + 1]
                            y0, y1 = y_known[j], y_known[j + 1]
                            y_interp[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                            break
            return y_interp
        
        # Interpolate both datasets
        normal_interpolated = simple_interpolate(normal_times, normal_healing, common_times)
        crack_membrane_interpolated = simple_interpolate(crack_membrane_times, crack_membrane_healing, common_times)
        
        # Write interpolated data
        for i, t in enumerate(common_times):
            writer.writerow([t, normal_interpolated[i], crack_membrane_interpolated[i]]) 
    
    print(f"Combined data saved to {csv_filename}")
    
    # Print comparison summary
    print(f"\nComparison Summary:")
    print(f"  Normal Model: Final healing = {normal_final:.2f}%")
    print(f"  Crack Membrane Model: Final healing = {crack_membrane_final:.2f}%")
    
    if len(normal_50_idx) > 0 and len(crack_membrane_50_idx) > 0:
        time_50_normal = normal_times[normal_50_idx[0]]
        time_50_crack = crack_membrane_times[crack_membrane_50_idx[0]]
        print(f"  50% healing: Normal = {time_50_normal:.2e}s, Crack = {time_50_crack:.2e}s")
        if time_50_normal < time_50_crack:
            print(f"  Normal model reaches 50% healing {time_50_crack/time_50_normal:.1f}x faster")
        else:
            print(f"  Crack membrane model reaches 50% healing {time_50_normal/time_50_crack:.1f}x faster")
    
    plt.show()
    print("Comparison analysis completed!")


if __name__ == "__main__":

    # Normal model parameters
    N = 32
    D_concrete = 0.00000001  # Concrete diffusivity (cm^2/s)
    D_air = 0.0000001  # Air diffusivity (cm^2/s)
    alpha = 0.01  # Healing rate constant
    p = 1  # Healing power parameter
    sigma = 0.0005 ** 0.5
    gamma = 0.001 ** 0.5
    T_init = 0.0
    T_final = 1500000  # 10x longer for complete healing; 1300000 for pretty much complete healing
    num_steps = 15000   # Much more time steps for smooth resolution 
    plot_every = 1
    angle_step = 5
    beta = 0
    sigma_min = 0.025  # Very narrow crack
    sigma_max = 0.040    # Very wide crack
    sigma_step = 0.001  # Step size for sigma values
    frame_delay = 0.001 # Delay between frames in seconds (speed of animation)
    
    # 3D Surface Plot of sigma vs gamma vs 95% healing time parameters
    sigma_min_3d = 0.015  # Minimum crack width for 3D plot
    sigma_max_3d = 0.025  # Maximum crack width for 3D plot
    sigma_step_3d = 0.0005  # Step size for sigma in 3D plot
    gamma_min_3d = 0.005  # Minimum smoothing parameter for 3D plot
    gamma_max_3d = 0.025  # Maximum smoothing parameter for 3D plot
    gamma_step_3d = 0.001  # Step size for gamma in 3D plot
    sigma_gamma_plot_num_steps = 4000  # Number of steps for sigma and gamma in 3D plot
    
    # SGB (Sigma-Gamma-Beta) Healing Time Data parameters
    sigma_start_sgb = 0.005  # Start value for sigma in SGB data
    sigma_end_sgb = 0.05    # End value for sigma in SGB data (0.1)
    sigma_step_sgb = 0.005   # Step size for sigma in SGB data
    gamma_start_sgb = 0.005  # Start value for gamma in SGB data
    gamma_end_sgb = 0.05    # End value for gamma in SGB data (0.1)
    gamma_step_sgb = 0.005   # Step size for gamma in SGB data 
    beta_start_sgb = 0    # Start value for beta in SGB data (radians)
    beta_end_sgb = np.pi/2  # End value for beta in SGB data (radians, π/2 = 90 degrees)
    beta_step_sgb = np.pi / 20  # Step size for beta in SGB data (radians, π/20 = 9 degrees)
    time_start_sgb = 0.0     # Start time for simulation (seconds)
    time_end_sgb = 3000000.0 # End time for simulation (seconds)
    time_num_steps_sgb = 10000 # Number of time steps for simulation
    
    # Crack membrane model parameters
    crack_N = 32  # Mesh refinement (64x64 for faster testing, can increase to 128)
    crack_dt = 20000  # Time step size in seconds (EXTREMELY small for immediate visible movement)
    crack_t_end = 4500000  # End time in seconds (10s total, but 100 time steps for smooth animation)
    crack_d_thr = 0.3  # Damage threshold for "band" cells (0.3 separates intact from cracked)
    crack_m_open = 1.0  # Openness exponent
    crack_p_log = 1.0  # Log-interp sharpness in matrix
    crack_D_conc = 0.00000001  # Intact concrete diffusivity (m^2/s)
    crack_D_mild = 0.0000001  # Slightly damaged diffusivity (m^2/s)
    
    # Animation visualization toggles
    use_logarithmic_saturation = True  # Toggle for logarithmic scaling of saturation field (crack membrane model)
    use_logarithmic_saturation_normal = True  # Toggle for logarithmic scaling of saturation field (normal model)
    show_50_percent_indicators = True  # Toggle for showing 50% healing time indicators in comparison plot
    crack_S_crit = 0.05  # Wetting threshold in saturation
    crack_dS_gate = 0.001  # Smoothing for logistic gate, sharpness of the moisture gate transition (smaller is sharper)
    crack_eps_gate = 0.00  # Tiny floor to avoid front pinning
    crack_p_heal = 1  # Healing power parameter for chi smoothing (basically q)

    crack_alpha_heal = 0.0001  # Healing rate constant (direct control - no scaling)
    crack_beta = 0  # Crack angle in radians (0 = vertical crack, π/4 = 45° crack)
    crack_sigma = 0.0005 ** 0.5  # Crack width parameter (smaller = narrower crack band)
    crack_frame_delay = 0.001  # Delay between frames in seconds (speed of animation)

    # Clear output directory before running
    clear_output_directory()

    # Choose which model to run
    run_normal_model = False      # Set to False to skip normal model
    run_crack_membrane = False   # Set to True to run crack membrane model
    run_comparison = False       # Set to True to run comparison analysis
    run_sigma_gamma_healingtime = False  # Set to True to create 3D surface plot
    run_important_angles_healing = False  # Set to True to analyze healing progress across important angles
    run_important_sigma_healing = False   # Set to True to analyze healing progress across important sigma values
    run_sgb_healing_time_data = True  # Set to True to generate SGB healing time data
    
    if run_normal_model:
        print("\n" + "="*60)
        print("RUNNING NORMAL MODEL")
        print("="*60)
        full_model(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, 
                  beta, sigma, angle_step, sigma_min, sigma_max, sigma_step, plot_every, frame_delay, use_logarithmic_saturation_normal)
    
    if run_crack_membrane:
        print("\n" + "="*60)
        print("RUNNING CRACK MEMBRANE MODEL")
        print("="*60)
        full_crack_membrane_model(crack_N, crack_dt, crack_t_end, crack_d_thr, 
                                crack_m_open, crack_p_log, crack_D_conc, crack_D_mild,
                                crack_S_crit, crack_dS_gate, crack_eps_gate, 
                                crack_alpha_heal, crack_p_heal, gamma, crack_beta, 
                                crack_sigma, crack_frame_delay, use_logarithmic_saturation)
    
    if run_comparison:
        print("\n" + "="*60)
        print("RUNNING COMPARISON ANALYSIS")
        print("="*60)
        compare_healing_progress(show_50_percent_indicators)
    
    if run_sigma_gamma_healingtime:
        print("\n" + "="*60)
        print("CREATING 3D SURFACE PLOT: HEALING TIME VS SIGMA AND GAMMA")
        print("="*60)
        create_sigma_gamma_healing_time_3d_plot(N, D_concrete, D_air, alpha, p, T_init, T_final, sigma_gamma_plot_num_steps, beta, 
                                               sigma_min_3d, sigma_max_3d, sigma_step_3d,
                                               gamma_min_3d, gamma_max_3d, gamma_step_3d,
                                               "3D Surface Plot Results")
    
    if run_important_angles_healing:
        print("\n" + "="*60)
        print("ANALYZING HEALING PROGRESS ACROSS IMPORTANT ANGLES")
        print("="*60)
        analyze_healing_progress_important_angles(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, sigma, "Healing Progress of Important Angles")
    
    if run_important_sigma_healing:
        print("\n" + "="*60)
        print("ANALYZING HEALING PROGRESS ACROSS IMPORTANT SIGMA VALUES")
        print("="*60)
        analyze_healing_progress_important_sigma(N, D_concrete, D_air, alpha, p, gamma, T_init, T_final, num_steps, beta, "Healing Progress of Important Sigma Values")
    
    if run_sgb_healing_time_data:
        print("\n" + "="*60)
        print("GENERATING SGB (SIGMA-GAMMA-BETA) HEALING TIME DATA")
        print("="*60)
        generate_sgb_healing_time_data(
            N, D_concrete, D_air, alpha, p,
            sigma_start_sgb, sigma_end_sgb, sigma_step_sgb,
            gamma_start_sgb, gamma_end_sgb, gamma_step_sgb,
            beta_start_sgb, beta_end_sgb, beta_step_sgb,
            T_init=time_start_sgb, T_final=time_end_sgb, num_steps=time_num_steps_sgb,
            results_dir="SGB Healing Time Data"
        )
    
    print("All done!")