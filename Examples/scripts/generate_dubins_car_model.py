"""Regenerate models/dubins_car/dynamics_1000regions.nc for the Dubin's car benchmark.

The PWA relaxation (linear bound propagation over a 10x10x10 grid) is computed with the Dubin's car
dynamics of Mathiesen, Calvert & Laurenti, "Safety certification for stochastic systems via neural
barrier functions" (CSL 2023), https://github.com/DAI-Lab-HERALD/neural-barrier-functions, using the
`dubin_walls.json` configuration. Only the nominal (noise-free) dynamics are bounded; the additive
noise is specified in Examples/src/systems/dubins_car.jl.

Usage (see generate_dubins_car_model.sh, which sets up the pinned environment):
    python generate_dubins_car_model.py <path-to-neural-barrier-functions-checkout> <output.nc>
"""
import os
import sys


def bound_nominal_dynamics(model, lower_x, upper_x):
    from bound_propagation import HyperRectangle
    from neural_barrier_functions.bounds import NBFBoundModelFactory
    from neural_barrier_functions.discretization import ButcherTableau, BoundButcherTableau
    from dubin.dynamics import DubinsCarNominalUpdate, BoundDubinsCarNominalUpdate, \
        DubinsFixedStrategy, BoundDubinsFixedStrategy, DubinSelect, BoundDubinSelect

    factory = NBFBoundModelFactory()
    factory.register(DubinsCarNominalUpdate, BoundDubinsCarNominalUpdate)
    factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
    factory.register(DubinSelect, BoundDubinSelect)
    factory.register(ButcherTableau, BoundButcherTableau)
    model = factory.build(model)

    input_bounds = HyperRectangle(lower_x, upper_x)
    dynamics_bounds = model.crown(input_bounds, alpha=True).cpu()

    return input_bounds.cpu(), dynamics_bounds


def save_bounds_to_netcdf(model, config, output_path):
    import numpy as np
    import torch
    import xarray as xr

    num_slices = config['partitioning']['num_slices']

    def grid_axis(low, high, n):
        space = torch.linspace(low, high, n + 1)
        return (space[1:] - space[:-1]) / 2, (space[:-1] + space[1:]) / 2

    x1_cell_widths, x1_slice_centers = grid_axis(-2.0, 2.0, num_slices[0])
    x2_cell_widths, x2_slice_centers = grid_axis(-2.0, 2.0, num_slices[1])
    x3_cell_widths, x3_slice_centers = grid_axis(-np.pi / 2, np.pi / 2, num_slices[2])

    cell_widths = torch.cartesian_prod(x1_cell_widths, x2_cell_widths, x3_cell_widths)
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers, x3_slice_centers)
    lower_x, upper_x = cell_centers - cell_widths, cell_centers + cell_widths

    input_set, dynamics_bounds = bound_nominal_dynamics(model, lower_x, upper_x)

    lu = ('dir', ['lower', 'upper'])
    x = ('x', list(range(1, 3 + 1)))
    y = ('y', list(range(1, 3 + 1)))
    region = ('region', list(range(1, len(input_set) + 1)))

    regions = xr.DataArray(
        name='regions',
        data=torch.stack((input_set.lower, input_set.upper), dim=1).numpy(),
        coords=[region, lu, x]
    )

    nominal_dynamics_A = xr.DataArray(
        name='nominal_dynamics_A',
        data=torch.stack((dynamics_bounds.lower[0], dynamics_bounds.upper[0]), dim=1).numpy(),
        coords=[region, lu, y, x]
    )

    nominal_dynamics_b = xr.DataArray(
        name='nominal_dynamics_b',
        data=torch.stack((dynamics_bounds.lower[1], dynamics_bounds.upper[1]), dim=1).numpy(),
        coords=[region, lu, y]
    )

    safe_set = xr.DataArray(
        name='safe_set',
        data=np.array([[-1.9, -1.9, -np.pi], [1.9, 1.9, np.pi]]),
        coords=[lu, x]
    )

    ds = xr.Dataset(
        data_vars=dict(
            safe_set=safe_set,
            regions=regions,
            nominal_dynamics_A=nominal_dynamics_A,
            nominal_dynamics_b=nominal_dynamics_b
        ),
        attrs=dict(
            num_regions=len(input_set)
        )
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path)


def main(nbf_path, output_path):
    nbf_path = os.path.abspath(nbf_path)
    output_path = os.path.abspath(output_path)

    sys.path.insert(0, nbf_path)
    sys.path.insert(0, os.path.join(nbf_path, "experiments"))

    import torch
    # As in neural-barrier-functions' experiments/main.py
    torch.set_default_dtype(torch.float64)
    # The config has zero noise on the deterministic dimensions, which torch rejects when validating
    # distribution arguments. The noise model is not used here, as only the nominal dynamics are bounded.
    torch.distributions.Distribution.set_default_validate_args(False)

    from neural_barrier_functions.utils import load_config
    from dubin.dubin import build_strategy
    from dubin.dynamics import DubinsCarStrategyComposition

    config = load_config(os.path.join(nbf_path, "experiments", "dubin", "config", "dubin_walls.json"))
    # The paper uses a grid of 10 segments along each axis, i.e. 1000 regions
    config["partitioning"]["num_slices"] = [10, 10, 10]
    strategy = build_strategy(config)
    dynamics = DubinsCarStrategyComposition(config["dynamics"], strategy)

    save_bounds_to_netcdf(dynamics.nominal_system, config, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
