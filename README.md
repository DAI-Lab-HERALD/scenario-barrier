# Scenario barrier

Research code underlying the publications ... and ...

## What This Repository Contains

- [BarrierFunctions](BarrierFunctions) is the core Julia package with the barrier synthesis code.
- [Examples](Examples) is a separate Julia project that defines the research experiments, standalone scripts, and result analysis.

Most experiment entry points live in [Examples/src/main.jl](Examples/src/main.jl), with problem definitions in [Examples/src/systems](Examples/src/systems).

## Requirements

The codebase targets Julia 1.11.x, matching the compat entry in [Examples/Project.toml](Examples/Project.toml).

Recommended setup:

1. Install Julia with `juliaup`.
2. Use Julia 1.11 for this repository.
3. Install the solver licenses required by the experiment scripts, especially Mosek.

On Linux, the easiest Julia installation is:

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.11
juliaup default 1.11
```

If you already have Julia installed, make sure `julia --version` reports a 1.11.x release before reproducing results.

## Solver Setup

The main experiment batch in [Examples/src/main.jl](Examples/src/main.jl) uses [Mosek](https://www.mosek.com/) through JuMP and MosekTools.

To obtain an academic Mosek license:

1. Create or sign in to a Mosek account.
2. Request an academic license from the [Mosek licensing page](https://www.mosek.com/products/academic-licenses/).
3. Download the generated license file.
4. Place it where Mosek can find it, usually `~/.mosek/mosek.lic`, or set `MOSEKLM_LICENSE_FILE` to the license path.

If you plan to run scripts that explicitly use Gurobi, obtain the corresponding academic Gurobi license as well. The main batch does not require Gurobi, but you can swap Mosek for Gurobi in `main.jl`.

After installation, sanity-check Mosek from the `Examples` project:

```bash
cd Examples
julia --project=. -e 'using Mosek, JuMP; model = Model(Mosek.Optimizer); println("Mosek OK")'
```

## Reproducing the Main Experiments

All experiment commands should be run from the `Examples` directory because the scripts use relative paths for logs, results, and figures.

From the repository root:

```bash
cd Examples
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. src/main.jl
```

The first command downloads the exact package versions pinned by [Examples/Manifest.toml](Examples/Manifest.toml). The second command runs the full experiment batch.

`src/main.jl` does the following for each problem:

- creates a log file in `Examples/logs/<problem>.log`
- performs one warmup solve (making sure Julia has compiled the timed section)
- runs 100 timed repetitions
- writes a CSV file to `Examples/results/<problem>.csv`

The CSV contains one row per timed repetition with the fields `success`, `probability`, and `computation_time`.

If you want to rerun the post-processing on already generated CSV files, run:

```bash
cd Examples
julia --project=. src/analysis.jl
```

## Running Individual Scripts

The standalone scripts in [Examples/src/standalone_scripts](Examples/src/standalone_scripts) are useful for reproducing figures or single-case experiments without running the entire batch.

Example:

```bash
cd Examples
julia --project=. src/standalone_scripts/forward_reachable_set.jl
```

Other scripts follow the same pattern. They may write PDF figures into `Examples/figures/` or produce additional CSV/log output depending on the script.

## Adding a New Experiment

The existing experiment definitions are intentionally repetitive so that new experiments are easy to add and to audit.

The usual workflow is:

1. Add a system builder or experiment constructor in one of the files under [Examples/src/systems](Examples/src/systems).
2. Construct a `Problem` value using the helper type defined in [Examples/src/problem.jl](Examples/src/problem.jl).
3. Put the new problem into a named vector such as `probs_my_experiment = [...]`.
4. Include the new file from [Examples/src/systems/systems.jl](Examples/src/systems/systems.jl).
5. Call `run_experiment(...)` or `run_experiments(...)` from [Examples/src/main.jl](Examples/src/main.jl), or add a standalone script under [Examples/src/standalone_scripts](Examples/src/standalone_scripts).

Concrete pattern used by the current code:

- A function builds the system and returns a `PiecewiseSystem` or `PolynomialSystem`.
- One or more `Problem` objects wrap that system with a horizon, barrier template, and expectation method.
- A collection named `probs_*` groups related problems for batch execution.

If your new experiment depends on a different optimizer or package, add the dependency to [Examples/Project.toml](Examples/Project.toml) and re-run `Pkg.instantiate()`.

## Extending the Core Library

Core implementation changes belong in [BarrierFunctions/src](BarrierFunctions/src). The package entry point is [BarrierFunctions/src/BarrierFunctions.jl](BarrierFunctions/src/BarrierFunctions.jl), which includes the modules used by the examples.

Typical extension points include:

- `BarrierFunctions/src/verify.jl` for verification/synthesis logic
- `BarrierFunctions/src/barrier.jl` for barrier template structure
- `BarrierFunctions/src/expectation.jl` data-driven reformulations of the expectations
- `BarrierFunctions/src/partition.jl` and `BarrierFunctions/src/system.jl` for system representation and state space partitioning
- `BarrierFunctions/src/operations/` and `BarrierFunctions/src/sets/` for geometry and affine-map helpers

If you edit the core package, re-run package instantiation from both projects so the local path dependency stays in sync:

```bash
cd BarrierFunctions
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
cd ../Examples
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Reproducibility Notes

For research replication, record the following alongside any results:

- Julia version
- exact `Examples/Manifest.toml`
- solver versions and license type
- operating system and CPU architecture
- any random seeds used in custom experiments

The committed example batch is deterministic at the script level, but solver behavior and stochastic methods can still vary across versions and solver configurations. For exact replay, use the same manifest and the same solver installation that produced the original data.