# Experiments of
#   Mathiesen, F. B., Romao, L., Calvert, S. C., Laurenti, L., & Abate, A. (2024).
#   A data-driven approach for safety quantification of non-linear stochastic systems
#   with unknown additive noise distribution. arXiv:2410.06662.
#
# Usage (from the Examples directory):
#   julia --project=. src/arxiv_2410_06662.jl [table1] [table2] [table2_oom]
#
# Without arguments, Table 1 and Table 2 are run. `table2_oom` runs the SAA cases with β = 1e-4
# that are reported as OOM in Table 2 (they exhaust 16GB of memory).
# Logs and results are written to `logs/arxiv_2410_06662/` and `results/arxiv_2410_06662/`.

using BarrierFunctions

include("problem.jl")
include("systems/systems.jl")
include("runner.jl")

# All benchmarks in the paper use a horizon of 10 time steps and ϵ = 0.005 (spread over the horizon
# for the scenario approach). The horizon is fixed explicitly here instead of using the global `H`
# from the system files, which is overwritten by every file that defines it.
const T_ARXIV = 10
const ϵ_ARXIV = 0.005
scenario(β) = ScenarioApproach(ϵ_ARXIV / T_ARXIV, β)
saa(β) = SampleAverage(ϵ_ARXIV, β)
pwa_template() = PiecewiseTemplate(LinearTemplate())
βname(β) = "1e_$(round(Int, -log10(β)))"

const OUTDIR_ARXIV = "arxiv_2410_06662"

# Table 1: certified safety and computation time at confidence 1 - β = 1 - 1e-9
linear_problem(β) = Problem("linear1d_uncertain_$(βname(β))", martingale1d(true, true), T_ARXIV, pwa_template(), scenario(β))
car_problem(split, name) = Problem("car_with_sidewind_$(name)", car_with_sidewind(split=split), T_ARXIV, pwa_template(), scenario(1e-9))
pendulum_problem(layers, regions, β) = () -> Problem("nndm_pendulum_$(layers)layer_$(regions)regions_$(βname(β))", nndm_system("pendulum", layers, regions), T_ARXIV, pwa_template(), scenario(β))

table1 = [
    # Linear, n = 1, 27 regions
    () -> linear_problem(1e-9),
    # Drone with uncertain mass m ∈ [0.75, 1.25]
    () -> Problem("drone_uncertain", drone(true), T_ARXIV, pwa_template(), scenario(1e-9)),
    # Vehicle, n = 2, 18 / 42 / 54 / 150 regions
    () -> car_problem("none", "no_partitioning"),
    () -> car_problem("lon", "lon_partitioning"),
    () -> car_problem("lat", "lat_partitioning"),
    () -> car_problem("latlon", "latlon_partitioning"),
    # Pendulum NNDM with 1 hidden layer ("2 layers" in the paper), 120 / 240 / 480 regions
    pendulum_problem(1, 120, 1e-9),
    pendulum_problem(1, 240, 1e-9),
    pendulum_problem(1, 480, 1e-9),
    # Pendulum NNDM with 2 hidden layers ("3 layers" in the paper), 120 / 240 / 480 regions
    pendulum_problem(2, 120, 1e-9),
    pendulum_problem(2, 240, 1e-9),
    pendulum_problem(2, 480, 1e-9),
    # Dubin's car, n = 3, 1000 regions
    () -> Problem("dubins_car_1000regions", load_nc_system("models/dubins_car/dynamics_1000regions.nc"), T_ARXIV, pwa_template(), scenario(1e-9)),
]

# Table 2: comparison against SAA for β ∈ {1e-2, 1e-3, 1e-4}
linear_saa_problem(β) = Problem("linear1d_uncertain_$(βname(β))_saa", polynomial_martingale1d(true, true), T_ARXIV, PolynomialTemplate(4), saa(β))
pendulum_saa_problem(β) = () -> Problem("nndm_pendulum_1layer_480regions_$(βname(β))_saa", nndm_polynomial_system("pendulum", 1, 480), T_ARXIV, PolynomialTemplate(2), saa(β))

table2 = [
    () -> linear_problem(1e-2),
    () -> linear_problem(1e-3),
    () -> linear_problem(1e-4),
    () -> linear_saa_problem(1e-2),
    () -> linear_saa_problem(1e-3),
    pendulum_problem(1, 480, 1e-2),
    pendulum_problem(1, 480, 1e-3),
    pendulum_problem(1, 480, 1e-4),
    pendulum_saa_problem(1e-2),
    pendulum_saa_problem(1e-3),
]

table2_oom = [
    () -> linear_saa_problem(1e-4),
    pendulum_saa_problem(1e-4),
]

tables = Dict("table1" => table1, "table2" => table2, "table2_oom" => table2_oom)

function main(args)
    selected = isempty(args) ? ["table1", "table2"] : args
    for name in selected
        haskey(tables, name) || error("Unknown experiment group $name; choose from $(sort(collect(keys(tables))))")
    end

    ensure_nndm_models!()
    if "table1" in selected && !isfile("models/dubins_car/dynamics_1000regions.nc")
        error("Dubin's car model missing; see README for how to obtain models/dubins_car/dynamics_1000regions.nc")
    end

    for name in selected
        @info "Running experiment group" name
        run_experiments(tables[name]; subdir=OUTDIR_ARXIV)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
