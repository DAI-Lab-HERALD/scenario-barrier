using Revise
using BarrierFunctions
using Gurobi, MosekTools, Mosek, GLPK, JuMP

using Plots
import PythonPlot: pygui

using Logging
using TerminalLoggers: TerminalLoggers, TerminalLogger
global_logger(TerminalLogger(right_justify=120))
using ProgressLogging

using CSV, DataFrames

pythonplot()
pygui(true)

include("problem.jl")
include("1d.jl")
include("car_with_sidewind.jl")
include("drone.jl")
include("nndm.jl")

# system = nndm_system("pendulum")
# H = 1
# system = car_with_sidewind(split="latlon")
# H = 10
# traj = monte_carlo(system, H, 10000)
# plot_monte_carlo_2d(system, traj, plot_safe=false)
# plot_forward_reachable_set_2d(system, 1000, plot_safe=false, xlims=[-10, 160], ylims=[-10, 10])

function run_experiment(problem)
    @info "Running experiment" name=problem.name
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
    res = []

    # Warmup
    @info "Warmup" name=problem.name
    verify(solver, problem.system, H=problem.horizon, ϵ=problem.ϵ, β=problem.β)

    # Experiment run
    @progress for _ in 1:100
        with_logger(TerminalLogger(stderr, LogLevel(1); right_justify=120)) do
            start_time = time()
            succ, _, prob, _ = verify(solver, problem.system, H=problem.horizon, ϵ=problem.ϵ, β=problem.β)
            exec_time = time() - start_time
            push!(res, (success=succ, probability=prob, computation_time=exec_time))
        end
    end

    @info "Done running" name=problem.name
    println(res)

    df = DataFrame(res)

    mkpath("results", mode=0o755)
    CSV.write("results/$(problem.name).csv", df)

    @info "Finished writing results"
end
function run_experiments(problems)
    for problem in problems
        run_experiment(problem)
    end
end

# run_experiments(probs_car_with_sidewind)
run_experiments(probs_drone)
run_experiments(probs_linear1d)