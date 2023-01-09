using Revise
using BarrierFunctions
using Gurobi, MosekTools, Mosek, GLPK, JuMP

using Plots
import PythonPlot: pygui

using Logging
using TerminalLoggers: TerminalLoggers, TerminalLogger

global_logger(TerminalLogger(right_justify=120))
using ProgressLogging, BenchmarkTools

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

function benchmark_experiment(problem)
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
    trial = with_logger(TerminalLogger(stderr, LogLevel(1); right_justify=120)) do
        return @benchmark(verify($solver, $(problem.system), H=$(problem.horizon), ϵ=$(problem.ϵ), β=$(problem.β)), samples=100, evals=1, seconds=10000)
    end

    return trial
end

function run_experiment(problem)
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
    res = []

    @progress for _ in 1:100
        with_logger(TerminalLogger(stderr, LogLevel(1); right_justify=120)) do
            succ, _, prob, _ = verify(solver, problem.system, H=problem.horizon, ϵ=problem.ϵ, β=problem.β)
            push!(res, (succ, prob))
        end
    end

    return res
end

function run_and_benchmark_experiment(problem)
    @info "Running experiment" name=problem.name

    trial = benchmark_experiment(problem)
    @info "Done with benchmark" name=problem.name 
    println(trial.times)

    res = run_experiment(problem)
    @info "Done with running" name=problem.name
    println(res)
end

function run_multiple_experiments(problems)
    for problem in problems
        run_and_benchmark_experiment(problem)
    end
end

# succ, prob = with_logger(TerminalLogger(stderr, LogLevel(1); right_justify=120)) do
#     solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
#     succ, _, prob, _ = verify(solver, system, H=H, ϵ=0.005/10, β=1e-9)
#     return succ, prob
# end
# @info "Done" succ prob
run_multiple_experiments(probs_car_with_sidewind)