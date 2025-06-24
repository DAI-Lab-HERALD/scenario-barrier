using Revise
using BarrierFunctions
using Gurobi, MosekTools, Mosek, GLPK, JuMP

using Plots
import PythonPlot: pygui

using Logging, LoggingExtras
using ProgressLogging

using CSV, DataFrames

pythonplot()
pygui(true)

include("problem.jl")
include("systems/systems.jl")
# system = nndm_system("pendulum")
# H = 1
# system = car_with_sidewind(split="latlon")
# H = 10
# traj = monte_carlo(system, H, 10000)
# plot_monte_carlo_2d(system, traj, plot_safe=false)
# plot_forward_reachable_set_2d(system, 1000, plot_safe=false, xlims=[-10, 160], ylims=[-10, 10])

function run_experiment(make_problem::Function)
    problem = make_problem()
    run_experiment(problem)
end

function run_experiment(problem::Problem)
    console_logger = MinLevelLogger(ConsoleLogger(stderr), Logging.Debug)
    file_logger = MinLevelLogger(FileLogger("logs/$(problem.name).log"), Logging.Debug)
    logger = TeeLogger(console_logger, file_logger)
    global_logger(logger)

    @info "Running experiment" name=problem.name
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER)
    res = []

    # Warmup
    @info "Warmup" name=problem.name
    succ, B, prob = @time verify(solver, problem.barrier_type, problem.system, problem.horizon, problem.expectation_method)
    println((succ, prob))
    GC.gc(true)

    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)

    console_logger = MinLevelLogger(ConsoleLogger(stderr), Logging.Warn)
    file_logger = MinLevelLogger(FileLogger("logs/$(problem.name).log"; append=true), Logging.Warn)
    experiment_logger = TeeLogger(console_logger, file_logger)

    # Experiment run
    for i in 1:4
        @info "Iteration $i/4"
        with_logger(experiment_logger) do
            start_time = time()
            succ, _, prob = verify(solver, problem.barrier_type, problem.system, problem.horizon, problem.expectation_method)
            exec_time = time() - start_time
            push!(res, (success=succ, probability=prob, computation_time=exec_time))
        end

        flush(stderr)
        GC.gc(true)
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

run_experiments(probs_car_with_sidewind)
run_experiments(probs_drone)
run_experiments(probs_linear1d)
run_experiments(probs_nndm_pendulum_1layer)
run_experiments(probs_nndm_pendulum_2layer)
run_experiment(prob_dubins_car_1000regions)
run_experiment(prob_room_temp)
run_experiment(prob_lane_keeping_2520regions_181410)

run_experiments(probs_drone_1e)
run_experiments(probs_drone_1e_saa)
run_experiments(probs_room_temp_1e)
run_experiments(probs_room_temp_1e_saa)
run_experiments(probs_nndm_pendulum_1layer_1e)
run_experiments(probs_nndm_pendulum_1layer_1e_saa)
