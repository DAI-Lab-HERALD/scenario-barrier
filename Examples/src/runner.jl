using BarrierFunctions
using MosekTools, Mosek, JuMP

using Logging, LoggingExtras

using CSV, DataFrames

# Number of timed repetitions per problem. The publications report the mean over 100 repetitions;
# set BARRIER_REPETITIONS to a smaller value for a quick smoke test of the pipeline.
const NUM_REPETITIONS = parse(Int, get(ENV, "BARRIER_REPETITIONS", "100"))

function run_experiment(make_problem::Function; kwargs...)
    problem = make_problem()
    run_experiment(problem; kwargs...)
end

function run_experiment(problem::Problem; subdir="", repetitions=NUM_REPETITIONS)
    logdir = joinpath("logs", subdir)
    resultdir = joinpath("results", subdir)
    mkpath(logdir, mode=0o755)
    mkpath(resultdir, mode=0o755)

    logpath = joinpath(logdir, "$(problem.name).log")

    console_logger = MinLevelLogger(ConsoleLogger(stderr), Logging.Debug)
    file_logger = MinLevelLogger(FileLogger(logpath), Logging.Debug)
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
    file_logger = MinLevelLogger(FileLogger(logpath; append=true), Logging.Warn)
    experiment_logger = TeeLogger(console_logger, file_logger)

    # Experiment run
    for i in 1:repetitions
        @info "Iteration $i/$repetitions"
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
    CSV.write(joinpath(resultdir, "$(problem.name).csv"), df)

    @info "Finished writing results"
end

function run_experiments(problems; kwargs...)
    for problem in problems
        run_experiment(problem; kwargs...)
    end
end
