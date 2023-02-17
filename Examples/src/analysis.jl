using CSV, DataFrames
using HypothesisTests

function analyze_problem_data(problem_name)
    df = DataFrame(CSV.File("results/$problem_name.csv"))

    succes_ratio = nrow(subset(df, :success => success -> success)) / nrow(df)

    t = OneSampleTTest(df[!, "computation_time"])
    lower, upper = confint(t)
    μₜ, wₜ = (lower + upper) / 2, (upper - lower) / 2

    df = filter(:success => s -> s, df)

    t = OneSampleTTest(df[!, "probability"])
    lower, upper = confint(t)
    μₛ, wₛ = (lower + upper) / 2, (upper - lower) / 2

    println((problem_name, succes_ratio, (μₛ, wₛ), (μₜ, wₜ)))
end

probs_to_analyze = [
    "linear1d_exact",
    "linear1d_uncertain",
    "car_with_sidewind_no_partitioning",
    "car_with_sidewind_lon_partitioning",
    "car_with_sidewind_lat_partitioning",
    "car_with_sidewind_latlon_partitioning",
    "drone_exact",
    "drone_uncertain",
    "nndm_pendulum_1layer_120regions",
    "nndm_pendulum_1layer_240regions",
    "nndm_pendulum_1layer_480regions",
    "nndm_pendulum_2layer_120regions",
    "nndm_pendulum_2layer_240regions",
    "nndm_pendulum_2layer_480regions",
    "nndm_pendulum_3layer_120regions",
    "nndm_pendulum_3layer_240regions",
    "nndm_pendulum_3layer_480regions",
]

for problem in probs_to_analyze
    analyze_problem_data(problem)
end