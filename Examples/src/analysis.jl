using CSV, DataFrames
using HypothesisTests

function analyze_problem_data(problem_name)
    df = DataFrame(CSV.File("results/$problem_name.csv"))

    succes_ratio = nrow(subset(df, :success => success -> success)) / nrow(df)

    t = OneSampleTTest(df[!, "computation_time"])
    lower, upper = confint(t)
    μₜ, wₜ = (lower + upper) / 2, (upper - lower) / 2

    df = filter(:success => s -> s, df)

    if nrow(df) >= 1
        t = OneSampleTTest(df[!, "probability"])
        lower, upper = confint(t)
        μₛ, wₛ = (lower + upper) / 2, (upper - lower) / 2
    else
        μₛ, wₛ = NaN, NaN
    end

    println((problem_name, succes_ratio, (μₛ, wₛ), (μₜ, wₜ)))
end

probs_to_analyze = [
    # "linear1d_exact",
    "linear1d_uncertain",
    "car_with_sidewind_no_partitioning",
    "car_with_sidewind_lon_partitioning",
    "car_with_sidewind_lat_partitioning",
    "car_with_sidewind_latlon_partitioning",
    "drone_exact",
    # "drone_uncertain",
    "nndm_pendulum_1layer_120regions",
    "nndm_pendulum_1layer_240regions",
    "nndm_pendulum_1layer_480regions",
    "nndm_pendulum_2layer_120regions",
    "nndm_pendulum_2layer_240regions",
    "nndm_pendulum_2layer_480regions",
    "dubins_car_1000regions",
    "room_temp",
    "lane_keeping_2520regions_181410"
]

# probs_to_analyze = [
#     "linear1d_uncertain_1e_2",
#     "linear1d_uncertain_1e_3",
#     "linear1d_uncertain_1e_4",
#     "linear1d_uncertain_1e_9",
#     "linear1d_uncertain_1e_2_saa",
#     "linear1d_uncertain_1e_3_saa",
#     "drone_1e_2",
#     "drone_1e_3",
#     "drone_1e_4",
#     "drone_1e_9",
#     "drone_1e_2_saa",
#     "drone_1e_3_saa",
#     "nndm_pendulum_1layer_480regions_1e_2",
#     "nndm_pendulum_1layer_480regions_1e_3",
#     "nndm_pendulum_1layer_480regions_1e_4",
#     "nndm_pendulum_1layer_480regions_1e_9",
#     "nndm_pendulum_1layer_480regions_1e_2_saa",
#     "nndm_pendulum_1layer_480regions_1e_3_saa",
# ]

# probs_to_analyze = [
#     "cdc_car_with_sidewind_no_partitioning",
#     "cdc_car_with_sidewind_lat_partitioning",
#     "cdc_car_with_sidewind_lon_partitioning",
#     "cdc_car_with_sidewind_latlon_partitioning",
#     "cdc_car_with_sidewind_saa",
#     "cdc_drone_exact",
#     "cdc_drone_exact_saa",
#     "cdc_linear1d",
#     "cdc_linear1d_ssa",
# ]

for problem in probs_to_analyze
    analyze_problem_data(problem)
end