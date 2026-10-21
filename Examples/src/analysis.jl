using CSV, DataFrames
using HypothesisTests

function analyze_problem_data(problem_name; subdir="")
    path = joinpath("results", subdir, "$problem_name.csv")
    if !isfile(path)
        println((problem_name, "missing"))
        return
    end
    df = DataFrame(CSV.File(path))

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
#     "drone_1e_2",
#     "drone_1e_3",
#     "drone_1e_4",
#     "drone_1e_9",
#     "drone_1e_2_saa",
#     "drone_1e_3_saa",
#     "room_temp_1e_2",
#     "room_temp_1e_3",
#     "room_temp_1e_4",
#     "room_temp_1e_9",
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

# Results of arXiv:2410.06662 (written by src/arxiv_2410_06662.jl), in the row order of Tables 1 and 2
probs_arxiv_2410_06662 = [
    # Table 1
    "linear1d_uncertain_1e_9",
    "drone_uncertain",
    "car_with_sidewind_no_partitioning",
    "car_with_sidewind_lon_partitioning",
    "car_with_sidewind_lat_partitioning",
    "car_with_sidewind_latlon_partitioning",
    "nndm_pendulum_1layer_120regions_1e_9",
    "nndm_pendulum_1layer_240regions_1e_9",
    "nndm_pendulum_1layer_480regions_1e_9",
    "nndm_pendulum_2layer_120regions_1e_9",
    "nndm_pendulum_2layer_240regions_1e_9",
    "nndm_pendulum_2layer_480regions_1e_9",
    "dubins_car_1000regions",
    # Table 2
    "linear1d_uncertain_1e_2",
    "linear1d_uncertain_1e_3",
    "linear1d_uncertain_1e_4",
    "linear1d_uncertain_1e_2_saa",
    "linear1d_uncertain_1e_3_saa",
    "linear1d_uncertain_1e_4_saa",
    "nndm_pendulum_1layer_480regions_1e_2",
    "nndm_pendulum_1layer_480regions_1e_3",
    "nndm_pendulum_1layer_480regions_1e_4",
    "nndm_pendulum_1layer_480regions_1e_2_saa",
    "nndm_pendulum_1layer_480regions_1e_3_saa",
    "nndm_pendulum_1layer_480regions_1e_4_saa",
]

if "arxiv_2410_06662" in ARGS
    for problem in probs_arxiv_2410_06662
        analyze_problem_data(problem; subdir="arxiv_2410_06662")
    end
else
    for problem in probs_to_analyze
        analyze_problem_data(problem)
    end
end