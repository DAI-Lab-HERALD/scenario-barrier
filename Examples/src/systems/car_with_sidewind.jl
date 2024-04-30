using BarrierFunctions
using LazySets, Symbolics
using Spatial

using Distributions, PDMats

function car_with_sidewind(;split="none")
    # x[1] = longitudinal coordinate in meters
    # x[2] = lateral coordinate in meters
    vars = Symbolics.@variables x[1:2]

    dt = 1.0

    Q = [
        HPolyhedron([x[2] ≥ 2.5], vars),
        HPolyhedron([x[2] ≤ -2.5], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≤ 0], vars),
        Hyperrectangle(low=[0, 0.5], high=[10, 2.5]),
        Hyperrectangle(low=[0, -0.5], high=[10, 0.5]),
        Hyperrectangle(low=[0, -2.5], high=[10, -0.5]),
        Hyperrectangle(low=[10, 0.5], high=[80, 2.5]),
        Hyperrectangle(low=[10, -0.5], high=[80, 0.5]),
        Hyperrectangle(low=[10, -2.5], high=[80, -0.5]),
        Hyperrectangle(low=[80, 0.5], high=[120, 2.5]),
        Hyperrectangle(low=[80, -0.5], high=[120, 0.5]),
        Hyperrectangle(low=[80, -2.5], high=[120, -0.5]),
        Hyperrectangle(low=[120, 0.5], high=[150, 2.5]),
        Hyperrectangle(low=[120, -0.5], high=[150, 0.5]),
        Hyperrectangle(low=[120, -2.5], high=[150, -0.5]),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 150], vars),
    ]

    v_long = 50 * (1000 / 3600) * dt
    inward_steering = 0.05
    A = Float64[1 0; 0 1-inward_steering]
    Ab = [AffineDynamics(A, [v_long, 0])]
    ρ = [1.204, 1.342]
    v_wind = [8, 10]
    area = [4, 5.79865]
    Cd = [0.8, 1.05]
    m = [1125, 1350]
    a_lat = [0.5 * ρ[1] * v_wind[1]^2 * area[1] * Cd[1] / m[2], 0.5 * ρ[2] * v_wind[2]^2 * area[2] * Cd[2] / m[1]]
    Ab_gust = [AffineDynamics(A, [v_long, 0.5 * a_lat[1] * dt^2]), AffineDynamics(A, [v_long, 0.5 * a_lat[2] * dt^2])]
    dyn = [Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab_gust, Ab_gust, Ab_gust, Ab, Ab, Ab, Ab, Ab, Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.01^2, 0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [5m, 10m] × [-0.5m, 0.5m]
    X0 = Hyperrectangle(low=[5, -0.5], high=[10, 0.5])

    # On the road is safe
    Xs = HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5], vars)

    # Outside the road is dangerous
    Xu = [
        HPolyhedron([x[2] ≥ 2.5], vars),
        HPolyhedron([x[2] ≤ -2.5], vars)
    ]

    split_spec = Int64[
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    ]

    if endswith(split, "lon")
        split_spec[1, :] .= [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1]
    end

    if startswith(split, "lat")
        split_spec[2, :] .= [1, 1, 1, 5, 2, 5, 5, 2, 5, 5, 2, 5, 5, 2, 5, 1, 1, 1]
    end
    Q, dyn = grid_split(Q, dyn, split_spec)

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_car_with_sidewind_no_partitioning = Problem("car_with_sidewind_no_partitioning", car_with_sidewind(split="none"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
prob_car_with_sidewind_lat_partitioning = Problem("car_with_sidewind_lat_partitioning", car_with_sidewind(split="lat"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
prob_car_with_sidewind_lon_partitioning = Problem("car_with_sidewind_lon_partitioning", car_with_sidewind(split="lon"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
prob_car_with_sidewind_latlon_partitioning = Problem("car_with_sidewind_latlon_partitioning", car_with_sidewind(split="latlon"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))

probs_car_with_sidewind = [
    prob_car_with_sidewind_no_partitioning,
    prob_car_with_sidewind_lat_partitioning,
    prob_car_with_sidewind_lon_partitioning,
    prob_car_with_sidewind_latlon_partitioning
];

prob_car_with_sidewind_no_partitioning_saa = Problem("car_with_sidewind_no_partitioning_saa", car_with_sidewind(split="none"), 10, PolynomialBarrier, SampleAverage(0.005, 1e-9))
prob_car_with_sidewind_lat_partitioning_saa = Problem("car_with_sidewind_lat_partitioning_saa", car_with_sidewind(split="lat"), 10, PolynomialBarrier, SampleAverage(0.005, 1e-9))
prob_car_with_sidewind_lon_partitioning_saa = Problem("car_with_sidewind_lon_partitioning_saa", car_with_sidewind(split="lon"), 10, PolynomialBarrier, SampleAverage(0.005, 1e-9))
prob_car_with_sidewind_latlon_partitioning_saa = Problem("car_with_sidewind_latlon_partitioning_saa", car_with_sidewind(split="latlon"), 10, PolynomialBarrier, SampleAverage(0.005, 1e-9))

probs_car_with_sidewind_saa = [
    prob_car_with_sidewind_no_partitioning_saa,
    prob_car_with_sidewind_lat_partitioning_saa,
    prob_car_with_sidewind_lon_partitioning_saa,
    prob_car_with_sidewind_latlon_partitioning_saa
];

function cdc_car_with_sidewind(;split="none")
    # x[1] = longitudinal coordinate in meters
    # x[2] = lateral coordinate in meters
    vars = Symbolics.@variables x[1:2]

    dt = 1.0

    Q = [
        HPolyhedron([x[2] ≥ 2.5], vars),
        HPolyhedron([x[2] ≤ -2.5], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≤ 0], vars),
        Hyperrectangle(low=[0, 0.5], high=[10, 2.5]),
        Hyperrectangle(low=[0, -0.5], high=[10, 0.5]),
        Hyperrectangle(low=[0, -2.5], high=[10, -0.5]),
        Hyperrectangle(low=[10, 0.5], high=[80, 2.5]),
        Hyperrectangle(low=[10, -0.5], high=[80, 0.5]),
        Hyperrectangle(low=[10, -2.5], high=[80, -0.5]),
        Hyperrectangle(low=[80, 0.5], high=[120, 2.5]),
        Hyperrectangle(low=[80, -0.5], high=[120, 0.5]),
        Hyperrectangle(low=[80, -2.5], high=[120, -0.5]),
        Hyperrectangle(low=[120, 0.5], high=[150, 2.5]),
        Hyperrectangle(low=[120, -0.5], high=[150, 0.5]),
        Hyperrectangle(low=[120, -2.5], high=[150, -0.5]),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 150], vars),
    ]

    v_long = 50 * (1000 / 3600) * dt
    inward_steering = 0.05
    A = Float64[1 0; 0 1-inward_steering]
    Ab = [AffineDynamics(A, [v_long, 0])]
    ρ = 1.342
    v_wind = 10
    area = 5.79865
    Cd = 1.05
    m = 1125
    a_lat = 0.5 * ρ * v_wind^2 * area * Cd / m
    Ab_gust = [AffineDynamics(A, [v_long, 0.5 * a_lat * dt^2])]
    dyn = [Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab_gust, Ab_gust, Ab_gust, Ab, Ab, Ab, Ab, Ab, Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.01^2, 0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [5m, 10m] × [-0.5m, 0.5m]
    X0 = Hyperrectangle(low=[5, -0.5], high=[10, 0.5])

    # On the road is safe
    Xs = HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5], vars)

    # Outside the road is dangerous
    Xu = [
        HPolyhedron([x[2] ≥ 2.5], vars),
        HPolyhedron([x[2] ≤ -2.5], vars)
    ]

    split_spec = Int64[
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    ]

    if endswith(split, "lon")
        split_spec[1, :] .= [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1]
    end

    if startswith(split, "lat")
        split_spec[2, :] .= [1, 1, 1, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 1, 1, 1]
    end
    Q, dyn = grid_split(Q, dyn, split_spec)

    elems = [DynamicsRegion(i, qi, dyni) for (i, (qi, dyni)) in enumerate(zip(Q, dyn))]

    Q = LinearIndex{Float64, DynamicsRegion{Float64}}()
    bulk_load!(Q, elems)

    return PiecewiseSystem{2, Float64}(Q, η, X0, Xs, Xu)
end

cdc_prob_car_with_sidewind_no_partitioning = Problem("cdc_car_with_sidewind_no_partitioning", cdc_car_with_sidewind(split="none"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
cdc_prob_car_with_sidewind_lat_partitioning = Problem("cdc_car_with_sidewind_lat_partitioning", cdc_car_with_sidewind(split="lat"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
cdc_prob_car_with_sidewind_lon_partitioning = Problem("cdc_car_with_sidewind_lon_partitioning", cdc_car_with_sidewind(split="lon"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))
cdc_prob_car_with_sidewind_latlon_partitioning = Problem("cdc_car_with_sidewind_latlon_partitioning", cdc_car_with_sidewind(split="latlon"), 10, PiecewiseBarrier, ScenarioApproach(0.005/10, 1e-9))

cdc_prob_car_with_sidewind_saa = Problem("cdc_car_with_sidewind_saa", cdc_car_with_sidewind(), 10, PolynomialBarrier, SampleAverage(0.005, 1e-6))