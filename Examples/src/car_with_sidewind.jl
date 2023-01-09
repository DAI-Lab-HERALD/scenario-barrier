using BarrierFunctions
using LazySets, Symbolics

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
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 0, x[1] ≤ 10], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 0, x[1] ≤ 10], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 0, x[1] ≤ 10], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 10, x[1] ≤ 80], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 10, x[1] ≤ 80], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 10, x[1] ≤ 80], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 80, x[1] ≤ 120], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 80, x[1] ≤ 120], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 80, x[1] ≤ 120], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 120, x[1] ≤ 150], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 120, x[1] ≤ 150], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 120, x[1] ≤ 150], vars),
        HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 150], vars),
        HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 150], vars),
    ]

    v_long = 50 * (1000 / 3600) * dt
    Ab = [AffineDynamics(Float64[1 0; 0 0.8], [v_long, 0])]
    ρ = [1.204, 1.342]
    v_wind = [8, 10]
    A = [4, 5.79865]
    Cd = [0.8, 1.05]
    m = [1125, 1350]
    a_lat = [0.5 * ρ[1] * v_wind[1]^2 * A[1] * Cd[1] / m[2], 0.5 * ρ[2] * v_wind[2]^2 * A[2] * Cd[2] / m[1]]
    Ab_gust = [AffineDynamics(Float64[1 0; 0 0.8], [v_long, 0.5 * a_lat[1] * dt^2]), AffineDynamics(Float64[1 0; 0 1], [v_long, 0.5 * a_lat[2] * dt^2])]
    dyn = [Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab_gust, Ab_gust, Ab_gust, Ab, Ab, Ab, Ab, Ab, Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.01^2, 0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [5m, 10m] × [-0.5m, 0.5m]
    X0 = HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 5, x[1] ≤ 10], vars)

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
        split_spec[1, :] .= [1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1]
    end

    if startswith(split, "lat")
        split_spec[2, :] .= [1, 1, 1, 20, 10, 20, 20, 10, 20, 20, 10, 20, 20, 10, 20, 1, 1, 1]
    end
    Q, dyn = grid_split(Q, dyn, split_spec)

    return System(Q, dyn, η, X0, Xs, Xu)
end

prob_car_with_sidewind_no_paritioning = Problem("car_with_sidewind_no_paritioning", car_with_sidewind(split="none"), 10, 0.005/10, 1e-9)
prob_car_with_sidewind_lat_paritioning = Problem("car_with_sidewind_lat_paritioning", car_with_sidewind(split="lat"), 10, 0.005/10, 1e-9)
prob_car_with_sidewind_lon_paritioning = Problem("car_with_sidewind_lon_paritioning", car_with_sidewind(split="lon"), 10, 0.005/10, 1e-9)
prob_car_with_sidewind_latlon_paritioning = Problem("car_with_sidewind_latlon_paritioning", car_with_sidewind(split="latlon"), 10, 0.005/10, 1e-9)

probs_car_with_sidewind = [
    prob_car_with_sidewind_no_paritioning,
    prob_car_with_sidewind_lat_paritioning,
    prob_car_with_sidewind_lon_paritioning,
    prob_car_with_sidewind_latlon_paritioning
]