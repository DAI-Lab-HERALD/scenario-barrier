using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

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
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 80, x[1] ≤ 90], vars),
    HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 80, x[1] ≤ 90], vars),
    HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 80, x[1] ≤ 90], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 90, x[1] ≤ 150], vars),
    HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 90, x[1] ≤ 150], vars),
    HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 90, x[1] ≤ 150], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ 0.5, x[1] ≥ 150], vars),
    HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 150], vars),
    HPolyhedron([x[2] ≤ -0.5, x[2] ≥ -2.5, x[1] ≥ 150], vars),
]

v_long = 50 * (1000 / 3600) * dt
Ab = [AffineDynamics(Float64[1 0; 0 1], [v_long, 0])]
ρ = [1.204, 1.342]
v_wind = [8, 10]
A = [4, 5.79865]
Cd = [0.8, 1.05]
m = [1125, 1350]
a_lat = [0.5 * ρ[1] * v_wind[1]^2 * A[1] * Cd[1] / m[2], 0.5 * ρ[2] * v_wind[2]^2 * A[2] * Cd[2] / m[1]]
Ab_gust = [AffineDynamics(Float64[1 0; 0 1], [v_long, 0.5 * a_lat[1] * dt^2]), AffineDynamics(Float64[1 0; 0 1], [v_long, 0.5 * a_lat[2] * dt^2])]
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

car_with_sidewind = System(Q, dyn, η, X0, Xs, Xu)