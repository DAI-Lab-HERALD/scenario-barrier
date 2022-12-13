using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

# x[1] = longitudinal coordinate in meters
# x[2] = lateral coordinate in meters
vars = @variables x[1:2]

Q = [
    HPolyhedron([x[2] ≥ 2.5], vars),
    HPolyhedron([x[2] ≤ -2.5], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≤ 0], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 150], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 0, x[1] ≤ 10], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 10, x[1] ≤ 80], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 80, x[1] ≤ 90], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 90, x[1] ≤ 150], vars)
]

dyn = ConvexHullDynamics{Float64}[
    singleton_dynamics([
        1 0;
        0 0.7;
    ], Float64[0, 0]),
    singleton_dynamics([
        1 0;
        0 0.7;
    ], Float64[0, 0]),
    singleton_dynamics([
        1 0;
        0 0.7;
    ], Float64[0, 0]),
    singleton_dynamics([
        1 0;
        0 0.7;
    ], Float64[0, 0]),
    singleton_dynamics([
        1 0;
        0 0.7;
    ], [50 * (1000 / 3600), 0]), # 50 km/h in m/s (assuming 1s time steps
    singleton_dynamics([
        1 0;
        0 0.7;
    ], [50 * (1000 / 3600), 0]),
    singleton_dynamics([
        1 0;
        0 0.7;
    ], [50 * (1000 / 3600), 1.0]), # A gust of wind from the side
    singleton_dynamics([
        1 0;
        0 0.7;
    ], [50 * (1000 / 3600), 0]),
]

μ = [0, 0]
Σ = PDiagMat([0.01, 0.01])
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

split_spec = [
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [3, 5],
    [5, 5],
    [3, 5],
    [5, 5]
]
Q, dyn = grid_split(Q, dyn, split_spec)

car_with_sidewind = System(Q, dyn, η, X0, Xs, Xu)