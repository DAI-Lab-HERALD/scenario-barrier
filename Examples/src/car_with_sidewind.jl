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
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 100], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 0, x[1] ≤ 10], vars),
    HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 10, x[1] ≤ 100], vars)
]

A = Matrix{Float64}[
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ]
]
b = Vector{Float64}[
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [50 * (1000 / 3600), 0], # 50 km/h in m/s (assuming 1s time steps)
    [50 * (1000 / 3600), 0]
]

μ = [0, 0]
Σ = PDiagMat([0.1, 0.1])
η = DiagNormal(μ, Σ)

# Start in a box [0m, 10m] × [-0.5m, 0.5m]
X0 = HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5, x[1] ≥ 0, x[1] ≤ 10], vars)

# On the road is safe
Xs = HPolyhedron([x[2] ≤ 2.5, x[2] ≥ -2.5], vars)

# Outside the road is dangerous
Xu = [
    HPolyhedron([x[2] ≥ 2.5], vars),
    HPolyhedron([x[2] ≤ -2.5], vars)
]

car_with_sidewind = System(Q, A, b, η, X0, Xs, Xu)