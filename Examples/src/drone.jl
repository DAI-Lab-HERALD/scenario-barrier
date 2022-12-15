using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

# Only modeling longitudinal dynamics
# x[1] = position
# x[2] = velocity
vars = @variables x[1:2]

Q = [
    HPolyhedron([x[1] ≤ 0], vars),
    HPolyhedron([x[1] ≥ 0, x[1] ≤ 12, x[2] ≥ -10, x[2] ≤ 10], vars),
    HPolyhedron([x[1] ≥ 12, x[2] ≥ -10, x[2] ≤ 10], vars),
    HPolyhedron([x[2] ≥ 10], vars),
    HPolyhedron([x[2] ≤ -10], vars),
]

m = 1.0
τ = 1.0

Ab = singleton_dynamics([
    1 τ;
    0 (m - 0.1 * τ) / m
], Float64[0, 0])
dyn = [Ab, Ab, Ab, Ab, Ab]

μ = [0, 0]
Σ = PDiagMat([0.1, 0.1])
η = DiagNormal(μ, Σ)

# Start in a box [0.5m, 1.0m] × [-0.5m/s, 0.5m/s]
X0 = HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 0.5, x[1] ≤ 1.0], vars)

# On the road is safe
Xs = HPolyhedron([x[2] ≤ 10, x[2] ≥ -10, x[1] ≥ 0, x[1] ≤ 12], vars)

# Outside the road is dangerous
Xu = [
    HPolyhedron([x[2] ≥ 10], vars),
    HPolyhedron([x[2] ≤ -10], vars),
    HPolyhedron([x[1] ≥ 12], vars),
    HPolyhedron([x[1] ≤ 0], vars)
]

split_spec = [
    [1, 1],
    [5, 5],
    [1, 1],
    [1, 1],
    [1, 1]
]
Q, dyn = grid_split(Q, dyn, split_spec)

drone = System(Q, dyn, η, X0, Xs, Xu)