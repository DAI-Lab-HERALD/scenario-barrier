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
A = [
    1, τ;
    0, (m - 0.1 * τ) / m
]
A = Matrix{Float64}[A, A, A, A, A]
b = [0, 0]
b = Vector{Float64}[b, b, b, b, b]

μ = [0, 0]
Σ = PDiagMat([0.1, 0.1])
η = DiagNormal(μ, Σ)

# Start in a box [5m, 10m] × [-0.5m, 0.5m]
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
    [24, 20],
    [1, 1],
    [1, 1],
    [1, 1]
]
Q, A, b = grid_split(Q, A, b, split_spec)

drone = System(Q, A, b, η, X0, Xs, Xu)