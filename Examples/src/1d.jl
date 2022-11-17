using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

vars = Symbolics.@variables x

Q = [
    HPolyhedron([x ≥ 2.5], vars),
    HPolyhedron([x ≤ -2.5], vars),
    HPolyhedron([x ≤ 2.5, x ≥ 0.5], vars),
    HPolyhedron([x ≤ 0.5, x ≥ -0.5], vars),
    HPolyhedron([x ≤ -0.5, x ≥ -2.5], vars)
]

A = Matrix{Float64}[
    [1][:,:],
    [1][:,:],
    [1][:,:],
    [1][:,:],
    [1][:,:]
]
b = Vector{Float64}[
    [0],
    [0],
    [0],
    [0],
    [0]
]

μ = [0]
Σ = PDiagMat([0.01^2])  # Square to make std.dev. to variance.
η = DiagNormal(μ, Σ)

# Start in a box [-0.5, 0.5]
X0 = HPolyhedron([x ≤ 0.5, x ≥ -0.5], vars)

# Center is safe
Xs = HPolyhedron([x ≤ 2.5, x ≥ -2.5], vars)

# Outside the center is dangerous
Xu = [
    HPolyhedron([x ≥ 2.5], vars),
    HPolyhedron([x ≤ -2.5], vars)
]

# split_spec = [
#     [1],
#     [1],
#     [30],
#     [10],
#     [30]
# ]
# Q, A, b = grid_split(Q, A, b, split_spec)

linear1d = System(Q, A, b, η, X0, Xs, Xu)