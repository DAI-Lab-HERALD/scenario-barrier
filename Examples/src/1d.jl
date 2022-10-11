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
Σ = PDiagMat([0.2])
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

linear1d = System(Q, A, b, η, X0, Xs, Xu)
split_spec = [
    [1],
    [1],
    [20],
    [50],
    [20]
]

linear1d = grid_split(linear1d, split_spec)