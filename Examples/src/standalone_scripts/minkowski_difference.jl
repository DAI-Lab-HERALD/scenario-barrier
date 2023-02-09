using LazySets, Polyhedra, JuMP
using BarrierFunctions

using LinearAlgebra

using Plots

N = Float64
X = LazySets.translate(rand(HPolytope; seed=42, num_vertices=5), N[5, 7])
ηs = rand(VPolytope; num_vertices=1000)
ηs = remove_redundant_vertices(ηs)
ηs = ηs * 0.1

diff = minkowski_difference(X, ηs)
sum = minkowski_sum(diff, ηs)

p = plot(X, label="X")
# plot!(p, ηs, label="noise")
plot!(p, diff, label="diff")
plot!(p, sum, label="sum")

display(p)
# From this, we see that minkowski_sum and minkowski_difference are not inverse operations.
# Hence, we cannot exploit this when tightening the dynamics.