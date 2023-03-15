using LazySets, Polyhedra, JuMP
using BarrierFunctions

using LinearAlgebra

using Plots

N = Float64
X = LazySets.translate(rand(HPolytope; seed=42, num_vertices=5), N[5, 7])
ηs = rand(VPolytope; num_vertices=1000)
ηs = remove_redundant_vertices(ηs)
ηs = ηs * 0.1

dirs = CustomDirections([h.a for h in constraints_list(X)])
under_ηs = underapproximate(ηs, dirs)
over_ηs = overapproximate(ηs, dirs)

function _underapproximate_directions(X::LazySet{N}, dirs) where {N, VN}
    H = Vector{LazySets.HalfSpace{N, VN}}()
    sizehint!(H, length(dirs))

    for d in dirs
        sf = ρ(d, X)
        if !isinf(sf)
            push!(H, LazySets.HalfSpace(d, sf))
        end
    end
    return HPolyhedron(H)
end

under_ηs2 = _underapproximate_directions(ηs, dirs)

p = plot(ηs, label="Noise")
plot!(p, under_ηs, label="Under-approximated noise")
plot!(p, over_ηs, label="Over-approximated noise")

display(p)

diff1 = minkowski_difference(X, under_ηs)
sum1 = minkowski_sum(diff1, ηs)
diff2 = minkowski_difference(X, ηs)
sum2 = minkowski_sum(diff2, ηs)

p = plot(X, label="X")
plot!(p, diff1, label="diff1")
plot!(p, sum1, label="sum1")
plot!(p, diff2, label="diff2")
plot!(p, sum2, label="sum2")

display(p)
# From this, we see that minkowski_sum and minkowski_difference are not inverse operations.
# Hence, we cannot exploit this when tightening the dynamics.