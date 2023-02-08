using LazySets, Polyhedra, JuMP
using BarrierFunctions

using LinearAlgebra

using Plots


function one_step_reachable_set(dyn, qᵢ)
    qᵢ = convert(HPolyhedron, qᵢ)
    qᵢ = convert(VPolyhedron, qᵢ)

    reachable_sets = [linear_map(Aᵢ, qᵢ) for Aᵢ in dyn]

    if length(reachable_sets) == 1
        return reachable_sets[1]
    end

    reachable_sets = UnionSetArray(reachable_sets)
    reachable_set = convex_hull(reachable_sets, prune=true)

    return reachable_set
end

function tighten_dynamics(dyn, qᵢ, qⱼ)
    dyn_polytope = VPolytope([vec(A) for A in dyn])
    dyn_polytope_hrep = tohrep(dyn_polytope)

    println(constraints_list(dyn_polytope_hrep))

    return dyn
end

A₁ = [1 -1; 0 1]
A₂ = [1 2; -1 0]

println((opnorm(A₁), opnorm(A₂)))
println((det(A₁), det(A₂)))

qᵢ = LazySets.Hyperrectangle(low=Float64[-5, 7], high=Float64[-4, 8])
qᵢ_next = one_step_reachable_set([A₁, A₂], qᵢ)
qⱼ = LazySets.translate(rand(HPolytope; seed=42, num_vertices=20), [5, 7])

dyn = tighten_dynamics([A₁, A₂], qᵢ, qⱼ)
qᵢ_next2 = one_step_reachable_set([A₁, A₂], qᵢ)

qⱼ_intersect_fqᵢ = intersection(qᵢ_next, qⱼ)

vs = vertices_list(qⱼ_intersect_fqᵢ)
vs_x = first.(vs)
vs_y = last.(vs)

Ys = [InverseLinearMap(A₁ * λ + A₂ * (1 - λ), qⱼ_intersect_fqᵢ) for λ in LinRange(0, 1, 100)]

convex_avg_vs = [inv(A₁ * 0.5 + A₂ * 0.5) * v for v in vs]
convex_avg_x = first.(convex_avg_vs)
convex_avg_y = last.(convex_avg_vs)
# smallest_eigval = eigvals(A₁ * 0.5 + A₂ * 0.5)[1]
# smallest_eigval = eigvals(A₂)[1]
# l2_ball_vs = [Ball2(v, 1/real(smallest_eigval)) for v in convex_avg_vs]

p = plot(qᵢ, label="qi")
plot!(p, qᵢ_next, label="f(qi) = A(λ)q_i")
plot!(p, qⱼ, label="qj")
plot!(p, vs_x, vs_y, seriestype=:scatter, label="qj ∩ f(qi) vertices")
# plot!(p, convex_avg_x, convex_avg_y, seriestype=:scatter, label="vertices back")
# for ball in l2_ball_vs
#     plot!(p, ball, fillalpha=0.0)
# end
for Y in Ys
    # if !isdisjoint(Y, qᵢ)
    plot!(p, Y, fillalpha=0.0)
    # end
end

for Y in [linear_map(A₁ * λ + A₂ * (1 - λ), qᵢ) for λ in LinRange(0, 1, 20)]
    plot!(p, Y, fillalpha=0.0)
end
display(p)

# p = plot(qⱼ, label="Next set")
# plot!(p, convex_avg_x, convex_avg_y, seriestype=:scatter)
# for ball in l2_ball_vs
#     plot!(p, ball, fillalpha=0.0)
# end
# plot!(p, Ys[1], fillalpha=0.0)
# plot!(p, Ys[end], fillalpha=0.0)
# display(p)