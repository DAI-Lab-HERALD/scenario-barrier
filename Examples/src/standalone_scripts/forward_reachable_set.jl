using LazySets, Polyhedra, JuMP
using BarrierFunctions

using Plots

function one_step_reachable_set(dyn, qᵢ)
    qᵢ = convert(HPolyhedron, qᵢ)
    qᵢ = convert(VPolyhedron, qᵢ)

    reachable_sets = [affine_map(Aᵢ, qᵢ, bᵢ) for (Aᵢ, bᵢ) in dyn]

    if length(reachable_sets) == 1
        return reachable_sets[1]
    end

    reachable_sets = UnionSetArray(reachable_sets)
    reachable_set = convex_hull(reachable_sets, prune=false)

    return reachable_set
end

dt = 1
v_long = 50 * (1000 / 3600) * dt
inward_steering = 0.05
ρ = [1.204, 1.342]
v_wind = [8, 10]
area = [4, 5.79865]
Cd = [0.8, 1.05]
m = [1125, 1350]
a_lat = [0.5 * ρ[1] * v_wind[1]^2 * area[1] * Cd[1] / m[2], 0.5 * ρ[2] * v_wind[2]^2 * area[2] * Cd[2] / m[1]]
Ab_gust = [AffineDynamics(Float64[1 0.2; -0.1 1-inward_steering], [v_long, 0.5 * a_lat[1] * dt^2]), AffineDynamics(Float64[1 0.2; 0.1 1-inward_steering], [v_long, 0.5 * a_lat[2] * dt^2])]

X = Hyperrectangle(Float64[0, 0], Float64[0.2, 0.2])
ηs = rand(VPolytope; num_vertices=1000)
ηs = remove_redundant_vertices(ηs)

Y = one_step_reachable_set(Ab_gust, X)

A₁, b₁ = Ab_gust[1]
A₂, b₂ = Ab_gust[2]
Ys = [affine_map(A₁ * λ + A₂ * (1 - λ), X, b₁ * λ + b₂ * (1 - λ)) for λ in LinRange(0, 1, 10)]

p = plot(Y, label="One step")
for Y2 in Ys
    plot!(p, Y2, fillalpha=0.0)
end
display(p)

p = plot(Y, label="One step forward reachable set")
plot!(p, Ys[1], label="λ₁ = 0")
plot!(p, Ys[end], label="λ₁ = 1")
savefig(p, "figures/forward_reachable_set.pdf")
display(p)