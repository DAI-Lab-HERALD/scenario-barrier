using Revise

using LazySets, Polyhedra, JuMP
using BarrierFunctions

using Random, Distributions, PDMats

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

function custom_minkowski_sum(qᵢ_next_nom::AbstractPolyhedron{N}, ηs) where {N}
    hrep_qᵢ_next_nom = HPolyhedron(constraints_list(qᵢ_next_nom))
    nomial_template_dirs = CustomDirections([normalize(h.a) for h in constraints_list(hrep_qᵢ_next_nom)])
    
    hrep_ηs = tohrep(ηs)
    noise_template_dirs = CustomDirections([normalize(h.a) for h in constraints_list(hrep_ηs)])

    H = Vector{LazySets.HalfSpace{N, Vector{N}}}()
    sizehint!(H, length(nomial_template_dirs) + length(noise_template_dirs))

    for d in nomial_template_dirs
        sf = LazySets.ρ(d, qᵢ_next_nom) + LazySets.ρ(d, ηs)
        if !isinf(sf)
            push!(H, LazySets.HalfSpace(d, sf))
        end
    end

    for d in noise_template_dirs
        sf = LazySets.ρ(d, qᵢ_next_nom) + LazySets.ρ(d, ηs)
        if !isinf(sf)
            push!(H, LazySets.HalfSpace(d, sf))
        end
    end

    qᵢ_next = HPolyhedron(H)
    qᵢ_next = normalize(qᵢ_next)

    return qᵢ_next
end

dt = 1
v_long = 50 * (1000 / 3600) * dt
inward_steering = 0.05
A = Float64[1 0.2; 0 1-inward_steering]
Ab = [AffineDynamics(A, [v_long, 0])]
ρ = [1.204, 1.342]
v_wind = [8, 10]
area = [4, 5.79865]
Cd = [0.8, 1.05]
m = [1125, 1350]
a_lat = [0.5 * ρ[1] * v_wind[1]^2 * area[1] * Cd[1] / m[2], 0.5 * ρ[2] * v_wind[2]^2 * area[2] * Cd[2] / m[1]]
Ab_gust = [AffineDynamics(A, [v_long, 0.5 * a_lat[1] * dt^2]), AffineDynamics(A, [v_long, 0.5 * a_lat[2] * dt^2])]

X = Hyperrectangle(low=[10, 0.5], high=[24, 0.6])

μ = [0, 0]
Σ = PDiagMat([0.01^2, 0.01^2])  # Square to make std.dev. to variance.
η = MvNormal(μ, Σ)
ηs = VPolytope([copy(η) for η in eachcol(rand(η, 10000))])
ηs = remove_redundant_vertices(ηs)
println(ηs)

# Y = one_step_reachable_set(Ab_gust, X)
# Ys = [affine_map(A, X, b) for (A, b) in Ab_gust]

p = plot(custom_minkowski_sum(Y, ηs), label="Z")
plot!(p, Y, label="Y")
display(p)

p = plot(minkowski_sum(Y, ηs), label="Z")
plot!(p, Y, label="Y")
display(p)

# difference = map(x -> x[1] .- x[2], zip(sort(vertices_list(custom_minkowski_sum(Y, ηs))), sort(vertices_list(minkowski_sum(Y, ηs)))))
# println(difference)

indirect_overapproximation = box_approximation(custom_minkowski_sum(Y, ηs))
println(low(indirect_overapproximation), high(indirect_overapproximation))

direct_overapproximation = box_approximation(MinkowskiSum(Y, ηs))
println(low(direct_overapproximation), high(direct_overapproximation))
