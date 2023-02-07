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

function expectation_set(dyn, qᵢ, qⱼ, expectation)
    Hᵢ, hᵢ = tosimplehrep(qᵢ)
    Hⱼ, hⱼ = tosimplehrep(qⱼ)
    Hη, hη = tosimplehrep(expectation.data.hrep)
    
    Hᵢ = hcat(Hᵢ, zero(Hᵢ))
    Hη = hcat(zero(Hη), Hη)

    backward_reachable_sets = [HPolyhedron(vcat(Hᵢ, Hη, hcat(Hⱼ * Aᵢ, Hⱼ)), vcat(hᵢ, hη, hⱼ - Hⱼ * bᵢ)) for (Aᵢ, bᵢ) in dyn]

    if length(backward_reachable_sets) == 1
        return backward_reachable_sets[1]
    end

    # backward_reachable_sets = [HPolyhedron(hcat(Hⱼ * Aᵢ, Hⱼ), hⱼ - Hⱼ * bᵢ) for (Aᵢ, bᵢ) in dyn]
    println(map(isempty, backward_reachable_sets))
    backward_reachable_sets = map(p -> convert(VPolyhedron, p), backward_reachable_sets)
    println(map(length ∘ vertices_list, backward_reachable_sets))

    backward_reachable_sets = filter(p -> !isempty(p), backward_reachable_sets)
    backward_reachable_sets = UnionSetArray{Float64, VPolyhedron{Float64, Vector{Float64}, Vector{Float64}}}(backward_reachable_sets)

    backward_reachable_set = convex_hull(backward_reachable_sets, prune=false)
    println(isempty(backward_reachable_set))
    backward_reachable_set = normalize(tohrep(backward_reachable_set))
    println(isempty(backward_reachable_set))

    println(length(constraints_list(backward_reachable_set)))

    return backward_reachable_set
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

X = Hyperrectangle(Float64[0, 0], Float64[0.5, 0.5])
ηs = rand(VPolytope; num_vertices=1000)
ηs = remove_redundant_vertices(ηs)

Y = one_step_reachable_set(Ab_gust, X)
Ys = [affine_map(A, X, b) for (A, b) in Ab_gust]

# p = plot(Y, label="One step")
# for Y2 in Ys
#     plot!(p, Y2)
# end
# display(p)

# p = plot(custom_minkowski_sum(Y, ηs))
# display(p)

# p = plot(minkowski_sum(Y, ηs))
# display(p)

difference = map(x -> x[1] .- x[2], zip(sort(vertices_list(custom_minkowski_sum(Y, ηs))), sort(vertices_list(minkowski_sum(Y, ηs)))))
println(difference)

# indirect_overapproximation = box_approximation(custom_minkowski_sum(Y, ηs))
# println(low(indirect_overapproximation), high(indirect_overapproximation))

# direct_overapproximation = box_approximation(MinkowskiSum(Y, ηs))
# println(low(direct_overapproximation), high(direct_overapproximation))