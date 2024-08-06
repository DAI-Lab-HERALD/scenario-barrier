using LazySets, Polyhedra, JuMP
using BarrierFunctions

using LinearAlgebra

using Plots, LaTeXStrings
pgfplotsx()


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

function single_entry_vector(i, n)
    v = zeros(n)
    v[i] = 1.0

    return v
end

dynamics_vertices(dyn, lambda_polytope) = [sum(x -> x[1] * x[2], zip(dyn, vertex)) for vertex in vertices_list(lambda_polytope)]

function tighten_dynamics(dyn, qᵢ, qⱼ)
    n = length(dyn)
    if n == 1
        return dyn
    end

    lambda_polytope = VPolytope([single_entry_vector(i, n) for i in 1:n])

    # For each lambda_i
    for i in 1:n
        normal_vector = fill(1/n, n)
        normal_vector[i] = -1.0

        excluded_vertex = σ(-normal_vector, lambda_polytope)

        vertices = vertices_list(lambda_polytope)
        vertices = filter(vertex -> !all(isapprox.(vertex, excluded_vertex)), vertices)

        plane_polytope = VPolytope(vertices)
        plane_dyn = dynamics_vertices(dyn, plane_polytope)

        if isdisjoint(one_step_reachable_set(plane_dyn, qᵢ), qⱼ)
            lambda_polytope = binary_search_tighten_dynamics(i, dyn, qᵢ, qⱼ, lambda_polytope, normal_vector, excluded_vertex, first(vertices))

            if isnothing(lambda_polytope)
                return nothing
            end
        end
    end

    dyn = dynamics_vertices(dyn, lambda_polytope)

    return dyn
end

function binary_search_tighten_dynamics(i, dyn, qᵢ, qⱼ, lambda_polytope, normal_vector, min_vertex, max_vertex; max_steps=5)
    min = dot(normal_vector, min_vertex)
    max = dot(normal_vector, max_vertex)

    step = 1

    # Perform binary search
    while step <= max_steps
        mid = (min + max) / 2.0

        halfspace = LazySets.HalfSpace(-normal_vector, -mid)
        prune_polytope = intersection(lambda_polytope, halfspace)
        prune_dyn = dynamics_vertices(dyn, prune_polytope)

        forward_set = one_step_reachable_set(prune_dyn, qᵢ)
        if isdisjoint(forward_set, qⱼ)
            halfspace = complement(halfspace)
            keep_polytope = intersection(lambda_polytope, halfspace)
            keep_dyn = dynamics_vertices(dyn, keep_polytope)

            forward_set = one_step_reachable_set(keep_dyn, qᵢ)
            if isdisjoint(forward_set, qⱼ)
                return nothing
            end

            lambda_polytope = keep_polytope
            max = mid
        else
            min = mid
        end

        step += 1
    end

    return lambda_polytope
end

function tighten_region(dyn, qᵢ, qⱼ)
    constraints = constraints_list(qᵢ)
    # For each halfspace constraint
    for h in constraints
        normal_vector = h.a
        max = h.b
        min = -ρ(-normal_vector, qᵢ)

        if !isinf(min)
            qᵢ = binary_search_tighten_region(dyn, qᵢ, qⱼ, normal_vector, max, min)
        end
    end

    forward_set = one_step_reachable_set(dyn, qᵢ)
    if isdisjoint(forward_set, qⱼ)
        return nothing
    end

    return qᵢ
end

function binary_search_tighten_region(dyn, qᵢ, qⱼ, normal_vector, max, min; max_steps=5)
    step = 1

    # Perform binary search
    while step <= max_steps
        mid = (min + max) / 2.0

        halfspace = LazySets.HalfSpace(-normal_vector, -mid)
        prune_polytope = intersection(qᵢ, halfspace, prune=false)

        forward_set = one_step_reachable_set(dyn, prune_polytope)
        if isdisjoint(forward_set, qⱼ)
            qᵢ = intersection(qᵢ, complement(halfspace), prune=false)
            max = mid
        else
            min = mid
        end

        step += 1
    end

    return qᵢ
end

N = Float64
A₁ = N[1.0 0; 0 1.0]
A₂ = N[-1.0 0; 0 -1.0]
A₃ = 0.5 * A₁ + 0.5 * A₂
A₄ = A₁ - A₂

display(A₃)
display(A₄)
println(("det", det(A₁), det(A₂), det(A₃), det(A₄)))

qᵢ = LazySets.Hyperrectangle(low=N[4, -8], high=N[5, -7])
qᵢ_next = one_step_reachable_set([A₁, A₂], qᵢ)
qⱼ = LazySets.Hyperrectangle(low=N[2, 2], high=N[3, 3]) #LazySets.translate(rand(Hyperrectangle; seed=42), N[5, 7])

# lower, upper = low(qᵢ), high(qᵢ)
# x_range = LinRange(lower[1], upper[1], 4)
# y_range = LinRange(lower[2], upper[2], 4)

# f̂1lower(x, y) = (A₁ * [x; y])[1]
# f̂2lower(x, y) = (A₁ * [x; y])[2]
# f̂1upper(x, y) = (A₂ * [x; y])[1]
# f̂2upper(x, y) = (A₂ * [x; y])[2]

# green_cd = cgrad([:green, :green])
# red_cd = cgrad([:red, :red])

# p = Plots.surface(x_range, y_range, f̂1lower, color=green_cd, colorbar=false, xlabel="x_1", ylabel="x_2", title="y_1")
# surface!(p, x_range, y_range, f̂1upper, color=red_cd)
# display(p)

# p = Plots.surface(x_range, y_range, f̂2lower, color=green_cd, colorbar=false, xlabel="x_1", ylabel="x_2", title="y_2")
# surface!(p, x_range, y_range, f̂2upper, color=red_cd)
# display(p)

# dyn = tighten_dynamics([A₁, A₂], qᵢ, qⱼ)
# qᵢ_next2 = one_step_reachable_set(dyn, qᵢ)

qⱼ_intersect_fqᵢ = intersection(qᵢ_next, qⱼ)
# qⱼ_intersect_fqᵢ2 = intersection(qᵢ_next2, qⱼ)

# qᵢ_tightened = tighten_region(dyn, qᵢ, qⱼ)

# vs2 = vertices_list(qⱼ_intersect_fqᵢ2)
# vs_x2 = first.(vs2)
# vs_y2 = last.(vs2)

Ys = [InverseLinearMap(A₁ * λ + A₂ * (1 - λ), qⱼ) for λ in LinRange(0.0, 1.0, 10)]
Ys_vertex = ConvexHull(Ys[1], Ys[end])

# p = plot(qᵢ, label="X", legend=:bottomleft)
# plot!(p, qᵢ_tightened, label="qi tightened")
# plot!(p, qᵢ_next, label="conv(A(λ)X + b(λ))")
# plot!(p, qᵢ_next2, label="f(qi) = A(λ_pruned)q_i")
# plot!(p, qⱼ, label="qj")
# plot!(p, vs_x, vs_y, seriestype=:scatter, label="(qj ∩ f(qi)) vertices")
# plot!(p, vs_x2, vs_y2, seriestype=:scatter, label="(qj ∩ f(qi) pruned) vertices")

p = plot(qⱼ, label=L"$Q_j$", xlabel=L"$x_1$", ylabel=L"$x_2$", legend=:bottomright, size=(375, 240)) #, xlims=(x_min, x_max), ylims=(y_min, y_max))
plot!(p, Ys_vertex, label=L"$\mathrm{conv}(A_i(0)^{-1}Q_j, A_i(1)^{-1}Q_j)$") #, xlims=(x_min, x_max), ylims=(y_min, y_max))

for Y in Ys
    plot!(p, Y, fillalpha=0.0)
end
# plot!(p, vs_x, vs_y, seriestype=:scatter, label="(qj ∩ f(qi)) vertices")
# plot!(p, vs_x2, vs_y2, seriestype=:scatter, label="(qj ∩ f(qi) pruned) vertices")
vertices = vertices_list(qᵢ)
Ys = [map(p -> (A₁ * λ + A₂ * (1 - λ)) * p, vertices) for λ in LinRange(0.0, 1.0, 11)]
Y0 = linear_map(A₁, qᵢ)
Y1 = linear_map(A₂, qᵢ)
Ys_vertex = ConvexHull(Y0, Y1)
# plot!(p, Ys_vertex, label="conv(A(λ)Pᵢ + b(λ))") #, xlims=(x_min, x_max), ylims=(y_min, y_max))
# for Y in Ys
#     # scatter!(p, [Y[1][1]], [Y[1][2]], color=:blue, label=nothing)
#     # scatter!(p, [Y[2][1]], [Y[2][2]], color=:red, label=nothing)
#     # scatter!(p, [Y[3][1]], [Y[3][2]], color=:green, label=nothing)
#     # scatter!(p, [Y[4][1]], [Y[4][2]], color=:yellow, label=nothing)
#     poly = VPolytope(Y)
#     plot!(p, poly, fillalpha=0.0)
# end
savefig(p, "figures/non_convex_uncertain_affine_transformation.pdf")
display(p)
