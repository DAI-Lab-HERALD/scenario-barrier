using BarrierFunctions
using LazySets, Symbolics
using MAT

using Distributions, PDMats

load_mat_system(system_name, bound_method, num_layers, num_regions) = load_mat_system("models/$system_name/$bound_method/$(num_layers)_layers/partition_data_$num_regions.mat")

function load_mat_system(path)
    file = matopen(path)

    regions = read(file, "partitions")
    state_space = read(file, "state_space")

    # Read Bounds
    Mₗ = read(file, "M_l")
    Bₗ = read(file, "B_l")
    Mₕ = read(file, "M_h")
    Bₕ = read(file, "B_h")

    n = size(state_space, 1)
    vars = Symbolics.@variables x[1:n]

    # Regions
    Q = []
    for region in eachrow(regions)
        push!(Q, HPolyhedron([region[1] .≤ x, x .≤ region[2]], vars))
    end

    # Dynamics
    dyn = []
    for (Mₗᵢ, Bₗᵢ, Mₕᵢ, Bₕᵢ) in zip(eachrow(Mₗ), eachrow(Bₗ), eachrow(Mₕ), eachrow(Bₕ))
        Mₗᵢ = transpose(Mₗᵢ)  # Why is the transpose necessary?
        Mₕᵢ = transpose(Mₕᵢ)

        dyn_i = [AffineDynamics(Mₗᵢ, Bₗᵢ), AffineDynamics(Mₕᵢ, Bₕᵢ)]
        push!(dyn, dyn_i)
    end

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # TODO: Figure out how to deal with the initial region
    X0 = HPolyhedron([x[2] ≤ 0.5, x[2] ≥ -0.5, x[1] ≥ 5, x[1] ≤ 10], vars)

    # Safe
    Xs = HPolyhedron([state_space[:, 1] .≤ x, x .≤ state_space[:, 2]], vars)

    # Unsafe
    remaining_region = Universe()

    Xu = []
    for (i, dim_bounds) in enumerate(eachrow(state_space))
        unsafe_lower = HPolyhedron([x[i] ≤ dim_bounds[1]], vars)
        unsafe_upper = HPolyhedron([x[i] ≥ dim_bounds[2]], vars)

        push!(Xu, unsafe_lower, unsafe_upper)

        unsafe_lower = intersection(unsafe_lower, remaining_region)
        unsafe_upper = intersection(unsafe_upper, remaining_region)

        push!(Q, unsafe_lower, unsafe_upper)

        # Just zero dynamics - not important as this region is unsafe and thus never propagated
        dyn_i = [AffineDynamics(zero(Mₗ[1]), zero(Bₗ[1]))]
        push!(dyn, dyn_i)

        remaining_region = intersection(remaining_region, HPolyhedron([dim_bounds[1] ≤ x[i], x[i] ≤ dim_bounds[2]], vars))
    end

    crown_dyn_system = System(Q, dyn, η, X0, Xs, Xu)
    return crown_dyn_system
end