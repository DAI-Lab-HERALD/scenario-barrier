using BarrierFunctions
using LazySets, Symbolics
using MAT, ZipFile

using Distributions, PDMats

load_mat_system(system_name, bound_method, num_layers, num_regions, initial_region_cb) = load_mat_system("models/$system_name/$bound_method/$(num_layers)_layers/partition_data_$num_regions.mat", initial_region_cb)

function load_mat_system(path, initial_region_cb)
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

    eachfirstdim(A) = eachslice(A, dims=(1,))

    # Regions
    Q = []
    for region in eachfirstdim(regions)
        push!(Q, HPolyhedron([[region[1, i] ≤ x[i] for i in 1:n]; [x[i] ≤ region[2, i] for i in 1:n]], vars))
    end

    # Dynamics
    dyn = []
    for (Mₗᵢ, Bₗᵢ, Mₕᵢ, Bₕᵢ) in zip(eachfirstdim(Mₗ), eachfirstdim(Bₗ), eachfirstdim(Mₕ), eachfirstdim(Bₕ))
        Mₗᵢ = transpose(Mₗᵢ)  # Why is the transpose necessary?
        Mₕᵢ = transpose(Mₕᵢ)

        dyn_i = [AffineDynamics(copy(Mₗᵢ), copy(Bₗᵢ[:, 1])), AffineDynamics(copy(Mₕᵢ), copy(Bₕᵢ[:, 1]))]
        push!(dyn, dyn_i)
    end

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    X0 = initial_region_cb(x, vars)

    # Safe
    Xs = HPolyhedron([[state_space[i, 1] ≤ x[i] for i in 1:n]; [x[i] ≤ state_space[i, 2] for i in 1:n]], vars)

    # Unsafe
    remaining_region = Universe(n)

    Xu = HPolyhedron{Float64, Vector{Float64}}[]
    for (i, dim_bounds) in enumerate(eachfirstdim(state_space))
        unsafe_lower = HPolyhedron([x[i] ≤ dim_bounds[1]], vars)
        unsafe_upper = HPolyhedron([x[i] ≥ dim_bounds[2]], vars)

        push!(Xu, unsafe_lower, unsafe_upper)

        unsafe_lower = intersection(unsafe_lower, remaining_region)
        unsafe_upper = intersection(unsafe_upper, remaining_region)

        push!(Q, unsafe_lower, unsafe_upper)

        # Just zero dynamics - not important as this region is unsafe and thus never propagated
        dyn_i = [AffineDynamics(zero(Mₗ[1, :, :]), zero(Bₗ[1, :, 1]))]
        push!(dyn, dyn_i, dyn_i)

        remaining_region = intersection(remaining_region, HPolyhedron([dim_bounds[1] ≤ x[i], x[i] ≤ dim_bounds[2]], vars))
    end

    return System(Q, dyn, η, X0, Xs, Xu)
end

initial_region_pendulum(x, vars) = HPolyhedron([-deg2rad(5) ≤ x[1], x[1] ≤ deg2rad(5), x[2] == 0], vars)
initial_region_cartpole(x, vars) = HPolyhedron([x[1] == 0, x[2] == 0, -deg2rad(5) ≤ x[3], x[3] ≤ deg2rad(5), x[4] == 0], vars)
initial_region_husky4d(x, vars) = HPolyhedron([x[1] ≤ 0.1, x[1] ≥ -0.1, x[2] ≤ 0.1, x[2] ≥ -0.1, x[3] == 0, x[4] == 0], vars)
initial_region_husky5d(x, vars) = HPolyhedron([x[1] ≤ 0.1, x[1] ≥ -0.1, x[2] ≤ 0.1, x[2] ≥ -0.1, x[3] == 0, x[4] == 0, x[5] == 0], vars)
initial_region_acrobot(x, vars) = HPolyhedron([x[1] ≤ 0.1, x[1] ≥ -0.1, x[2] ≤ 0.1, x[2] ≥ -0.1, x[3] ≤ 0.1, x[3] ≥ -0.1, x[4] ≤ 0.1, x[4] ≥ -0.1, x[5] == 0, x[6] == 0], vars)

function nndm_system(system_name; download_models=false)
    if download_models
        download_nndm_models!()
    end

    if system_name == "pendulum"
        return load_mat_system("pendulum", "alpha", 1, 120, initial_region_pendulum)
    elseif system_name == "cartpole"
        return load_mat_system("cartpole", "alpha", 1, 960, initial_region_cartpole)
    elseif system_name == "husky4d"
        return load_mat_system("husky4d", "alpha", 1, 900, initial_region_husky4d)
    elseif system_name == "husky5d"
        return load_mat_system("husky5d", "alpha", 1, 432, initial_region_husky5d)
    elseif system_name == "acrobot"
        return load_mat_system("acrobot", "alpha", 1, 144, initial_region_acrobot)
    else
        throw("NNDM system not available")
    end
end

function download_nndm_models!()
    # Download entire repo
    tmp_path = download("https://github.com/aria-systems-group/NeuralNetControlBarrier/archive/9a927f3a90c1d79260cf0ddab89e734b5dea5b96.zip")
    
    # Extract models from zip
    zarchive = ZipFile.Reader(tmp_path)
    folder = joinpath("NeuralNetControlBarrier-9a927f3a90c1d79260cf0ddab89e734b5dea5b96", "NeuralNetControlBarrier.jl", "models")
    for f in zarchive.files
        if startswith(f.name, folder)
            output_path = joinpath(splitpath(f.name)[3:end])
            if isdirpath(f.name)
                mkdir(output_path)
            else
                write(output_path, read(f))
            end
        end
    end
    close(zarchive)
end