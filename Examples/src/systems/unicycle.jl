using BarrierFunctions
import Spatial: has_mbr
using LazySets, Symbolics
using NCDatasets

using Distributions, PDMats

function load_nc_system(path; eps=1e-12)
    system = NCDataset(path, "r") do ds
        regions = ds[:regions][:, :, :]
        state_space = ds[:safe_set][:, :]

        # Read Bounds
        A = convert.(Float64, ds[:nominal_dynamics_A][:, :, :, :])
        b = ds[:nominal_dynamics_b][:, :, :]

        n = size(state_space, 1)
        vars = Symbolics.@variables x[1:n]

        eachlastdim(A) = eachslice(A, dims=(ndims(A),))

        # Regions
        Q = []
        for region in eachlastdim(regions)
            push!(Q, Hyperrectangle(low=region[:, 1], high=region[:, 2]))
        end

        # Dynamics
        dyn = []
        for (Aᵢ, bᵢ) in zip(eachlastdim(A), eachlastdim(b))

            dyn_i = [
                AffineDynamics(transpose(copy(Aᵢ[:, :, 1])), copy(bᵢ[:, 1]) .- eps),
                AffineDynamics(transpose(copy(Aᵢ[:, :, 2])), copy(bᵢ[:, 2]) .+ eps)
            ]
            push!(dyn, dyn_i)
        end

        # Noise
        stddev = [0.1, 0.05, 0.01, 0.05]
        μ = zero(stddev)
        Σ = PDiagMat(stddev .^ 2)  # Square to make std.dev. to variance.
        η = MvNormal(μ, Σ)

        X0 = Hyperrectangle([-0.20, -0.20, 0.005, 0.15], [0.05, 0.05, 0.005, 0.05])

        # Safe
        Xs = HPolyhedron([[state_space[i, 1] ≤ x[i] for i in 1:n]; [x[i] ≤ state_space[i, 2] for i in 1:n]], vars)

        # Unsafe
        remaining_region = Universe(n)

        eachfirstdim(A) = eachslice(A, dims=(1,))

        Xu = HPolyhedron{Float64, Vector{Float64}}[]
        for (i, dim_bounds) in enumerate(eachfirstdim(state_space))
            unsafe_lower = HPolyhedron([x[i] ≤ dim_bounds[1]], vars)
            unsafe_upper = HPolyhedron([x[i] ≥ dim_bounds[2]], vars)


            unsafe_lower = intersection(unsafe_lower, remaining_region)
            unsafe_upper = intersection(unsafe_upper, remaining_region)
            push!(Xu, unsafe_lower, unsafe_upper)
            push!(Q, unsafe_lower, unsafe_upper)

            # Just zero dynamics - not important as this region is unsafe and thus never propagated
            dyn_i = [AffineDynamics(zero(A[:, :, 1, 1]), zero(b[:, 1, 1]))]
            push!(dyn, dyn_i, dyn_i)

            remaining_region = intersection(remaining_region, HPolyhedron([dim_bounds[1] ≤ x[i], x[i] ≤ dim_bounds[2]], vars))
        end

        return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
    end

    return system
end

H = 10
prob_unicycle_1250regions_5555 = () -> Problem("unicycle_1250regions_5555", load_nc_system("models/unicycle/dynamics_1250regions_5555.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))
prob_unicycle_1800regions_6655 = () -> Problem("unicycle_1800regions_6655", load_nc_system("models/unicycle/dynamics_1800regions_6655.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))
prob_unicycle_1800regions_5566 = () -> Problem("unicycle_1800regions_5566", load_nc_system("models/unicycle/dynamics_1800regions_5566.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))
prob_unicycle_2400regions_5568 = () -> Problem("unicycle_2400regions_5568", load_nc_system("models/unicycle/dynamics_2400regions_5568.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))
prob_unicycle_2400regions_5586 = () -> Problem("unicycle_2400regions_5586", load_nc_system("models/unicycle/dynamics_2400regions_5586.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))

probs_unicycle = [
    prob_unicycle_1250regions_5555,
    prob_unicycle_1800regions_6655,
    prob_unicycle_1800regions_5566,
    prob_unicycle_2400regions_5568,
    prob_unicycle_2400regions_5586
]