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
        σ = 0.1
        μ = 1.05263157895
        η₃ = Normal(μ, σ)

        η₁₂ = Dirac(0.0)

        η = product_distribution(η₁₂, η₁₂, η₃)

        X0 = Singleton([-0.95, 0.0, 0.0])

        # Safe
        Xs = Hyperrectangle(low=[-1.9, -1.9, -π / 2], high=[1.9, 1.9, π / 2])

        # Unsafe
        remaining_region = convert(HPolyhedron, CartesianProduct(Universe(2), Interval(-π / 2, π / 2)))

        eachfirstdim(A) = eachslice(A, dims=(1,))

        Xu = HPolyhedron{Float64, Vector{Float64}}[]
        for i in [1, 2]
            dim_bounds = state_space[i, :]
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
prob_dubins_car_1000regions = () -> Problem("dubins_car_1000regions", load_nc_system("models/dubins_car/dynamics_1000regions.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))
