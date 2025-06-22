using BarrierFunctions
import Spatial: has_mbr
using LazySets, Symbolics
using NCDatasets

using Distributions, PDMats

function load_lk_nc_system(path; eps=1e-12)
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
        stddev = [0.01, 0.01, 0.001]
        μ = zero(stddev)
        Σ = PDiagMat(stddev .^ 2)  # Square to make std.dev. to variance.
        η = DiagNormal(μ, Σ)

        X0 = Hyperrectangle(;low=[1.0, -0.5, -0.005], high=[2.0, 0.05, 0.005])

        # Safe
        Xs = Hyperrectangle(;low=[1.0, -6.0, -0.05], high=[10.0, 6.0, 0.05])

        # Unsafe
        remaining_region = Universe(n)

        eachfirstdim(A) = eachslice(A, dims=(1,))

        Xu = [
            Hyperrectangle(;low=[1.0, -7.0, -0.05], high=[10.0, -6.0, -0.05])
            Hyperrectangle(;low=[1.0, 6.0, -0.05], high=[10.0, 7.0, -0.05])
        ]

        return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
    end

    return system
end

H = 3
prob_lane_keeping_2520regions_181410 = () -> Problem("lane_keeping_2520regions_181410", load_lk_nc_system("models/lane_keeping/dynamics_2520.nc"), H, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/H, 1e-9))

probs_lane_keeping = [
    prob_lane_keeping_2520regions_181410
]