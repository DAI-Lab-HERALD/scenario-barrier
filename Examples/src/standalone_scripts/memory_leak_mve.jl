using Revise

using MosekTools, Mosek, GLPK, JuMP

using BarrierFunctions
import Spatial: has_mbr
using LazySets, Symbolics
using MAT, ZipFile

using Distributions, PDMats

load_mat_system(system_name, bound_method, num_layers, num_regions, initial_region_cb, stddev; eps=1e-12) = load_mat_system("models/$system_name/$bound_method/$(num_layers)_layers/partition_data_$num_regions.mat", initial_region_cb, stddev; eps=eps)

function load_mat_system(path, initial_region_cb, stddev; eps=1e-12)
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
        push!(Q, convert(HPolyhedron, Hyperrectangle(low=region[1, :], high=region[2, :])))
    end

    # Dynamics
    dyn = []
    for (Mₗᵢ, Bₗᵢ, Mₕᵢ, Bₕᵢ) in zip(eachfirstdim(Mₗ), eachfirstdim(Bₗ), eachfirstdim(Mₕ), eachfirstdim(Bₕ))
        Mₗᵢ = transpose(Mₗᵢ)  # Why is the transpose necessary?
        Mₕᵢ = transpose(Mₕᵢ)

        dyn_i = [AffineDynamics(copy(Mₗᵢ), copy(Bₗᵢ[:, 1]) .- eps), AffineDynamics(copy(Mₕᵢ), copy(Bₕᵢ[:, 1]))]
        push!(dyn, dyn_i)
    end

    μ = zero(stddev)
    Σ = PDiagMat(stddev .^ 2)  # Square to make std.dev. to variance.
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

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

initial_region_pendulum(x, vars) = HPolyhedron([-deg2rad(5) ≤ x[1], x[1] ≤ deg2rad(5), x[2] == 0], vars)

function nndm_system(system_name, num_layers, num_regions; eps=1e-12)
    # Noise of 1/100th of the width of the safe set across each dimension is applied.

    if system_name == "pendulum"
        # θ × θ̇
        noise = [deg2rad(12) / 100, 1 / 100]
        # noise = [0.01, 0.01]
        return load_mat_system("pendulum", "alpha", num_layers, num_regions, initial_region_pendulum, noise; eps=eps)
    else
        throw("NNDM system not available")
    end
end

function iterate_over_problem(H=5)
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
    # solver = JuMP.optimizer_with_attributes(() -> GLPK.Optimizer(method=GLPK.SIMPLEX))
    system, expectation_method = nndm_system("pendulum", 1, 480), ScenarioApproach(0.005/H, 1e-9)

    for i in 1:1000
        @info "Iteration $i/1000"

        # out = read(`top -bn1 -p $(getpid())`, String)
        # println("before $(split(split(out,  "\n")[end-1])[6])")

        problem(solver, system, H, expectation_method)
        # problem(solver)

        GC.gc(true)

        # out = read(`top -bn1 -p $(getpid())`, String)
        # println("after $(split(split(out,  "\n")[end-1])[6])")
    end
end

# function problem(solver; n = 2, m = 5)
#     model = Model(solver; add_bridges = false)
#     set_string_names_on_creation(model, false)

#     @variable(model, x[1:n])
#     @variable(model, x₀, upper_bound=1.0)

#     @objective(model, Max, 1.0 * x₀)

#     A = rand(m, n)
#     b = rand(m)

#     # Intuitively, if x₀ > 0 then the intersection is full-dimensional.
#     @constraint(model, A * x .+ x₀ .≤ b)

#     optimize!(model)

#     iszeromeasure = objective_value(model) ≤ 0

#     return iszeromeasure
# end

function problem(solver, system, H, expectation_method)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")

    # solver = dual_optimizer(solver)
    model = Model(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0)
    @variable(model, c, lower_bound = 0)
    B = barrier_variable!(model, system)

    # Constraints
    constraints!(model, system, B, c, γ, expectation_method, solver)

    # Objective
    @objective(model, Min, γ + c * H)

    @info "Done constructing the optimization problem"

    # Solve
    JuMP.optimize!(model)
    @debug JuMP.solution_summary(model)

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B, c, γ = barrier_value(B), value(c), value(γ)
    prob = 1 - (γ + c * H)

    @info "Termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), safety prob $prob"

    return succ, prob
end