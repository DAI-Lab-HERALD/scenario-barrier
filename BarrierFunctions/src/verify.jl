using JuMP, MosekTools, Dualization
using LinearAlgebra

using Distributions

function verify(system::System; H::Int, ϵ::Number, num_samples::Int)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")
    @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
    @assert(num_samples >= 1, "Number of samples must be greater than or equal to one (is $num_samples)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false
    )
    # solver = dual_optimizer(solver)
    model = Model(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0)
    @variable(model, c, lower_bound = 0)
    B = barrier_variable!(model, system)

    # Constraints
    constraints!(model, system, B, c, γ, ϵ, num_samples)

    # Objective
    @objective(model, Min, γ + c * H)

    # Solve
    JuMP.optimize!(model)
    println(JuMP.solution_summary(model))

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B, c, γ = barrier_value(B), value(c), value(γ)
    println("γ $γ, c $c")

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = 1 - (γ + c * H)

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), safety prob $prob")

    return succ, B, prob
end

function barrier_variable!(model, system)
    B = Dict(
           q => partition_barrier_variable!(model, system, q)
           for q in partitions(system)
       )

    return B
end

function partition_barrier_variable!(model, system, q)
    n = state_space_size(system) + 1

    Xs = safe_set(system)
    if (iszeromeasure ∘ intersection)(Xs, q)
        unsafe_barrier = zeros(n)
        unsafe_barrier[end] = 1

        return unsafe_barrier
    else
        return @variable(model, [1:n], base_name = "B[$q]")
    end
end

function barrier_value(B)
    B = Dict(i => value.(Bi) for (i, Bi) in B)

    return B
end

function constraints!(model, system, B, c, γ, ϵ, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)

    for q in partitions(system)
        println("Partition $q")
        partition_constraints!(model, system, B, c, γ, ϵ, q, ηs)
    end
end

function partition_constraints!(model, system, B, c, γ, ϵ, q, ηs)
    non_negativity_constraint(model, B, q)
    initial_constraint!(model, system, B, γ, q)
    unsafe_constraint!(model, system, B, q)
    expectation_constraint!(model, system, B, c, ϵ, q, ηs)
end

tosimplehrep(q::Partition) = tosimplehrep(region(q))

function non_negativity_constraint(model, B, q)
    Hi, hi = tosimplehrep(q)
    m = size(Hi, 1)

    yi = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yi ⋅ hi ≤ B[q][end])
    @constraint(model, Hi' * yi .== -B[q][1:end - 1])
end

function initial_constraint!(model, system, B, γ, q)
    X0 = initial_set(system)

    if (iszeromeasure ∘ intersection)(X0, q)
        return
    end

    H0, h0 = tosimplehrep(intersection(X0, q))
    m = size(H0, 1)

    yi = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yi ⋅ h0 ≤ γ - B[q][end])
    @constraint(model, H0' * yi .== B[q][1:end - 1])
end

function unsafe_constraint!(model, system, B, q)
    Xs = safe_set(system)
    if (iszeromeasure ∘ intersection)(Xs, q)
        return
    end

    # Note unsafe set can consist of multiple disjoint components
    Xu = unsafe_set(system)
    overlaps_unsafe = false
    for Xu_subset in Xu
        if !(iszeromeasure ∘ intersection)(Xu_subset, q)
            overlaps_unsafe = true
            break
        end
    end

    if !overlaps_unsafe
        return
    end

    # We may need to refine partitions if a partition cover too large part of the safe set
    Hi, hi = tosimplehrep(q)
    m = size(Hi, 1)

    yi = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yi ⋅ hi ≤ B[q][end] - 1)
    @constraint(model, Hi' * yi .== -B[q][1:end - 1])
end

function expectation_constraint!(model, system, B, c, ϵ, qi, ηs)
    Xs = safe_set(system)
    if (iszeromeasure ∘ intersection)(Xs, qi)
        return
    end

    for qj in partitions(system)
        println("Next partition $qj")
        expectation_partition_constraint!(model, B, c, ϵ, qi, qj, ηs)
    end
end

function expectation_partition_constraint!(model, B, c, ϵ, qi, qj, ηs)
    pi, vi = B[qi][1:end - 1], B[qi][end]
    pj, vj = B[qj][1:end - 1], B[qj][end]

    Ai, bi = dynamics(qi)

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    M = 1  
    d = ϵ * M

    Hi, hi = tosimplehrep(qi)
    m = size(Hi, 1)

    yi = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yi ⋅ hi ≤ vi + c - pj ⋅ bi - vj)
    @constraint(model, Hi' * yi .== Ai' * pj .- pi)

    # for η in eachcol(ηs)

    #     biη = bi + η

    #     AiPjbiη = Ai' * Pj * biη

    #     η_mat = vcat(
    #         hcat(zero(Ai), AiPjbiη),
    #         hcat(AiPjbiη', biη' * Pj * biη .+ pj' * biη)
    #     )
    #     # - s_mat
    #     @constraint(model, shared_mat + η_mat ≤ 0, PSDCone())
    # end
end
