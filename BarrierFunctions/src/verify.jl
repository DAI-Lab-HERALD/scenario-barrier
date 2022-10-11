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
           qᵢ => partition_barrier_variable!(model, system, qᵢ)
           for qᵢ in partitions(system)
       )

    return B
end

function partition_barrier_variable!(model, system, qᵢ)
    n = state_space_size(system) + 1

    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        unsafe_barrier = zeros(n)
        unsafe_barrier[end] = 1

        return unsafe_barrier
    else
        return @variable(model, [1:n], base_name = "B[$qᵢ]")
    end
end

barrier_blocks(B, qᵢ) = B[qᵢ][1:end - 1], B[qᵢ][end]

function barrier_value(B)
    B = Dict(i => value.(Bᵢ) for (i, Bᵢ) in B)

    return B
end

function constraints!(model, system, B, c, γ, ϵ, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)



    for qᵢ in partitions(system)
        println("Partition $qᵢ")
        partition_constraints!(model, system, B, c, γ, ϵ, qᵢ, ηs)
    end
end



function partition_constraints!(model, system, B, c, γ, ϵ, qᵢ, ηs)
    non_negativity_constraint(model, B, qᵢ)
    initial_constraint!(model, system, B, γ, qᵢ)
    unsafe_constraint!(model, system, B, qᵢ)
    expectation_constraint!(model, system, B, c, ϵ, qᵢ, ηs)
end

function non_negativity_constraint(model, B, qᵢ)
    Hᵢ, hᵢ = tosimplehrep(qᵢ)
    m = size(Hᵢ, 1)

    pᵢ, vᵢ = barrier_blocks(B, qᵢ)
    yᵢ = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yᵢ ⋅ hᵢ ≤ vᵢ)
    @constraint(model, Hᵢ' * yᵢ .== -pᵢ)
end

function initial_constraint!(model, system, B, γ, qᵢ)
    X₀ = initial_set(system)

    if is_intersection_zeromeasure(X₀, qᵢ)
        return
    end

    H₀, h₀ = tosimplehrep(intersection(X₀, qᵢ))
    m = size(H₀, 1)

    pᵢ, vᵢ = barrier_blocks(B, qᵢ)
    yᵢ = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yᵢ ⋅ h₀ ≤ γ - vᵢ)
    @constraint(model, H₀' * yᵢ .== pᵢ)
end

function unsafe_constraint!(model, system, B, qᵢ)
    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        return
    end

    if !overlaps_unsafe(system, qᵢ)
        return
    end

    Hᵢ, hᵢ = tosimplehrep(qᵢ)
    m = size(Hᵢ, 1)

    pᵢ, vᵢ = barrier_blocks(B, qᵢ)
    yᵢ = @variable(model, [1:m], lower_bound = 0)

    @constraint(model, yᵢ ⋅ hᵢ ≤ vᵢ - 1)
    @constraint(model, Hᵢ' * yᵢ .== -pᵢ)
end

function overlaps_unsafe(system, qᵢ)
    Xᵤ = unsafe_set(system)
    # Note the unsafe set can consist of multiple disjoint components
    for Xᵤ_subset in Xᵤ
        if !is_intersection_zeromeasure(Xᵤ_subset, qᵢ)
            return true
        end
    end

    return false
end

function expectation_constraint!(model, system, B, c, ϵ, qᵢ, ηs)
    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        return
    end

    for qⱼ in partitions(system)
        # println("Next partition $qj")
        expectation_partition_constraint!(model, B, c, ϵ, qᵢ, qⱼ, ηs)
    end
end

function expectation_partition_constraint!(model, B, c, ϵ, qᵢ, qⱼ, ηs)
    Aᵢ, bᵢ = dynamics(qᵢ)

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    M = 1  
    d = ϵ * M

    Hᵢ, hᵢ = tosimplehrep(qᵢ)
    m = size(Hᵢ, 1)

    qᵢ_next = Aᵢ * region(qᵢ) + bᵢ

    pᵢ, vᵢ = barrier_blocks(B, qᵢ)
    pⱼ, vⱼ = barrier_blocks(B, qⱼ)

    for η in eachcol(ηs)
        # println("$qi -> $(qi_next + η), cap $qj")
        if is_intersection_zeromeasure(qⱼ, qᵢ_next + η)
            continue
        end

        yᵢ = @variable(model, [1:m], lower_bound = 0)

        @constraint(model, yᵢ ⋅ hᵢ ≤ vᵢ + c - d - pⱼ ⋅ (bᵢ + η) - vⱼ)
        @constraint(model, Hᵢ' * yᵢ .== Aᵢ' * pⱼ .- pᵢ)
    end
end
