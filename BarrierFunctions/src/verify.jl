using JuMP, MosekTools

function verify(system::System, σ::Number, H::Int; 𝔼::Expectation = AnalyticExpectation())
    @assert(σ >= 0, "Standard deviation must be greater or equal to zero (is $σ)")
    @assert(H >= 1, "Horizon N must be greater than or equal to one (is $H)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false
        )
    model = Model(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0, upper_bound = 1)
    @variable(model, β, lower_bound = 0, upper_bound = 1)
    B = barrier_variable!(model, system)

    # Constraints
    constraints!(model, system, B, β, γ, σ, 𝔼)

    # Objective
    @objective(model, Min, γ + β * H)

    # Solve
    JuMP.optimize!(model)
    # println(JuMP.solution_summary(model))

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B, β, γ = value(B), value(β), value(γ)
    println("γ $γ, beta $β")

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = γ + β * H

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), prob $prob")

    return succ, B, prob
end

function barrier_variable!(model, system)
    n = size(state(system))

    B = Dict{Int, Array{VariableRef, 2}}(
           q.index => @variable(model, [1:n, 1:n], Symmetric, base_name = "B[$q]")
           for q in partitions(system)
       )

    return B
end

function constraints!(model, system, B, β, γ)
    for q in partitions(system)
        println("Partition $q")
        partition_constraints!(model, system, B, β, γ, q)
    end
end

function partition_constraints!(model, system, B, β, γ, q)
    non_negativity_constraint(model, B, q)
    unsafe_constraint!(model, system, B, q)
    initial_constraint!(model, system, B, γ, q)
    expectation_constraint!(model, system, B, β, q)
end

s_procedure(q::IndexedConvexPolytope) = s_procedure(q.polytope)
function s_procedure(p)
    m, n = size(p)
    τ = @variable(model, [1:m], lower_bound = 0)

    Hτ = p.H' * τ
    hτ = -2 * p.h' * τ

    mat = vcat(
        hcat(zeros(n), Hτ),
        hcat(Hτ', hτ)
    )

    return mat
end

function non_negativity_constraint(model, B, q)
    s_mat = s_procedure(q)

    @constraint(model, B[p.i] + s_mat >= 0, PSDCone())
end

function unsafe_constraint!(model, system, B, q)
    # Note unsafe set can consist of multiple disjoint components
    for unsafe_subset in unsafe_set(system)
        println("Unsafe subset ", unsafe_subset)
        # TODO ignore constraint if (ℝ^n \ S) ∩ q = ∅
    end

    # We may need to refine partitions if a partition cover too large part of the safe set

    s_mat = s_procedure(q)

    one_mat = zero(s_mat)
    one_mat[end, end] = 1

    @constraint(model, B - one_mat + s_mat >= 0, PSDCone())
end

function initial_constraint!(model, system, B, γ, q)
    X_0 = initial_set(system)

    if isempty(X_0 ∩ q)
        return

    s_mat = s_procedure(X_0 ∩ q)

    γ_mat = zero(s_mat)
    γ_mat[end, end] = γ

    @constraint(model, -B[p] + γ_mat + s_mat ≥ 0, PSDCone())
end

function expectation_constraint!(model, system, B, β, q_i)
    for q_j in partitions(system)
        println("Next partition $q_j")
        expectation_partition_constraint!(model, system, B, β, q_i, q_j)
    end
end

barrier_blocks(B, q) = B[q.index][:end - 1, :end - 1], B[q.index][:end - 1, end], B[q.index][end, end]
function expectation_partition_constraint!(model, system, B, β, q_i, q_j)
    P_i, p_i, v_i = barrier_blocks(B, q_i)
    P_j, p_j, v_j = barrier_blocks(B, q_j)

    # TODO: Replace with LMI
    # TODO: Include d
end
