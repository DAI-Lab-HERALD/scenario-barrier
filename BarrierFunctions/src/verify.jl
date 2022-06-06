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
           p.i => @variable(model, [1:n, 1:n], Symmetric, base_name = "B[$p]")
           for p in partitions(system)
       )

    return B
end

function constraints!(model, system, B, β, γ, σ, 𝔼)
    for p in partitions(system)
        println("Partition $p")
        partition_constraints!(model, system, B, β, γ, σ, 𝔼, p)
    end
end

function partition_constraints!(model, system, B, β, γ, σ, 𝔼, p)
    non_negativity_constraint(model, B, p)
    unsafe_constraint!(model, system, B, p)
    initial_constraint!(model, system, B, γ, p)
    expectation_constraint!(model, system, B, β, σ, 𝔼, p)
end

s_procedure(p::IndexedConvexPolytope) = s_procedure(p.p)
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

function non_negativity_constraint(model, B, p)
    s_mat = s_procedure(p)

    @constraint(model, B[p.i] + s_mat >= 0, PSDCone())
end

function unsafe_constraint!(model, system, B, p)
    # Note unsafe set can consist of multiple disjoint components
    for unsafe_subset in unsafe_set(system)
        println("Unsafe subset ", unsafe_subset)
        # TODO ignore constraint if (ℝ^n \ S) ∩ p = ∅
    end

    # We may need to refine partitions if a partition cover too large part of the safe set

    s_mat = s_procedure(p)

    one_mat = zero(s_mat)
    one_mat[end, end] = 1

    @constraint(model, B - one_mat + s_mat >= 0, PSDCone())
end

function initial_constraint!(model, system, B, γ, p)
    X_0 = initial_set(system)

    if isempty(X_0 ∩ p)
        return

    s_mat = s_procedure(X_0 ∩ p)

    γ_mat = zero(s_mat)
    γ_mat[end, end] = γ

    @constraint(model, -B[p] + γ_mat + s_mat ≥ 0, PSDCone())
end

function expectation_constraint!(model, system, B, β, σ, 𝔼, p_i)
    for p_j in partitions(system)
        println("Partition $p")
        partition_constraint!(model, system, B, β, σ, 𝔼, p_i, p_j)
    end
end

function partition_constraint!(model, system, B, β, σ, 𝔼, p_i, p_j)
    𝔼Bfx = expectation(𝔼, system, B[p_j], σ, p)

    @constraint(model, 𝔼Bfx <= B[p_i] + β, domain = p_i, basis = ScaledMonomialBasis)
    # TODO: Replace with LMI
    # TODO: Include d
end

# Patch for ScaledMonomialBasis to match regular MonomialBasis
MP.polynomialtype(::Type{<:ScaledMonomialBasis{MT}}, T::Type) where MT = MP.polynomialtype(MT, promote_type(T, Float64))
