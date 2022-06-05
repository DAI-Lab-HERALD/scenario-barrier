using JuMP, MosekTools, SumOfSquares, Hypatia
using SemialgebraicSets, DynamicPolynomials, MultivariatePolynomials

function verify(system::System, σ::Number, H::Int; 𝔼::Expectation = AnalyticExpectation())
    @assert(σ >= 0, "Standard deviation must be greater or equal to zero (is $σ)")
    @assert(H >= 1, "Horizon N must be greater than or equal to one (is $H)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false
        )
    model = SOSModel(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0, upper_bound = 1)
    @variable(model, β, lower_bound = 0, upper_bound = 1)

    monos = monomials(state(system), 0:2)
    @variable(model, B[partitions(system)], Poly(ScaledMonomialBasis(monos)))

    # Constraints
    partition_constraints!(model, system, B, β, γ, σ, 𝔼)

    # Objective
    @objective(model, Min, γ + β * H)

    # Solve
    JuMP.optimize!(model)
    # println(JuMP.solution_summary(model))

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B = value(B)
    β = value(β)
    γ = value(γ)

    println("γ $γ, beta $β")

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = γ + β * H

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), prob $prob")

    return succ, B, prob
end

function partition_constraints!(model, system, B, β, γ, σ, 𝔼)
    for p in partitions(system)
        println("Partition $p")

        non_negativity_constraint(model, B, p)
        unsafe_constraint!(model, system, B, p)
        initial_constraint!(model, system, B, γ, p)

        expectation_constraint!(model, system, B, β, σ, 𝔼, p)
    end
end

function non_negativity_constraint(model, B, p) 
    @constraint(model, B[p] >= 0, basis = ScaledMonomialBasis, domain = p)
    # TODO: Replace with LMI
end

function unsafe_constraint!(model, system, B, p)
    # Note unsafe set can consist of multiple disjoint components
    for unsafe_subset in unsafe_set(system)
        println("Unsafe subset ", unsafe_subset)
        # TODO ignore constraint if (ℝ^n \ S) ∩ p = ∅
    end

    # We may need to refine partitions if a partition cover too large part of the safe set

    @constraint(model, B >= 1, domain = p, basis = ScaledMonomialBasis)
    # TODO: Replace with LMI
end

function initial_constraint!(model, system, B, γ, p)
    X_0 = initial_set(system)
    # TODO ignore constraint if X_0 ∩ p = ∅

    @constraint(model, B[p] <= γ, domain = X_0 ∩ p, basis = ScaledMonomialBasis)
    # TODO: Replace with LMI
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
