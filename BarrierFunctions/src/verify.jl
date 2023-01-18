using LinearAlgebra

using Distributions
using Dualization

export verify, barrier_variable!, partition_barrier_variable!, barrier_value, count_variables, overlaps_unsafe
export constraints!, assymetric_dual_constraint!, unsafe_constraint!, initial_constraint!
export region_constraints!, non_negativity_constraint!

function verify(solver, system::AbstractSystem, H::Int, expectation_method::ExpectationMethod)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")

    solver = dual_optimizer(solver)
    model = Model(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0)
    @variable(model, c, lower_bound = 0)
    B = barrier_variable!(model, system)

    # Constraints
    constraints!(model, system, B, c, γ, expectation_method, solver)

    # Objective
    @objective(model, Min, γ + c * H)

    # Solve
    JuMP.optimize!(model)
    @debug JuMP.solution_summary(model)

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B, c, γ = barrier_value(B), value(c), value(γ)
    @debug "γ $γ, c $c"

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = 1 - (γ + c * H)

    @info "Termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), safety prob $prob"

    return succ, B, prob
end

function barrier_variable!(model, system::PiecewiseSystem)
    B = PiecewiseBarrier(
           qᵢ => region_barrier_variable!(model, system, qᵢ)
           for qᵢ in regions(system)
       )

    return B
end

function barrier_variable!(model, system::PolynomialSystem)
    vars = variables(system)
    monos = monomials(vars, 0:4)
    # Use SOSPoly to enforce non-negativity
    B = @variable(model, variable_type=SOSPoly(monos))

    return PolynomialBarrier(B)
end

# Optimization variables
function region_barrier_variable!(model, system, qᵢ)
    n = dimensionality(system)

    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        p = zeros(n)
        v = 1.0
    else
        Bᵢ = @variable(model, [1:n + 1], base_name = "B[$qᵢ]")
        p, v = Bᵢ[1:end - 1], Bᵢ[end]
    end

    return BarrierRegion(index(qᵢ), region(qᵢ), LinearBarrier(p, v))
end

barrier_value(B::PiecewiseBarrier) = PiecewiseBarrier(index(Bᵢ) => barrier_value(Bᵢ) for Bᵢ in B)
barrier_value(B::BarrierRegion) = BarrierRegion(index(B), region(B), barrier_value(B.fun))
barrier_value(B::LinearBarrier) = LinearBarrier(value.(B.p), value(B.v))
barrier_value(B::ConstantBarrier) = ConstantBarrier(value(B.v))
barrier_value(B::PolynomialBarrier) = PolynomialBarrier(value(B.p))

count_variables(B::PiecewiseBarrier) = sum(Bᵢ -> count_variables(Bᵢ), B)
count_variables(B::BarrierRegion) = count_variables(B.fun)
count_variables(B::LinearBarrier{T, VT}) where {T<:AbstractVariableRef, VT} = length(B.p) + 1
count_variables(B::LinearBarrier{T, VT}) where {T<:Number, VT} = 0
count_variables(B::ConstantBarrier{T}) where {T<:AbstractVariableRef} = 1
count_variables(B::ConstantBarrier{T}) where {T<:Number} = 0
count_variables(B::PolynomialBarrier) = length(monomials(B.p))

# Piece-wise constraints
function constraints!(model, system::PiecewiseSystem, B, c, γ, expectation_method, solver)
    n_variables = count_variables(B) + 2
    expectation = prepare(system, expectation_method, solver, n_variables)

    for qᵢ in regions(system)
        @debug "Region $qᵢ, $(region(qᵢ))"
        region_constraints!(model, system, B, c, γ, qᵢ, expectation)
    end
end

function region_constraints!(model, system, B, c, γ, qᵢ, expectation)
    Bᵢ = B[qᵢ]

    non_negativity_constraint!(model, Bᵢ)
    initial_constraint!(model, system, Bᵢ, γ)
    unsafe_constraint!(model, system, Bᵢ)
    expectation_constraint!(model, system, B, c, qᵢ, expectation)
end

function overlaps_unsafe(system, q)
    Xᵤ = unsafe_set(system)
    # Note the unsafe set can consist of multiple disjoint components
    for Xᵤ_subset in Xᵤ
        if !is_intersection_zeromeasure(Xᵤ_subset, q)
            return true
        end
    end

    return false
end

# Piece-wise linear barrier
function non_negativity_constraint!(model, B::BarrierRegion{T, LinearBarrier{T1, VT1}}) where {T, T1, VT1}
    H, h = tosimplehrep(region(B))
    p, v = B.fun.p, B.fun.v
    assymetric_dual_constraint!(model, -p, v, H, h)

    # Upper bound by M for more robust solving
    M = 5
    assymetric_dual_constraint!(model, p, M - v, H, h)
end

function initial_constraint!(model, system, B::BarrierRegion{T, LinearBarrier{T1, VT1}}, γ) where {T, T1, VT1}
    X₀ = initial_set(system)

    if isdisjoint(X₀, region(B))  # Must be strictly disjoint, as this constraint does not involve dynamics
        return
    end

    H₀, h₀ = tosimplehrep(intersection(X₀, region(B)))
    p, v = B.fun.p, B.fun.v
    assymetric_dual_constraint!(model, p, γ - v, H₀, h₀)
end

function unsafe_constraint!(model, system, B::BarrierRegion{T, LinearBarrier{T1, VT1}}) where {T, T1, VT1}
    if overlaps_unsafe(system, region(B))
        H, h = tosimplehrep(region(B))
        p, v = B.fun.p, B.fun.v
        assymetric_dual_constraint!(model, -p, v - 1.0, H, h)
    end
end

function assymetric_dual_constraint!(model, c, rhs, H, h)
    # Constraints of the form              [max_x  cᵀx, s.t. Hx ≤ h] ≤ rhs
    # reformulated to the asymmetric dual  [min_{y ≥ 0} yᵀh, s.t. Hᵀx = c] ≤ rhs
    # and lifted to the outer minimization.
    # s.t. y ≥ 0
    #      yᵀh ≤ rhs
    #      Hᵀx = c

    m = size(H, 1)
    y = @variable(model, [1:m], lower_bound = 0)

    H, h = tosimplehrep(normalize(HPolyhedron(H, h)))

    @constraint(model, y ⋅ h ≤ rhs)
    @constraint(model, H' * y .== c)
end

# Piece-wise constant barrier
function non_negativity_constraint!(model, B::BarrierRegion{T, ConstantBarrier{T1}}) where {T, T1}
    @constraint(model, B.fun.v ≥ 0.0)
end

function initial_constraint!(model, system, B::BarrierRegion{T, ConstantBarrier{T1}}, γ) where {T, T1}
    X₀ = initial_set(system)

    if is_intersection_zeromeasure(X₀, region(B))
        return
    end

    @constraint(model, γ ≥ B.fun.v)
end

function unsafe_constraint!(model, system, B::BarrierRegion{T, ConstantBarrier{T1}}) where {T, T1}
    if overlaps_unsafe(system, region(B))
        @constraint(model, B.fun.v ≥ 1.0)
    end
end

# Piece-wise quadratic barrier
function non_negativity_constraint!(model, B::BarrierRegion{T, QuadraticBarrier{T1, VT1, VVT1}}) where {T, T1, VT1, VVT1}
    q = region(B)
    s_mat = s_procedure(model, q)

    @constraint(model, -matrix(B) - s_mat ≤ 0, PSDCone())
end

function initial_constraint!(model, system, B::BarrierRegion{T, QuadraticBarrier{T1, VT1, VVT1}}, γ) where {T, T1, VT1, VVT1}
    X₀ = initial_set(system)
    q = region(B)

    if is_intersection_zeromeasure(X₀, q)
        return
    end

    s_mat = s_procedure(model, intersection(X₀, q))

    γ_mat = zero(s_mat)
    γ_mat[end, end] = γ

    @constraint(model, matrix(B) - γ_mat - s_mat ≤ 0, PSDCone())
end

function unsafe_constraint!(model, system, B::BarrierRegion{T, QuadraticBarrier{T1, VT1, VVT1}}) where {T, T1, VT1, VVT1}
    q = region(B)
    if overlaps_unsafe(system, q)
        # We may need to refine partitions if a partition cover too large part of the safe set

        s_mat = s_procedure(model, q)

        one_mat = zero(s_mat)
        one_mat[end, end] = 1

        @constraint(model, -matrix(B) + one_mat - s_mat ≤ 0, PSDCone())
    end
end

function s_procedure(model, p)
    H, h = tosimplehrep(p)
    m, n = size(H)
    τ = @variable(model, [1:m], lower_bound = 0)

    Hτ = H' * τ
    hτ = -2 * h' * τ

    mat = vcat(
        hcat(zeros(n, n), Hτ),
        hcat(Hτ', hτ)
    )

    return mat
end

# Polynomial constraints
function constraints!(model, system::PolynomialSystem, B, c, γ, expectation_method, solver)
    n_variables = count_variables(B) + 2
    expectation = prepare(system, expectation_method, solver, n_variables)

    initial_constraint!(model, system, B, γ)
    unsafe_constraint!(model, system, B)
    expectation_constraint!(model, system, B, c, expectation)
end

function initial_constraint!(model, system, B::PolynomialBarrier, γ)
    X₀ = initial_set(system)

    @constraint(model, γ - B.p in SOSCone(), domain=X₀)
end

function unsafe_constraint!(model, system, B::PolynomialBarrier)
    Xᵤ = unsafe_set(system)

    for subset in Xᵤ
        @constraint(model, B.p - 1.0 in SOSCone(), domain=subset)
    end
end