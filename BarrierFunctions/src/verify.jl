using LinearAlgebra

using Distributions
using Dualization

export verify, barrier_variable!, partition_barrier_variable!, barrier_value, count_variables, overlaps_unsafe
export constraints!, asymmetric_dual_constraint!, unsafe_constraint!, initial_constraint!
export region_constraints!, non_negativity_constraint!

function verify(solver, barrier_type::Type{<:BarrierFunction}, system::AbstractSystem, H::Int, expectation_method::ExpectationMethod)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")

    # solver = dual_optimizer(solver)
    model = Model(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0)
    @variable(model, c, lower_bound = 0)
    B = barrier_variable!(model, barrier_type, system)

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
    @debug "γ $γ, c $c"

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = 1 - (γ + c * H)

    @info "Termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), safety prob $prob"

    return succ, B, prob
end

function barrier_variable!(model, ::Type{<:PiecewiseBarrier}, system::PiecewiseSystem)
    B = PiecewiseBarrier(
           qᵢ => region_barrier_variable!(model, system, qᵢ)
           for qᵢ in regions(system)
       )

    return B
end

function barrier_variable!(model, ::Type{<:PolynomialBarrier}, system::PolynomialSystem)
    vars = variables(system)
    monos = monomials(vars, 0:2)
    # Encode non-negativity with SOSPoly
    @variable(model, B, variable_type=SOSPoly(monos))

    return PolynomialBarrier(B, vars)
end

function barrier_variable!(model, ::Type{<:PolynomialBarrier}, system::PiecewiseSystem)
    @polyvar x[1:dimensionality(system)]
    monos = monomials(x, 0:2)
    # Encode non-negativity with SOSPoly
    @variable(model, B, variable_type=SOSPoly(monos))

    return PolynomialBarrier(B, x)
end

# Optimization variables
function region_barrier_variable!(model, system, qᵢ)
    n = dimensionality(system)

    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, region(qᵢ))
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
barrier_value(B::PolynomialBarrier) = PolynomialBarrier(value(B.p), B.vars)

count_variables(B::PiecewiseBarrier) = sum(Bᵢ -> count_variables(Bᵢ), B)
count_variables(B::BarrierRegion) = count_variables(B.fun)
count_variables(B::LinearBarrier{T, VT}) where {T<:AbstractVariableRef, VT} = length(B.p) + 1
count_variables(B::LinearBarrier{T, VT}) where {T<:Number, VT} = 0
count_variables(B::PolynomialBarrier) = length(monomials(B.p))

# Piece-wise constraints
function constraints!(model, system, B::PiecewiseBarrier, c, γ, expectation_method, solver)
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
    upper_bound_constraint!(model, Bᵢ)
    initial_constraint!(model, system, Bᵢ, γ)
    unsafe_constraint!(model, system, Bᵢ)
    expectation_constraint!(model, system, B, c, qᵢ, expectation)
end

# Piece-wise linear barrier
function non_negativity_constraint!(model, B::BarrierRegion{T, <:LinearBarrier}) where {T}
    H, h = tosimplehrep(region(B))
    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, @expression(model, -p), v, H, h)
end

function upper_bound_constraint!(model, B::BarrierRegion{T, <:LinearBarrier}; M = 5) where {T}
    H, h = tosimplehrep(region(B))
    p, v = B.fun.p, B.fun.v

    M = 5
    asymmetric_dual_constraint!(model, p, @expression(model, M - v), H, h)
end

function initial_constraint!(model, system, B::BarrierRegion{T, <:LinearBarrier}, γ) where {T}
    X₀ = initial_set(system)

    if isdisjoint(X₀, region(B))  # Must be strictly disjoint, as this constraint does not involve dynamics
        return nothing
    end

    H₀, h₀ = tosimplehrep(intersection(X₀, region(B)))
    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, p, @expression(model, γ - v), H₀, h₀)
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

function unsafe_constraint!(model, system, B::BarrierRegion{T, <:LinearBarrier}) where {T}
    if !overlaps_unsafe(system, region(B))
        return nothing
    end

    H, h = tosimplehrep(region(B))
    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, @expression(model, -p), @expression(model, v - 1.0), H, h)
end

function asymmetric_dual_constraint!(model, c, rhs, H, h)
    # Constraints of the form              [max_x  cᵀx, s.t. Hx ≤ h] ≤ rhs
    # reformulated to the asymmetric dual  [min_{y ≥ 0} yᵀh, s.t. Hᵀx = c] ≤ rhs
    # and lifted to the outer minimization.
    # s.t. y ≥ 0
    #      yᵀh ≤ rhs
    #      Hᵀx = c

    m = size(H, 1)
    y = @variable(model, [1:m], lower_bound = 0)

    H, h = tosimplehrep(normalize(HPolyhedron(H, h)))

    @constraints(model, begin
        y' * h ≤ rhs
        H' * y .== c
    end)
end

# Polynomial constraints
function constraints!(model, system, B::PolynomialBarrier, c, γ, expectation_method, solver)
    n_variables = count_variables(B) + 2
    expectation = prepare(system, expectation_method, solver, n_variables)

    initial_constraint!(model, system, B, γ)
    unsafe_constraint!(model, system, B)
    expectation_constraint!(model, system, B, c, expectation)
end

function initial_constraint!(model, system::PolynomialSystem, B::PolynomialBarrier, γ)
    X₀ = initial_set(system)

    @constraint(model, γ - B.p in SOSCone(), domain=X₀)
end

function unsafe_constraint!(model, system::PolynomialSystem, B::PolynomialBarrier)
    Xᵤ = unsafe_set(system)

    for subset in Xᵤ
        @constraint(model, B.p - 1.0 in SOSCone(), domain=subset)
    end
end

function initial_constraint!(model, system::PiecewiseSystem, B::PolynomialBarrier, γ)
    X₀ = initial_set(system)
    x = variables(B)

    H, h = tosimplehrep(X₀)
    X₀_symbolic = BasicSemialgebraicSet(FullSpace(), h .- H * x)  # Hx ≤ h

    @constraint(model, γ - B.p in SOSCone(), domain=X₀_symbolic)
end

function unsafe_constraint!(model, system::PiecewiseSystem, B::PolynomialBarrier)
    Xᵤ = unsafe_set(system)
    x = variables(B)

    for subset in Xᵤ
        H, h = tosimplehrep(subset)
        subset_symbolic = BasicSemialgebraicSet(FullSpace(), h .- H * x)  # Hx ≤ h
        @constraint(model, B.p - 1.0 in SOSCone(), domain=subset_symbolic)
    end
end