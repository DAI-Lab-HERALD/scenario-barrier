using LinearAlgebra, SparseArrays

using Distributions
using Dualization

export verify, barrier_variable!, partition_barrier_variable!, barrier_value, count_variables, overlaps_unsafe
export constraints!, asymmetric_dual_constraint!, unsafe_constraint!, initial_constraint!
export region_constraints!, non_negativity_constraint!

function verify(solver, barrier_type::BarrierTemplate, system::AbstractSystem, H::Int, expectation_method::ExpectationMethod)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")

    # solver = dual_optimizer(solver)
    model = Model(solver)
    set_string_names_on_creation(model, false)

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

function barrier_variable!(model, template::PiecewiseTemplate, system::PiecewiseSystem)
    B = PiecewiseBarrier(Dict(
            index(qᵢ) => region_barrier_variable!(model, template.underlying_template, system, qᵢ) 
            for qᵢ in regions(system)
        ))

    return B
end

function barrier_variable!(model, template::PolynomialTemplate, system::PolynomialSystem)
    vars = variables(system)
    monos = monomials(vars, 0:div(template.degree, 2))
    # Encode non-negativity with SOSPoly
    @variable(model, B, variable_type=SOSPoly(monos))

    return PolynomialBarrier(B, vars)
end

function barrier_variable!(model, template::PolynomialTemplate, system::PiecewiseSystem)
    @polyvar x[1:dimensionality(system)]
    monos = monomials(x, 0:div(template.degree, 2))
    # Encode non-negativity with SOSPoly
    @variable(model, B, variable_type=SOSPoly(monos))

    return PolynomialBarrier(B, x)
end

# Optimization variables
function region_barrier_variable!(model, ::LinearTemplate, system, qᵢ)
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

function region_barrier_variable!(model, ::ConstantTemplate, system, qᵢ)
    if overlaps_unsafe(system, region(qᵢ))
        v = 1.0
    else
        v = @variable(model, base_name = "B[$qᵢ]")
    end

    return BarrierRegion(index(qᵢ), region(qᵢ), ConstantBarrier(v))
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

barrier_value(B::PiecewiseBarrier) = PiecewiseBarrier(Dict(index(Bᵢ) => barrier_value(Bᵢ) for Bᵢ in B))
barrier_value(B::BarrierRegion) = BarrierRegion(index(B), region(B), barrier_value(B.fun))
barrier_value(B::LinearBarrier) = LinearBarrier(value.(B.p), value(B.v))
barrier_value(B::ConstantBarrier) = ConstantBarrier(value(B.v))
barrier_value(B::PolynomialBarrier) = PolynomialBarrier(value(B.p), B.vars)

count_variables(B::PiecewiseBarrier) = sum(Bᵢ -> count_variables(Bᵢ), B)
count_variables(B::BarrierRegion) = count_variables(B.fun)
count_variables(B::LinearBarrier{T}) where {T<:AbstractVariableRef} = length(B.p) + 1
count_variables(B::LinearBarrier{T}) where {T<:Number} = 0
count_variables(B::ConstantBarrier{T}) where {T<:AbstractVariableRef} = 1
count_variables(B::ConstantBarrier{T}) where {T<:Number} = 0
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

region_constraints!(model, system, B, c, γ, qᵢ, expectation) = region_constraints!(model, system, B, c, γ, B[index(qᵢ)], qᵢ, expectation)
function region_constraints!(model, system, B, c, γ, Bᵢ::BarrierRegion{<:LinearBarrier{<:VariableRef}}, qᵢ, expectation)
    non_negativity_constraint!(model, Bᵢ)
    upper_bound_constraint!(model, Bᵢ)
    initial_constraint!(model, system, Bᵢ, γ)
    unsafe_constraint!(model, system, Bᵢ)
    expectation_constraint!(model, system, B, c, qᵢ, expectation)
end

function region_constraints!(model, system, B, c, γ, Bᵢ::BarrierRegion{<:LinearBarrier{<:Number}}, qᵢ, expectation) end

# Piece-wise linear barrier
function non_negativity_constraint!(model, B::BarrierRegion{<:LinearBarrier})
    H, h = tosimplehrep(region(B))
    H = sparse(H)
    
    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, @expression(model, -p), v, H, h)
end

function upper_bound_constraint!(model, B::BarrierRegion{<:LinearBarrier}; M = 5)
    H, h = tosimplehrep(region(B))
    H = sparse(H)

    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, p, @expression(model, M - v), H, h)
end

function initial_constraint!(model, system, B::BarrierRegion{<:LinearBarrier}, γ)
    X₀ = initial_set(system)

    if isdisjoint(X₀, region(B))  # Must be strictly disjoint, as this constraint does not involve dynamics
        return nothing
    end

    H₀, h₀ = tosimplehrep(intersection(X₀, region(B)))
    H₀ = sparse(H₀)

    p, v = B.fun.p, B.fun.v
    asymmetric_dual_constraint!(model, p, @expression(model, γ - v), H₀, h₀)
end

function unsafe_constraint!(model, system, B::BarrierRegion{<:LinearBarrier})
    Xᵤ = unsafe_set(system)
    p, v = B.fun.p, B.fun.v
    q = region(B)

    # Note the unsafe set can consist of multiple disjoint components
    for Xᵤ_subset in Xᵤ
        if !is_intersection_zeromeasure(Xᵤ_subset, q)
            H, h = tosimplehrep(intersection(region(B), Xᵤ_subset))
            asymmetric_dual_constraint!(model, @expression(model, -p), @expression(model, v - 1.0), H, h)
        end
    end
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

    @constraints(model, begin
        y' * h ≤ rhs
        H' * y .== c
    end)
end

# Piece-wise constant barrier
function non_negativity_constraint!(model, B::BarrierRegion{<:ConstantBarrier{<:AbstractVariableRef}})
    v = B.fun.v
    set_lower_bound(v, 0)
end

function non_negativity_constraint!(model, B::BarrierRegion{<:ConstantBarrier{<:Number}}) end

function upper_bound_constraint!(model, B::BarrierRegion{<:ConstantBarrier{<:AbstractVariableRef}}; M = 5)
    v = B.fun.v
    set_upper_bound(v, M)
end

function upper_bound_constraint!(model, B::BarrierRegion{<:ConstantBarrier{<:Number}}; M = 5) end

function initial_constraint!(model, system, B::BarrierRegion{<:ConstantBarrier}, γ)
    X₀ = initial_set(system)

    if isdisjoint(X₀, region(B))  # Must be strictly disjoint, as this constraint does not involve dynamics
        return nothing
    end

    v = B.fun.v
    @constraint(model, v ≤ γ)
end

function initial_constraint!(model, system, B::BarrierRegion{<:ConstantBarrier{<:AbstractVariableRef}}, γ)
    X₀ = initial_set(system)

    if isdisjoint(X₀, region(B))  # Must be strictly disjoint, as this constraint does not involve dynamics
        return nothing
    end

    v = B.fun.v
    @constraint(model, v ≤ γ)
end

function initial_constraint!(model, system, B::BarrierRegion{<:ConstantBarrier{<:Number}}, γ) end

function unsafe_constraint!(model, system, B::BarrierRegion{<:ConstantBarrier}) end

# Polynomial constraints
function constraints!(model, system, B::PolynomialBarrier, c, γ, expectation_method, solver)
    n_variables = count_variables(B) + 2
    expectation = prepare(system, expectation_method, solver, n_variables)

    # upper_bound_constraint!(model, system, B)
    initial_constraint!(model, system, B, γ)
    unsafe_constraint!(model, system, B)
    expectation_constraint!(model, system, B, c, expectation)
end

function upper_bound_constraint!(model, system::PiecewiseSystem, B::PolynomialBarrier; M = 2)
    x = variables(B)

    for qᵢ in regions(system)
        H, h = tosimplehrep(region(qᵢ))
        qᵢ_symbolic = BasicSemialgebraicSet(FullSpace(), h .- H * x)  # Hx ≤ h

        @constraint(model, M - B.p in SOSCone(), domain=qᵢ_symbolic)
    end
end

function upper_bound_constraint!(model, system::PolynomialSystem, B::PolynomialBarrier; M = 2)
    Xₛ = safe_set(system)

    @constraint(model, M - B.p in SOSCone(), domain=Xₛ)
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
    initial_constraint!(model, initial_set(system), B::PolynomialBarrier, γ)
end

function initial_constraint!(model, region::AbstractHyperrectangle, B::PolynomialBarrier, γ)
    x = variables(B)

    X₀_symbolic = BasicSemialgebraicSet(FullSpace(), radius_hyperrectangle(region) .^ 2 - (x .- center(region)) .^ 2)

    @constraint(model, γ - B.p in SOSCone(), domain=X₀_symbolic)
end

function unsafe_constraint!(model, system::PiecewiseSystem, B::PolynomialBarrier)
    Xᵤ = unsafe_set(system)
    x = variables(B)

    for subset in Xᵤ
        unsafe_constraint!(model, subset, B, x)
    end
end

function unsafe_constraint!(model, region::Complement{T, <:AbstractHyperrectangle}, B::PolynomialBarrier, x) where {T}
    Xₛ = Complement(region)
    subset_symbolic = BasicSemialgebraicSet(FullSpace(), x .^ 2 - radius_hyperrectangle(Xₛ) .^ 2) 
    @constraint(model, B.p - 1.0 in SOSCone(), domain=subset_symbolic)
end

function unsafe_constraint!(model, region::HPolyhedron, B::PolynomialBarrier, x)
    dom_encoding = sos_hpoly_lag(model, x, region)

    @constraint(model, B.p - 1.0 - dom_encoding in SOSCone())
end

function sos_hpoly_lag(model, x, region)
    # Quadratic H-polytope encoding
    H, h = tosimplehrep(region)
    halfspace_constraints = H * x - h

    domain = sum(Iterators.product(halfspace_constraints, halfspace_constraints)) do (constraint1, constraint2)
        # TODO: Skip if the `constraint1 * constraint2` is zero (i.e. the constraints are orthogonal)
        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    return domain
end

function unsafe_constraint!(model, region::Complement{T, <:Universe}, B::PolynomialBarrier, x) where {T}
    # No constraints
end

function unsafe_constraint!(model, region::Complement{T, <:CartesianProduct}, B::PolynomialBarrier, x) where {T}
    Xₛ = Complement(region)
    X₁, X₂ = Xₛ

    n = LazySets.dim(X₁)
    x₁, x₂ = x[1:n], x[1 + n:end]

    unsafe_constraint!(model, Complement(X₁), B, x₁)
    unsafe_constraint!(model, Complement(X₂), B, x₂)
end

function unsafe_constraint!(model, region::Complement{T, <:CartesianProductArray}, B::PolynomialBarrier, x) where {T}
    Xₛ = Complement(region)

    for Xᵢ in array(Xₛ)
        xᵢ = x[1:LazySets.dim(Xᵢ)]
        x = x[LazySets.dim(Xᵢ) + 1:end]
        unsafe_constraint!(model, Complement(Xᵢ), B, xᵢ)
    end
end