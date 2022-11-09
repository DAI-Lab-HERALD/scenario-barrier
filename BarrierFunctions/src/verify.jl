using LinearAlgebra

using Distributions
using MosekTools


export verify, barrier_variable!, partition_barrier_variable!, barrier_value, overlaps_unsafe
export constraints!, assymetric_dual_constraint!, unsafe_constraint!, initial_constraint!
export region_constraints!, expectation_constraint!, non_negativity_constraint!, expectation_partition_constraint!

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

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    M = 1
    @variable(model, d, lower_bound = M * ϵ)
    @constraint(model, d ≤ c)

    n_variables = count_variables(B) + 3

    # Constraints
    constraints!(model, system, B, c, γ, d, num_samples)

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

    return succ, B, prob, n_variables
end

function barrier_variable!(model, system)
    B = PiecewiseBarrier(
           qᵢ => region_barrier_variable!(model, system, qᵢ)
           for qᵢ in regions(system)
       )

    return B
end

# Optimization variables
function region_barrier_variable!(model, system, qᵢ)
    n = state_space_size(system)

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

barrier_blocks(B, qᵢ) = B[qᵢ].fun.p, B[qᵢ].fun.v

barrier_value(B::PiecewiseBarrier) = PiecewiseBarrier(i => barrier_value(Bᵢ) for (i, Bᵢ) in B)
barrier_value(B::BarrierRegion) = BarrierRegion(index(B), region(B), barrier_value(B.fun))
barrier_value(B::LinearBarrier) = LinearBarrier(value.(B.p), value(B.v))
barrier_value(B::ConstantBarrier) = ConstantBarrier(value(B.v))

count_variables(B::PiecewiseBarrier) = sum(((i, Bᵢ),) -> count_variables(Bᵢ), B)
count_variables(B::BarrierRegion) = count_variables(B.fun)
count_variables(B::LinearBarrier{T, VT}) where {T<:AbstractVariableRef, VT} = length(B.p) + 1
count_variables(B::LinearBarrier{T, VT}) where {T<:Number, VT} = 0
count_variables(B::ConstantBarrier{T}) where {T<:AbstractVariableRef} = 1
count_variables(B::ConstantBarrier{T}) where {T<:Number} = 0

# Constraints
function constraints!(model, system, B, c, γ, d, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)

    for qᵢ in regions(system)
        println("Partition $qᵢ")
        region_constraints!(model, system, B, c, γ, d, qᵢ, ηs)
    end
end

function region_constraints!(model, system, B, c, γ, d, qᵢ, ηs)
    Bᵢ = B[qᵢ]

    non_negativity_constraint!(model, Bᵢ)
    initial_constraint!(model, system, Bᵢ, γ)
    unsafe_constraint!(model, system, Bᵢ)
    expectation_constraint!(model, system, B, c, d, qᵢ, ηs)
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
end

function initial_constraint!(model, system, B::BarrierRegion{T, LinearBarrier{T1, VT1}}, γ) where {T, T1, VT1}
    X₀ = initial_set(system)

    if is_intersection_zeromeasure(X₀, region(B))
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
        assymetric_dual_constraint!(model, p, v - 1.0, H, h)
    end
end

function expectation_constraint!(model, system, B, c, d, qᵢ, ηs)
    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        return
    end

    Bᵢ = B[qᵢ]
    Aᵢ, bᵢ = dynamics(qᵢ)
    qᵢ_next = Aᵢ * region(Bᵢ) + bᵢ

    for η in eachcol(ηs)
        query = RegionIntersectsQuery(qᵢ_next + η)
        next_regions = Spatial.findall(query, regions(system))

        for qⱼ in next_regions
            # println("Next partition $qj")
            expectation_partition_constraint!(model, Aᵢ, bᵢ, Bᵢ, B[qⱼ], c, d, η)
        end
    end
end

function expectation_partition_constraint!(model, Aᵢ, bᵢ, Bᵢ::BarrierRegion{T, LinearBarrier{T1, VT1}}, Bⱼ::BarrierRegion{T, LinearBarrier{T2, VT2}}, c, d, η) where {T, T1, VT1, T2, VT2}
    Hᵢ, hᵢ = tosimplehrep(region(Bᵢ))
    Hⱼ, hⱼ = tosimplehrep(region(Bᵢ))

    pᵢ, vᵢ = Bᵢ.fun.p, Bᵢ.fun.v
    pⱼ, vⱼ = Bⱼ.fun.p, Bⱼ.fun.v

    assymetric_dual_constraint!(model, Aᵢ' * pⱼ .- pᵢ, vᵢ - vⱼ + c - d - pⱼ ⋅ (bᵢ + η), vcat(Hᵢ, Hⱼ * Aᵢ), vcat(hᵢ, hⱼ - Hⱼ * (bᵢ + η)))
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

function expectation_partition_constraint!(model, Aᵢ, bᵢ, Bᵢ::BarrierRegion{T, ConstantBarrier{T1}}, Bⱼ::BarrierRegion{T, ConstantBarrier{T2}}, c, d, η) where {T, T1, T2}
    qᵢ_next = Aᵢ * region(Bᵢ) + bᵢ

    vᵢ = Bᵢ.fun.v
    vⱼ = Bⱼ.fun.v

    for η in eachcol(ηs)
        # println("$qi -> $(qi_next + η), cap $qj")
        if is_intersection_zeromeasure(region(Bⱼ), qᵢ_next + η)
            continue
        end

        @constraint(model, vⱼ - vᵢ - c + d ≤ 0.0)
    end
end