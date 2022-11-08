using LinearAlgebra

using Distributions
using MosekTools


export verify, barrier_variable!, partition_barrier_variable!, barrier_value, overlaps_unsafe
export constraints!, dual_constraint!, unsafe_constraint!, initial_constraint!
export partition_constraints!, expectation_constraint!, non_negativity_constraint!, expectation_partition_constraint!

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
    B = PiecewiseBarrier(
           qᵢ => partition_barrier_variable!(model, system, qᵢ)
           for qᵢ in partitions(system)
       )

    return B
end

# Optimization variables
function partition_barrier_variable!(model, system, qᵢ)
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

# Constraints
function constraints!(model, system, B, c, γ, ϵ, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)

    for qᵢ in partitions(system)
        println("Partition $qᵢ")
        partition_constraints!(model, system, B, c, γ, ϵ, qᵢ, ηs)
    end
end

function partition_constraints!(model, system, B, c, γ, ϵ, qᵢ, ηs)
    Bᵢ = B[qᵢ]

    non_negativity_constraint!(model, Bᵢ)
    initial_constraint!(model, system, Bᵢ, γ)
    unsafe_constraint!(model, system, Bᵢ)
    expectation_constraint!(model, system, B, c, ϵ, qᵢ, ηs)
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
function non_negativity_constraint!(model, B::BarrierRegion{I, P, LinearBarrier{T, VT}}) where {I, P, T, VT}
    H, h = tosimplehrep(region(B))
    p, v = B.fun.p, B.fun.v
    dual_constraint!(model, -p, v, H, h)
end

function initial_constraint!(model, system, B::BarrierRegion{I, P, LinearBarrier{T, VT}}, γ) where {I, P, T, VT}
    X₀ = initial_set(system)

    if is_intersection_zeromeasure(X₀, region(B))
        return
    end

    H₀, h₀ = tosimplehrep(intersection(X₀, region(B)))
    p, v = B.fun.p, B.fun.v
    dual_constraint!(model, p, γ - v, H₀, h₀)
end

function unsafe_constraint!(model, system, B::BarrierRegion{I, P, LinearBarrier{T, VT}}) where {I, P, T, VT}
    if overlaps_unsafe(system, region(B))
        H, h = tosimplehrep(region(B))
        p, v = B.fun.p, B.fun.v
        dual_constraint!(model, p, v - 1.0, H, h)
    end
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
    Hⱼ, hⱼ = tosimplehrep(qᵢ)

    qᵢ_next = Aᵢ * region(qᵢ) + bᵢ

    pᵢ, vᵢ = B[qᵢ].fun.p, B[qᵢ].fun.v
    pⱼ, vⱼ = B[qⱼ].fun.p, B[qⱼ].fun.v

    for η in eachcol(ηs)
        # println("$qi -> $(qi_next + η), cap $qj")
        if is_intersection_zeromeasure(qⱼ, qᵢ_next + η)
            continue
        end

        dual_constraint!(model, Aᵢ' * pⱼ .- pᵢ, vᵢ - vⱼ + c - d - pⱼ ⋅ (bᵢ + η), vcat(Hᵢ, Hⱼ * Aᵢ), vcat(hᵢ, hⱼ - Hⱼ * (bᵢ + η)))
    end
end

function dual_constraint!(model, c, rhs, H, h)
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
function non_negativity_constraint!(model, B::BarrierRegion{I, P, ConstantBarrier{T}}) where {I, P, T}
    @constraint(model, B.fun.v ≥ 0.0)
end

function initial_constraint!(model, system, B::BarrierRegion{I, P, ConstantBarrier{T}}, γ) where {I, P, T}
    X₀ = initial_set(system)

    if is_intersection_zeromeasure(X₀, region(B))
        return
    end

    @constraint(model, γ ≥ B.fun.v)
end

function unsafe_constraint!(model, system, B::BarrierRegion{I, P, ConstantBarrier{T}}) where {I, P, T}
    if overlaps_unsafe(system, region(B))
        @constraint(model, B.fun.v ≥ 1.0)
    end
end
