using JuMP, MosekTools

function verify(system::System, H::Int, ϵ::Number, num_samples::Int)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")
    @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
    @assert(num_samples >= 1, "Number of samples must be greater than or equal to one (is $num_samples)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false
    )
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
    # println(JuMP.solution_summary(model))

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B, c, γ = value(B), value(c), value(γ)
    println("γ $γ, c $c")

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = γ + c * H

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), prob $prob")

    return succ, B, prob
end

function barrier_variable!(model, system)
    n = size(state(system)) + 1

    B = Dict{Int, Array{VariableRef, 2}}(
           q.index => @variable(model, [1:n, 1:n], Symmetric, base_name = "B[$q]")
           for q in partitions(system)
       )

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
    unsafe_constraint!(model, system, B, q)
    initial_constraint!(model, system, B, γ, q)
    expectation_constraint!(model, system, B, c, ϵ, q, ηs)
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

    @constraint(model, B[q.index] + s_mat >= 0, PSDCone())
end

function unsafe_constraint!(model, system, B, q)
    # Note unsafe set can consist of multiple disjoint components
    overlaps_unsafe = false
    for unsafe_subset in unsafe_set(system)
        if !isempty(unsafe_subset ∩ q)
            overlaps_unsafe = true
            break
        end
    end

    if !overlaps_unsafe
        return
    end

    # We may need to refine partitions if a partition cover too large part of the safe set

    s_mat = s_procedure(q)

    one_mat = zero(s_mat)
    one_mat[end, end] = 1

    @constraint(model, B[q.index] - one_mat + s_mat >= 0, PSDCone())
end

function initial_constraint!(model, system, B, γ, q)
    X_0 = initial_set(system)

    if isempty(X_0 ∩ q)
        return
    end

    s_mat = s_procedure(X_0 ∩ q)

    γ_mat = zero(s_mat)
    γ_mat[end, end] = γ

    @constraint(model, -B[q.index] + γ_mat + s_mat ≥ 0, PSDCone())
end

function expectation_constraint!(model, system, B, c, ϵ, qi, ηs)
    for q_j in partitions(system)
        println("Next partition $q_j")
        expectation_partition_constraint!(model, system, B, c, ϵ, qi, qj, ηs)
    end
end

barrier_blocks(B, q) = B[q.index][:end - 1, :end - 1], B[q.index][:end - 1, end], B[q.index][end, end]
function expectation_partition_constraint!(model, system, B, c, ϵ, qi, qj, ηs)
    Pi, pi, vi = barrier_blocks(B, qi)
    Pj, pj, vj = barrier_blocks(B, qj)

    Ai, bi = next_state(system, qi)

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    M = 1  
    d = ϵ * M

    # Separate LMI into two matrices where one depends on η and one does not
    # This way, we can share computations between constraints

    shared_mat = vcat(
        hcat(Ai' * Pj * Ai - Pi, -pi / 2),
        hcat(-pi' / 2, vj - vi - c + d)
    )

    for η in ηs
        s_mat = s_procedure(qi)

        biη = bi + η

        Aibiη = A_i * biη

        η_mat = vcat(
            hcat(zeros(Pi), Aibiη),
            hcat(Aibiη', biη' * Pj * biη + pj' * biη)
        )

        @constraint(model, -shared_mat - η_mat + s_mat ≥ 0, PSDCone())
    end
end
