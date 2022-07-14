using JuMP, MosekTools, Dualization

using Distributions

function verify(system::System; H::Int, ϵ::Number, num_samples::Int)
    @assert(H >= 1, "Horizon H must be greater than or equal to one (is $H)")
    @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
    @assert(num_samples >= 1, "Number of samples must be greater than or equal to one (is $num_samples)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false
    )
    solver = dual_optimizer(solver)
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

    B, c, γ = barrier_value(B), value(c), value(γ)
    println("γ $γ, c $c")

    # Alternatively use B(x_0) for γ if x_0 is a point
    prob = γ + c * H

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), prob $prob")

    return succ, B, prob
end

function barrier_variable!(model, system)
    B = Dict(
           q => partition_barrier_variable!(model, system, q)
           for q in partitions(system)
       )

    return B
end

function partition_barrier_variable!(model, system, q)
    n = state_space_size(system) + 1

    Xs = safe_set(system)
    if (iszeromeasure ∘ intersection)(Xs, q)
        unsafe_barrier = zeros(n, n)
        unsafe_barrier[end, end] = 1

        return unsafe_barrier
    else
        return @variable(model, [1:n, 1:n], Symmetric, base_name = "B[$q]")
    end
end

function barrier_value(B)
    B = Dict(i => value.(Bi) for (i, Bi) in B)

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

s_procedure(model, q::Partition) = s_procedure(model, region(q))
function s_procedure(model, p)
    m, n = size(p)
    τ = @variable(model, [1:m], lower_bound = 0)
    H, h = tosimplehrep(p)

    Hτ = H' * τ
    hτ = -2 * h' * τ

    mat = vcat(
        hcat(zeros(n, n), Hτ),
        hcat(Hτ', hτ)
    )

    return mat
end

function non_negativity_constraint(model, B, q)
    s_mat = s_procedure(model, q)

    @constraint(model, -B[q] - s_mat ≤ 0, PSDCone())
end

function unsafe_constraint!(model, system, B, q)
    # Note unsafe set can consist of multiple disjoint components
    overlaps_unsafe = false
    for unsafe_subset in unsafe_set(system)
        if (iszeromeasure ∘ intersection)(unsafe_subset, q)
            overlaps_unsafe = true
            break
        end
    end

    if !overlaps_unsafe
        return
    end

    # We may need to refine partitions if a partition cover too large part of the safe set

    s_mat = s_procedure(model, q)

    one_mat = zero(s_mat)
    one_mat[end, end] = 1

    @constraint(model, -B[q] + one_mat - s_mat ≤ 0, PSDCone())
end

function initial_constraint!(model, system, B, γ, q)
    X0 = initial_set(system)

    if (iszeromeasure ∘ intersection)(X0, q)
        return
    end

    s_mat = s_procedure(model, intersection(X0, q))

    γ_mat = zero(s_mat)
    γ_mat[end, end] = γ

    @constraint(model, B[q] - γ_mat - s_mat ≤ 0, PSDCone())
end

function expectation_constraint!(model, system, B, c, ϵ, qi, ηs)
    Xs = safe_set(system)
    if (iszeromeasure ∘ intersection)(Xs, qi)
        return
    end

    for qj in partitions(system)
        println("Next partition $qj")
        expectation_partition_constraint!(model, B, c, ϵ, qi, qj, ηs)
    end
end

barrier_blocks(B, q, N) = B[q][1:N, 1:N], B[q][1:N, N + 1], B[q][N + 1, N + 1]

function expectation_partition_constraint!(model, B, c, ϵ, qi, qj, ηs)
    N = size(qi, 2)
    Pi, pi, vi = barrier_blocks(B, qi, N)
    Pj, pj, vj = barrier_blocks(B, qj, N)

    Ai, bi = dynamics(qi)

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    M = 1  
    d = ϵ * M

    # Separate LMI into two matrices where one depends on η and one does not
    # This way, we can share computations between constraints

    Aipjpi = (Ai' * pj - pi) / 2
    shared_mat = vcat(
        hcat(Ai' * Pj * Ai - Pi, Aipjpi),
        hcat(Aipjpi', vj - vi - c + d)
    )

    # s_mat = s_procedure(model, qi)
    for η in eachcol(ηs)

        biη = bi + η

        AiPjbiη = Ai' * Pj * biη

        η_mat = vcat(
            hcat(zero(Ai), AiPjbiη),
            hcat(AiPjbiη', biη' * Pj * biη .+ pj' * biη)
        )
        # - s_mat
        @constraint(model, shared_mat + η_mat ≤ 0, PSDCone())
    end
end
