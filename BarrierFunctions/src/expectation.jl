using Polyhedra, LinearAlgebra

export ExpectationMethod, SampleAverage, ScenarioApproach, OverrideNumSamples
export ExpectationData

export confidence, required_samples, prepare
export expectation_constraint!, expectation_partition_constraint!

### Generics
abstract type ExpectationMethod end
abstract type DataDrivenExpectationMethod <: ExpectationMethod end

function prepare(system, method::DataDrivenExpectationMethod, solver, d)
    n_samples = required_samples(method, d)
    @info "Required number of samples given $d variables and the desired confidence $(confidence(method)) is $n_samples."
    return prepare(system, method, solver, d, n_samples)
end

struct ExpectationData{T <: ExpectationMethod}
    method::T
    data
end

### Sample avergae
struct SampleAverage <: DataDrivenExpectationMethod
    ϵ
    β

    function SampleAverage(ϵ, β)
        @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
        @assert(0 < β < 1, "Beta must be in range (0, 1) (is $β)")

        return new(ϵ, β)
    end
end
confidence(method::SampleAverage) = method.β
required_samples(method::SampleAverage, d) = ceil(Int64, 1.0 / (method.ϵ^2 * method.β))

function prepare(system, method::SampleAverage, solver, d, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)
    ηs = [copy(η) for η in eachcol(ηs)]

    return ExpectationData(
        method,
        ηs
    )
end

### The scenario approach
struct ScenarioApproach <: DataDrivenExpectationMethod
    ϵ
    β
    conv_hull

    function ScenarioApproach(ϵ, β, conv_hull=true)
        @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
        @assert(0 < β < 1, "Beta must be in range (0, 1) (is $β)")

        return new(ϵ, β, conv_hull)
    end
end
confidence(method::ScenarioApproach) = method.β

function required_samples(method::ScenarioApproach, d)
    lower_bound = d + 1
    upper_bound = ceil(Int, (ℯ / (ℯ - 1)) * (1 / method.ϵ) * (d + log(1 / method.β)))  # The funny ℯ is Euler's constant

    beta(N) = sum(i -> binomial(big(N), i) * method.ϵ^i * (1 - method.ϵ)^(N - i), 0:(d - 1))

    while lower_bound + 1 < upper_bound
        mid = floor(Int, (lower_bound + upper_bound) / 2)
        b = beta(mid)

        if b > method.β
            lower_bound = mid
        else
            upper_bound = mid
        end
    end

    return upper_bound
end

function prepare(system::PiecewiseSystem, method::ScenarioApproach, solver, d, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)

    if method.conv_hull
        old_length = size(ηs, 2)
        ηs = quick_convex_hull(ηs, solver)
        new_length = size(ηs, 2)
        @debug "Using the convex hull, we reduce the number of samples from $old_length to $new_length"
    end

    ηs = [copy(η) for η in eachcol(ηs)]
    ηs = VPolytope(ηs)
    box_ηs = box_approximation(ηs)

    hrep_ηs = normalize(tohrep(ηs))
    
    template_dirs = CustomDirections([h.a for h in constraints_list(hrep_ηs)])
    template_dirs = [(d, ρ(d, ηs)) for d in template_dirs]

    return ExpectationData(
        method,
        (vrep=ηs, hrep=hrep_ηs, box=box_ηs, template_dirs=template_dirs)
    )
end

function prepare(system::PolynomialSystem, method::ScenarioApproach, solver, d, num_samples)
    distribution = noise(system)
    ηs = rand(distribution, num_samples)
    ηs = [copy(η) for η in eachcol(ηs)]

    if method.conv_hull
        @warn "Convex hull approach does not work polynomial systems"
    end

    return ExpectationData(
        method,
        ηs
    )
end

### Override number of samples for data-driven expectation methods
struct OverrideNumSamples <: ExpectationMethod
    underlying_method::DataDrivenExpectationMethod
    num_samples
end
required_samples(method::OverrideNumSamples, d) = required_samples(method.underlying_method, d)
confidence(method::OverrideNumSamples) = confidence(method.underlying_method)

function prepare(system, method::OverrideNumSamples, d)
    n_samples = required_samples(method, d)
    if method.num_samples < n_samples
        @warn "The requested number of samples $num_samples is less than the required number of samples $n_samples"
    end
    
    return prepare(system, method.underlying_method, d, method.num_samples)
end

### Sample average expecation
function expectation_constraint!(model, system, B::PolynomialBarrier, c, expectation::ExpectationData{SampleAverage})
    Xₛ = safe_set(system)
    f = dynamics(system)

    # We can let M = 1 [max var(B(f(x, w))] as B(x) can be modified aposterori to be clipped to [0, 1] without violating the theory
    ϵ = expectation.method.ϵ

    @polyvar w[1:dimensionality(system)]
    B_next = subs(B.p, variables(system) => f + w)
    B_next = sum(η -> subs(B_next, w => η), expectation.data) / length(expectation.data)

    @constraint(model, -B_next + B.p + c - ϵ in SOSCone(), domain=Xₛ)
end

### Scenario approach expectation for different families of barriers
function expectation_constraint!(model, system, B::PiecewiseBarrier, c, qᵢ, expectation::ExpectationData{ScenarioApproach})
    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        return
    end
    
    qᵢ_next_nom = one_step_reachable_set(dynamics(qᵢ), region(qᵢ))
    qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

    if LazySets.isbounded(qᵢ_next_nom)
        box = minkowski_sum(box_approximation(qᵢ_next_nom), expectation.data.box)
    else
        box = nothing
    end

    query = RegionIntersectsQuery(qᵢ_next, box)
    next_regions = Spatial.findall(query, regions(system))

    @debug "Number of next regions is $(length(next_regions))"

    for qⱼ in next_regions
        expectation_partition_constraint!(model, dynamics(qᵢ), B[qᵢ], B[qⱼ], c, expectation)
    end
end

function one_step_reachable_set(dyn, qᵢ)
    qᵢ = convert(HPolyhedron, qᵢ)
    qᵢ = convert(VPolyhedron, qᵢ)

    reachable_sets = [affine_map(Aᵢ, qᵢ, bᵢ) for (Aᵢ, bᵢ) in dyn]

    if length(reachable_sets) == 1
        return reachable_sets[1]
    end

    reachable_sets = UnionSetArray(reachable_sets)
    reachable_set = convex_hull(reachable_sets, prune=false)

    return reachable_set
end

function custom_minkowski_sum(qᵢ_next_nom::AbstractPolyhedron{N}, expectation) where {N}
    n = LazySets.dim(qᵢ_next_nom)

    # Center and rescale qᵢ_next_nom for finding the normal vectors, to ameliorate numerical issues.
    # Centering and rescaling does not impact the direction of the normal vectors, so
    # we can do this without risk.
    v = vertices_list(qᵢ_next_nom)
    center = reduce((.*), v) ./ length(v)
    centered_qᵢ_next_nom = LazySets.translate(qᵢ_next_nom, -center)
    max_norm = maximum(norm, vertices_list(centered_qᵢ_next_nom))
    δ = inv(prevfloat(typemax(max_norm)))
    if max_norm ≥ δ # Safe to multiply with inverse
        invnrm = inv(max_norm)
        centered_rescaled_qᵢ_next_nom = linear_map(Matrix(I * invnrm, n, n), centered_qᵢ_next_nom)
    else # scale elements to avoid overflow
        εδ = eps(one(max_norm))/δ
        centered_rescaled_qᵢ_next_nom = linear_map(Matrix(I * εδ, n, n), centered_qᵢ_next_nom)
        centered_rescaled_qᵢ_next_nom = linear_map(Matrix(I * inv(max_norm*εδ), n, n), centered_rescaled_qᵢ_next_nom)
    end
    centered_rescaled_hrep_qᵢ_next_nom = tohrep(centered_rescaled_qᵢ_next_nom)
    nomial_template_dirs = [normalize(h.a) for h in constraints_list(centered_rescaled_hrep_qᵢ_next_nom)]
    
    # Use support vectors to find halfspace constant term. 
    H = Vector{LazySets.HalfSpace{N, Vector{N}}}()
    sizehint!(H, length(nomial_template_dirs) + length(expectation.data.template_dirs))

    for d in nomial_template_dirs
        sf = ρ(d, qᵢ_next_nom) + ρ(d, expectation.data.vrep)
        if !isinf(sf)
            push!(H, LazySets.HalfSpace(d, sf))
        end
    end

    for (d, ρ_val_noise) in expectation.data.template_dirs
        sf = ρ(d, qᵢ_next_nom) + ρ_val_noise
        if !isinf(sf)
            push!(H, LazySets.HalfSpace(d, sf))
        end
    end

    qᵢ_next = HPolyhedron(H)
    qᵢ_next = normalize(qᵢ_next)

    return qᵢ_next
end

function expectation_partition_constraint!(model, dyn, Bᵢ::BarrierRegion{T, LinearBarrier{T1, VT1}}, Bⱼ::BarrierRegion{T, LinearBarrier{T2, VT2}}, c, expectation::ExpectationData{ScenarioApproach}) where {T, T1, VT1, T2, VT2}
    dyn = tighten_dynamics(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(dyn)
        @debug "Tightening dynamics shows non-overlap"
        return
    end

    qᵢ = tighten_region(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(qᵢ)
        @debug "Tightening region non-overlap"
        return
    end
    
    input_noise_sets = cartesian_product(qᵢ, expectation.data.hrep)
    H, h = tosimplehrep(input_noise_sets)

    pᵢ, vᵢ = Bᵢ.fun.p, Bᵢ.fun.v
    pⱼ, vⱼ = Bⱼ.fun.p, Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    d = M * ϵ

    for (Aᵢ, bᵢ) in dyn
        a = vcat(Aᵢ' * pⱼ .- (pᵢ / (1 - ϵ)), pⱼ)
        rhs = (vᵢ + c - d) / (1 - ϵ) - vⱼ - pⱼ ⋅ bᵢ

        assymetric_dual_constraint!(model, a, rhs, H, h)
    end
end

function single_entry_vector(i, n)
    v = zeros(n)
    v[i] = 1.0

    return v
end

dynamics_vertices(dyn, lambda_polytope) = [AffineDynamics(
    sum(x -> x[1].A * x[2], zip(dyn, vertex)),
    sum(x -> x[1].b * x[2], zip(dyn, vertex))) for vertex in vertices_list(lambda_polytope)]

function tighten_dynamics(dyn, qᵢ, qⱼ, expectation)
    n = length(dyn)
    if n == 1
        return dyn
    end

    lambda_polytope = VPolytope([single_entry_vector(i, n) for i in 1:n])

    # For each lambda_i
    for i in 1:n
        normal_vector = fill(1/n, n)
        normal_vector[i] = -1.0

        excluded_vertex = σ(-normal_vector, lambda_polytope)

        vertices = vertices_list(lambda_polytope)
        vertices = filter(vertex -> !all(isapprox.(vertex, excluded_vertex)), vertices)

        plane_polytope = VPolytope(vertices)
        plane_dyn = dynamics_vertices(dyn, plane_polytope)

        qᵢ_next_nom = one_step_reachable_set(plane_dyn, qᵢ)
        qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

        if isdisjoint(qᵢ_next, qⱼ)
            lambda_polytope = binary_search_tighten_dynamics(dyn, qᵢ, qⱼ, expectation, lambda_polytope, normal_vector, excluded_vertex, first(vertices))
        end
    end

    dyn = dynamics_vertices(dyn, lambda_polytope)

    qᵢ_next_nom = one_step_reachable_set(dyn, qᵢ)
    qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

    if isdisjoint(qᵢ_next, qⱼ)
        return nothing
    end

    return dyn
end

function binary_search_tighten_dynamics(dyn, qᵢ, qⱼ, expectation, lambda_polytope, normal_vector, min_vertex, max_vertex; max_steps=5)
    min = dot(normal_vector, min_vertex)
    max = dot(normal_vector, max_vertex)

    step = 1

    # Perform binary search
    while step <= max_steps
        mid = (min + max) / 2.0

        halfspace = LazySets.HalfSpace(-normal_vector, -mid)
        prune_polytope = intersection(lambda_polytope, halfspace, prune=false)
        prune_dyn = dynamics_vertices(dyn, prune_polytope)

        qᵢ_next_nom = one_step_reachable_set(prune_dyn, qᵢ)
        qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

        if isdisjoint(qᵢ_next, qⱼ)
            lambda_polytope = intersection(lambda_polytope, complement(halfspace), prune=false)
            max = mid
        else
            min = mid
        end

        step += 1
    end

    return lambda_polytope
end

function tighten_region(dyn, qᵢ, qⱼ, expectation)
    constraints = constraints_list(qᵢ)
    # For each halfspace constraint
    for h in constraints
        normal_vector = h.a
        max = h.b
        min = -ρ(-normal_vector, qᵢ)

        if !isinf(min)
            qᵢ = binary_search_tighten_region(dyn, qᵢ, qⱼ, expectation, normal_vector, max, min)
        end
    end

    qᵢ_next_nom = one_step_reachable_set(dyn, qᵢ)
    qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

    if isdisjoint(qᵢ_next, qⱼ)
        return nothing
    end

    return qᵢ
end

function binary_search_tighten_region(dyn, qᵢ, qⱼ, expectation, normal_vector, max, min; max_steps=5)
    step = 1

    # Perform binary search
    while step <= max_steps
        mid = (min + max) / 2.0

        halfspace = LazySets.HalfSpace(-normal_vector, -mid)
        prune_polytope = intersection(qᵢ, halfspace, prune=false)

        qᵢ_next_nom = one_step_reachable_set(dyn, prune_polytope)
        qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

        if isdisjoint(qᵢ_next, qⱼ)
            qᵢ = intersection(qᵢ, complement(halfspace), prune=false)
            max = mid
        else
            min = mid
        end

        step += 1
    end

    return qᵢ
end

function expectation_partition_constraint!(model, Aᵢ, bᵢ, Bᵢ::BarrierRegion{T, ConstantBarrier{T1}}, Bⱼ::BarrierRegion{T, ConstantBarrier{T2}}, c, expectation::ExpectationData{ScenarioApproach}) where {T, T1, T2}
    qᵢ_next = Aᵢ * region(Bᵢ) + bᵢ

    vᵢ = Bᵢ.fun.v
    vⱼ = Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 5
    d = M * ϵ / (1 - ϵ)

    for η in vertices_list(expectation.data[1])
        if is_intersection_zeromeasure(region(Bⱼ), qᵢ_next + η)
            continue
        end

        @constraint(model, vⱼ - vᵢ - c + d ≤ 0.0)
    end
end

function expectation_partition_constraint!(model, Aᵢ, bᵢ, Bᵢ::BarrierRegion{T, QuadraticBarrier{T1, VT1, VVT1}}, Bⱼ::BarrierRegion{T, QuadraticBarrier{T2, VT2, VVT2}}, c, expectation::ExpectationData{ScenarioApproach}) where {T, T1, VT1, VVT1, T2, VT2, VVT2}
    Pᵢ, pᵢ, vᵢ = Bᵢ.fun.P, Bᵢ.fun.p, Bᵢ.fun.v
    Pⱼ, pⱼ, vⱼ = Bⱼ.fun.P, Bⱼ.fun.p, Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    d = M * ϵ / (1 - ϵ)

    for η in vertices_list(expectation.data[1])
        bᵢη = bᵢ + η
        quadratic = Aᵢ' * Pⱼ * Aᵢ - Pᵢ
        linear = Aᵢ' * Pⱼ * bᵢη + 0.5 * Aᵢ' * pⱼ - 0.5 * pᵢ 
        constant = bᵢη' * Pⱼ * bᵢη + bᵢη' * pⱼ + vⱼ - vᵢ - c + d

        mat = vcat(
            hcat(quadratic, linear),
            hcat(linear', constant)
        )

        s_mat = s_procedure(model, qi)
        @constraint(model, mat - s_mat ≤ 0, PSDCone())
    end
end

function expectation_constraint!(model, system, B::PolynomialBarrier, c, expectation::ExpectationData{ScenarioApproach})
    Xₛ = safe_set(system)
    f = dynamics(system)

    @polyvar w[1:dimensionality(system)]
    B_next = subs(B.p, variables(system) => f + w)

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    d = M * ϵ

    for η in expectation.data
        @constraint(model, -subs(B_next, w => η) + (B.p + c - d) / (1 - ϵ) in SOSCone(), domain=Xₛ)
    end
end