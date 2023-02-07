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
required_samples(method::SampleAverage, d) = 1.0 / (method.ϵ^2 * method.β)

function prepare(system, method::SampleAverage, solver, d)
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
function expectation_constraint!(model, system, B::PolynomialSystem, c, expectation::ExpectationData{SampleAverage})
    Xₛ = safe_set(system)
    f = dynamics(system)


    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    d = M * ϵ

    @polyvar w[1:dimensionality(system)]
    B_next = subs(B, variables(system) => f + w)
    B_next = mean([subs(B_next, w => η) for η in expectation.data])

    @constraint(model, -B_next + B + c - d in SOSCone(), domain=Xₛ)
end

### Scenario approach expectation for different families of barriers
function expectation_constraint!(model, system, B::PiecewiseBarrier, c, qᵢ, expectation::ExpectationData{ScenarioApproach})
    Xₛ = safe_set(system)
    if is_intersection_zeromeasure(Xₛ, qᵢ)
        return
    end
    
    qᵢ_next_nom = one_step_reachable_set(qᵢ)
    qᵢ_next = custom_minkowski_sum(qᵢ_next_nom, expectation)

    if LazySets.isbounded(qᵢ_next_nom)
        box = minkowski_sum(box_approximation(qᵢ_next_nom), expectation.data.box)
    else
        box = nothing
    end

    query = RegionIntersectsQuery(qᵢ_next, box)
    next_regions = Spatial.findall(query, regions(system))

    for qⱼ in next_regions
        expectation_partition_constraint!(model, qᵢ, B[qᵢ], B[qⱼ], c, expectation)
    end
end

function one_step_reachable_set(qᵢ)
    dyn = dynamics(qᵢ)
    qᵢ = convert(HPolyhedron, region(qᵢ))
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
    hrep_qᵢ_next_nom = HPolyhedron(constraints_list(qᵢ_next_nom))
    nomial_template_dirs = CustomDirections([normalize(h.a) for h in constraints_list(hrep_qᵢ_next_nom)])
    
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

function expectation_partition_constraint!(model, qᵢ, Bᵢ::BarrierRegion{T, LinearBarrier{T1, VT1}}, Bⱼ::BarrierRegion{T, LinearBarrier{T2, VT2}}, c, expectation::ExpectationData{ScenarioApproach}) where {T, T1, VT1, T2, VT2}
    set = expectation_set(qᵢ, region(Bⱼ), expectation)
    
    if isnothing(set)
        return
    end

    H, h = tosimplehrep(set)

    pᵢ, vᵢ = Bᵢ.fun.p, Bᵢ.fun.v
    pⱼ, vⱼ = Bⱼ.fun.p, Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    d = M * ϵ

    for (Aᵢ, bᵢ) in dynamics(qᵢ)
        a = vcat(Aᵢ' * pⱼ .- (pᵢ / (1 - ϵ)), pⱼ)
        rhs = (vᵢ + c - d) / (1 - ϵ) - vⱼ - pⱼ ⋅ bᵢ

        assymetric_dual_constraint!(model, a, rhs, H, h)
    end
end

function expectation_set(qᵢ, qⱼ, expectation)
    dyn = dynamics(qᵢ)
    qᵢ = region(qᵢ)

    input_noise_sets = cartesian_product(qᵢ, expectation.data.hrep)

    output_set(A, b) = intersection(input_noise_sets, HPolyhedron(constraints_list(InverseLinearMap(hcat(A, I), LazySets.translate(qⱼ, -b)))), prune=false)

    if length(dyn) == 1
        (Aᵢ, bᵢ) = dyn[1]
        backward_reachable_set = output_set(Aᵢ, bᵢ)
        return backward_reachable_set
    end

    output_sets = [output_set(Aᵢ, bᵢ) for (Aᵢ, bᵢ) in dyn]

    output_sets = map(p -> convert(VPolyhedron, p), output_sets)
    output_sets = UnionSetArray{Float64, VPolyhedron{Float64, Vector{Float64}, Vector{Float64}}}(output_sets)

    backward_reachable_set = convex_hull(output_sets, prune=false)

    if isempty(backward_reachable_set)
        return nothing
    end

    backward_reachable_set = normalize(tohrep(backward_reachable_set))

    return backward_reachable_set
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