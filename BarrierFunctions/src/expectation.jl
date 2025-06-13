using Polyhedra, LinearAlgebra, SparseArrays, StaticArrays
using Statistics

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
struct SampleAverage{T<:Real} <: DataDrivenExpectationMethod
    ϵ::T
    β::T

    function SampleAverage(ϵ::T, β::T) where {T}
        @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
        @assert(0 < β < 1, "Beta must be in range (0, 1) (is $β)")

        return new{T}(ϵ, β)
    end
end
confidence(method::SampleAverage) = method.β
function required_samples(method::SampleAverage, d) 
    M = 2.0
    # n = ceil(Int64, -log(method.β) * (0 - M)^2 / (2 * method.ϵ^2))
    n = ceil(Int64, M / (method.ϵ^2 * method.β))

    return n
end

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
struct ScenarioApproach{T<:Real} <: DataDrivenExpectationMethod
    ϵ::T
    β::T
    conv_hull::Bool

    function ScenarioApproach(ϵ::T, β::T, conv_hull=true) where {T}
        @assert(0 < ϵ < 1, "Epsilon must be in range (0, 1) (is $ϵ)")
        @assert(0 < β < 1, "Beta must be in range (0, 1) (is $β)")

        return new{T}(ϵ, β, conv_hull)
    end
end
confidence(method::ScenarioApproach) = method.β

function required_samples(method::ScenarioApproach, d::T) where {T<:Int}
    lower_bound = d + 1
    upper_bound = ceil(T, (2 / method.ϵ) * log(1 / method.β) + 2 * d + (2 * d / method.ϵ) * log(2 / method.ϵ))

    # Bisection
    while lower_bound + 1 < upper_bound
        mid = floor(T, (lower_bound + upper_bound) / 2)

        # Compute beta for `mid` number of samples this way, as it is numerically sound.
        # If doing it the naive way, then you end up with multiplication of very large and small numbers.
        η = Binomial(mid, method.ϵ)
        b = cdf(η, d - 1)

        if b > method.β
            lower_bound = mid
        else
            upper_bound = mid
        end
    end

    return upper_bound
end

function prepare(system::PiecewiseSystem{N, T}, method::ScenarioApproach, solver, d, num_samples) where {N, T}
    distribution = noise(system)
    ηs = rand(distribution, num_samples)

    if method.conv_hull
        old_length = size(ηs, 2)
        ηs = quick_convex_hull(ηs, solver)
        new_length = size(ηs, 2)
        @debug "Using the convex hull, we reduce the number of samples from $old_length to $new_length"
    end

    ηs = VPolytope(ηs)
    box_ηs = box_approximation(ηs)
    
    vout = Vector{Vector{T}}(undef, length(vertices_list(ηs)) * 2 * 2^N)
    for i in eachindex(vout)
        vout[i] = zeros(T, N)
    end

    return ExpectationData(
        method,
        (vrep=ηs, box=box_ηs, vout=vout)
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
function expectation_constraint!(model, system::PolynomialSystem, B::PolynomialBarrier, c, expectation::ExpectationData{<:SampleAverage})
    Xₛ = safe_set(system)
    f = dynamics(system)
    x = variables(B)

    ϵ = expectation.method.ϵ

    @polyvar ws[1:dimensionality(system)]
    B_next = subs(B.p, x => f + ws)

    # Fast and robust computation of the sample average
    B_next = sum(terms(B_next)) do t
        new_term = subs(t, ws => ones(length(ws)))
        exponents = [MultivariatePolynomials.degree(t, w) for w in ws]
        # Compute 1/N * sum_{i = 1}^N wᵢᵅ where wᵢ is a data vector and α is a vector of exponents.
        empirical_mean = mean(η -> prod(splat(^), zip(η, exponents)), expectation.data)

        return new_term * empirical_mean
    end

    @constraint(model, -B_next - ϵ + B.p + c in SOSCone(), domain=Xₛ)
end

function expectation_constraint!(model, system::PiecewiseSystem, B::PolynomialBarrier, c, expectation::ExpectationData{<:SampleAverage})
    Xₛ = safe_set(system)
    x = variables(B)

    ϵ = expectation.method.ϵ

    @polyvar ws[1:dimensionality(system)]
    @polyvar y[1:dimensionality(system)]
    B_next = subs(B.p, x => y + ws)

    # Fast and robust computation of the sample average
    B_next = sum(terms(B_next)) do t
        new_term = subs(t, ws => ones(length(ws)))
        exponents = [MultivariatePolynomials.degree(t, w) for w in ws]
        # Compute 1/N * sum_{i = 1}^N wᵢᵅ where wᵢ is a data vector and α is a vector of exponents.
        empirical_mean = mean(η -> prod(splat(^), zip(η, exponents)), expectation.data)

        return new_term * empirical_mean
    end

    base_constraint = @expression(model, -B_next - ϵ + B.p + c)

    for qᵢ in regions(system)
        if is_intersection_zeromeasure(Xₛ, qᵢ)
            continue
        end

        fᵢ = dynamics(qᵢ)

        qᵢ = region(qᵢ)
        qᵢ_symbolic = lazysets2semialgebraic(qᵢ, x)

        if length(fᵢ) == 1
            (Aᵢ, bᵢ) = fᵢ[1]

            constraint = subs(base_constraint, y => Aᵢ * x + bᵢ)
            @constraint(model, constraint in SOSCone(), domain=qᵢ_symbolic)
        elseif length(fᵢ) == 2
            (A̲ᵢ, b̲ᵢ), (A̅ᵢ, b̅ᵢ) = fᵢ

            f̲ᵢ = A̲ᵢ * x + b̲ᵢ
            f̅ᵢ = A̅ᵢ * x + b̅ᵢ
    
            y_domain = (f̅ᵢ - y) .* (y - f̲ᵢ)
            y_symbolic = BasicSemialgebraicSet(FullSpace(), y_domain)
    
            @constraint(model, base_constraint in SOSCone(), domain=intersect(qᵢ_symbolic, y_symbolic))
        else
            error("Unsupported number of vertices of convex hull dynamics")
        end
    end
end

function lazysets2semialgebraic(region::AbstractHyperrectangle, x)
    return BasicSemialgebraicSet(FullSpace(), radius_hyperrectangle(region) .^ 2 - (x - center(region)) .^ 2)
end

function lazysets2semialgebraic(region::AbstractPolyhedron, x)
    H, h = tosimplehrep(region)
    return BasicSemialgebraicSet(FullSpace(), h .- H * x)
end

### Scenario approach expectation for different families of barriers
function expectation_constraint!(model, system, B::PiecewiseBarrier, c, qᵢ, expectation::ExpectationData{<:ScenarioApproach})
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
        expectation_partition_constraint!(model, dynamics(qᵢ), B[index(qᵢ)], B[index(qⱼ)], c, expectation)
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

function custom_minkowski_sum(qᵢ_next_nom::VPolyhedron{N}, expectation) where {N}
    rlist1 = rays_list(qᵢ_next_nom)
    vlist1 = vertices_list(qᵢ_next_nom)
    vlist2 = vertices_list(expectation.data.vrep)

    vout = vpolytope_minkowski_sum!(expectation.data.vout, vlist1, vlist2)

    qᵢ_next = VPolyhedron(vout, rlist1)

    return qᵢ_next
end

function vpolytope_minkowski_sum!(vout, vlist1, vlist2)
    i = 1

    for vi in vlist1
        for vj in vlist2
            vout[i] .= vi .+ vj
            i += 1
        end
    end

    vout = vout[1:i - 1]
end

@commutative function LazySets.isdisjoint(P::VPolyhedron{N}, X::AbstractPolyhedron; solver=default_lp_solver(N)) where {N}
    if isempty(P)
        return true
    end

    for v in vertices_list(P)
        if v ∈ X
            return false
        end
    end

    return _isdisjoint_vpolyhedron_lp(P, X; solver=solver)
end

@commutative function LazySets.isdisjoint(P::VPolyhedron{N}, X::AbstractZonotope; solver=default_lp_solver(N)) where {N}
    if isempty(P)
        return true
    end

    for v in vertices_list(P)
        if v ∈ X
            return false
        end
    end

    return _isdisjoint_vpolyhedron_lp(P, X; solver=solver)
end

@commutative function LazySets.isdisjoint(P::VPolyhedron{N}, X::AbstractHyperrectangle; solver=default_lp_solver(N)) where {N}
    if isempty(P)
        return true
    end

    for v in vertices_list(P)
        if v ∈ X
            return false
        end
    end

    HP = box_approximation(VPolytope(vertices_list(P)))
    if LazySets.isbounded(P) && isdisjoint(HP, X)
        return true
    end

    return _isdisjoint_vpolyhedron_lp(P, X; solver=solver)
end


function _isdisjoint_vpolyhedron_lp(P::VPolyhedron{N}, X; solver=default_lp_solver(N)) where {N}
    vertices = vertices_list(P)
    m = length(vertices)

    rays = rays_list(P)
    o = length(rays)

    n = LazySets.dim(P)
    @assert n == LazySets.dim(X) "Dimensionalities must match"

    model = Model(solver)
    set_string_names_on_creation(model, false)

    @variable(model, λ[1:m] ≥ 0)
    @variable(model, μ[1:o] ≥ 0)
    @variable(model, x₀, upper_bound=1.0)

    @objective(model, Max, 1.0 * x₀)

    @constraint(model, sum(λ) == 1)

    x = @expression(model, sum(λ[j] * vertices[j] for j in 1:m) + sum(μ[j] * rays[j] for j in 1:o))

    H, h = tosimplehrep(X)
    @constraint(model, H * x .+ x₀ ≤ h)

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        error("LP returned status $(solution_summary(model)) unexpectedly")
    end

    isdisjoint = objective_value(model) < 0.0
    return isdisjoint
end

function expectation_partition_constraint!(model, dyn, Bᵢ::BarrierRegion{<:LinearBarrier}, Bⱼ::BarrierRegion{<:LinearBarrier}, c, expectation::ExpectationData{<:ScenarioApproach})
    dyn = tighten_dynamics(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(dyn)
        @debug "Tightening dynamics shows non-overlap"
        return nothing
    end

    qᵢ = tighten_region(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(qᵢ)
        @debug "Tightening region non-overlap"
        return nothing
    end

    pᵢ, vᵢ = Bᵢ.fun.p, Bᵢ.fun.v
    pⱼ, vⱼ = Bⱼ.fun.p, Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    ν = M * ϵ / (1 - ϵ)

    t = @variable(model)
    vertices = vertices_list(expectation.data.vrep)
    @constraint(model, [i=eachindex(vertices)], dot(pⱼ, vertices[i]) ≤ t)

    H, h = tosimplehrep(qᵢ)
    H = sparse(H)

    for (Aᵢ, bᵢ) in dyn
        a = @expression(model, Aᵢ' * pⱼ .- pᵢ)
        rhs = @expression(model, vᵢ + c - ν - vⱼ - pⱼ ⋅ bᵢ - t)

        asymmetric_dual_constraint!(model, a, rhs, H, h)
    end
end

function expectation_partition_constraint!(model, dyn, Bᵢ::BarrierRegion{<:ConstantBarrier}, Bⱼ::BarrierRegion{<:ConstantBarrier}, c, expectation::ExpectationData{<:ScenarioApproach})
    dyn = tighten_dynamics(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(dyn)
        @debug "Tightening dynamics shows non-overlap"
        return nothing
    end

    qᵢ = tighten_region(dyn, region(Bᵢ), region(Bⱼ), expectation)
    if isnothing(qᵢ)
        @debug "Tightening region non-overlap"
        return nothing
    end

    vᵢ = Bᵢ.fun.v
    vⱼ = Bⱼ.fun.v

    # We can let M = 1 [max B(x)] as B(x) can be modified aposterori to be capped at 1 without violating the theory
    ϵ = expectation.method.ϵ
    M = 1
    ν = M * ϵ / (1 - ϵ)

    @constraint(model, vⱼ + ν ≤ vᵢ + c)
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