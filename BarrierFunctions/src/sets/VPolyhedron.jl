import Base: rand,
             ∈
using Random, Polyhedra

export VPolyhedron,
       vertices_list,
       rays_list,
       linear_map,
       remove_redundant_vertices,
       tohrep,
       tovrep

"""
    VPolyhedron{N, VNV<:AbstractVector{N}, VNR<:AbstractVector{N}} <: AbstractPolyhedron{N}
Type that represents a convex polyhedron in vertex/ray/rays representation.
### Fields
- `vertices` -- list of vertices
- `rays` -- list of rays
### Examples
A polyhedron in vertex/ray representation can be constructed by passing the list of
vertices. For example, we can build the tetrahedron:
```jldoctest
julia> P = VPolyhedron([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]);
julia> P.vertices
4-element Vector{Vector{Int64}}:
 [0, 0, 0]
 [1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]
```
Alternatively, a `VPolyhedron` can be constructed passing a matrix of vertices,
where each *column* represents a vertex, and a matrix of rays, where each *column*
represents a ray:
```jldoctest
julia> M = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'
3×4 adjoint(::Matrix{Int64}) with eltype Int64:
 0  1  0  0
 0  0  1  0
 0  0  0  1
julia> P = VPolyhedron(M);
julia> P.vertices
4-element Vector{Vector{Int64}}:
 [0, 0, 0]
 [1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]
```
"""
##### TODO: Update jldoctest


struct VPolyhedron{N, VNV<:AbstractVector{N}, VNR<:AbstractVector{N}} <: AbstractPolyhedron{N}
    vertices::Vector{VNV}
    rays::Vector{VNR}
end

LazySets.isoperationtype(::Type{<:VPolyhedron}) = false

# constructor with empty vertices and rays lists
VPolyhedron{N}() where {N} = VPolyhedron(Vector{Vector{N}}(), Vector{Vector{N}}())

# constructor with no vertices or rays of type Float64
VPolyhedron() = VPolyhedron{Float64}()

# constructor from rectangular matrix
function VPolyhedron(vertices_matrix::MV, rays_matrix::MR) where {N, MV<:AbstractMatrix{N}, MR<:AbstractMatrix{N}}
    vertices = [vertices_matrix[:, j] for j in axes(vertices_matrix, 2)]
    rays = [rays_matrix[:, j] for j in axes(rays_matrix, 2)]
    return VPolyhedron(vertices, rays)
end

"""
    dim(P::VPolyhedron)
Return the dimension of a polyhedron in vertex/ray representation.
### Input
- `P` -- polyhedron in vertex/ray representation
### Output
The ambient dimension of the polyhedron in vertex/ray representation.
If `P` is empty, the result is ``-1``.
### Examples
```jldoctest
julia> P = VPolyhedron();
julia> isempty(P.vertices) && isempty(P.rays)
true
julia> dim(P)
-1
julia> P = VPolyhedron([ones(3)], []);
julia> P.vertices
1-element Vector{Vector{Float64}}:
 [1.0, 1.0, 1.0]
julia> dim(P) == 3
true
julia> P = VPolyhedron([], [ones(3)]);
julia> P.rays
1-element Vector{Vector{Float64}}:
 [1.0, 1.0, 1.0]
julia> dim(P) == 3
true
```
"""
function LazySets.dim(P::VPolyhedron)
    if !isempty(P.vertices)
        return @inbounds length(P.vertices[1])
    elseif !isempty(P.rays)
        return @inbounds length(P.rays[1])
    else
        return -1
    end
end

"""
    ρ(d::AbstractVector{M}, P::VPolyhedron{N};
      solver=default_lp_solver(N)) where {M, N}
Evaluate the support function of a polyhedron in vertex/ray representation in a
given direction.
### Input
- `d`      -- direction
- `P`      -- polyhedron in vertex/ray representation
- `solver` -- (optional, default: `default_lp_solver(N)`) the backend used to
              solve the linear program
### Output
The evaluation of the support function for the polyhedron.
If a polyhedron is unbounded in the given direction, the result is `Inf`.
"""
function LazySets.ρ(d::AbstractVector{M}, P::VPolyhedron{N};
           solver=default_lp_solver(N)) where {M, N}
    res, unbounded = σ_helper(d, P, solver)
    if unbounded
        return N(Inf)
    end
    return dot(d, res)
end

"""
    σ(d::AbstractVector{M}, P::VPolyhedron{N};
      solver=default_lp_solver(M, N)) where {M, N}
Return a support vector of a polyhedron in vertex/ray representation in a given
direction.
### Input
- `d` -- direction
- `P` -- polyhedron in vertex/ray representation
### Output
A support vector in the given direction.
### Algorithm
A support vector maximizes the support function.
For a polyhedron, the support function is always maximized in some vertex.
Hence it is sufficient to check all vertices.
"""
function LazySets.σ(d::AbstractVector{M}, P::VPolyhedron{N};
    solver=default_lp_solver(N)) where {M, N}

    res, unbounded = σ_helper(d, P, solver)
    if unbounded
        ray = sol
        res = Vector{N}(undef, length(ray))
        @inbounds for i in 1:length(ray)
            if ray[i] == zero(N)
                res[i] = zero(N)
            elseif ray[i] > zero(N)
                res[i] = N(Inf)
            else
                res[i] = N(-Inf)
            end
        end
    end

    return res
end

function σ_helper(d, P, solver)
    vertices = P.vertices
    rays = P.rays

    # base case
    m = length(vertices)
    o = length(rays)

    if m == 0
        error("the support function of an empty polyhedron is undefined")
    end

    if o == 0
        if m == 1
            @inbounds return vertices[1], false
        end

        # evaluate support function in every vertex
        N = promote_type(eltype(d), eltype(P))
        max_ρ = N(-Inf)
        max_idx = 0
        for (i, vi) in enumerate(vertices)
            ρ_i = dot(d, vi)
            if ρ_i > max_ρ
                max_ρ = ρ_i
                max_idx = i
            end
        end
        @inbounds return vertices[max_idx], false
    end

    model = Model(solver)
    n = LazySets.dim(P)

    @variable(model, x[1:n])
    @variable(model, λ[1:m] ≥ 0)
    @variable(model, μ[1:o] ≥ 0)

    @constraint(model, sum(λ) == 1)

    multiply((w, v)) = w .* v
    @constraint(model, sum(multiply, zip(λ, vertices)) .+ sum(multiply, zip(μ, rays)) .== x)

    @objective(model, Max, dot(d, x))

    optimize!(model)

    if termination_status(model) == MathOptInterface.OPTIMAL
        res = value.(x)
        unbounded = false
    elseif primal_status(model) == MathOptInterface.INFEASIBILITY_CERTIFICATE
        res = value.(x)
        unbounded = true
    else
        error("LP solver did not return an infeasibility ray")
    end

    return res, unbounded
end

"""
    ∈(x::AbstractVector{N}, P::VPolyhedron{N};
      solver=default_lp_solver(N)) where {N}
Check whether a given point is contained in a polyhedron in vertex/ray representation.
### Input
- `x`      -- point/vector
- `P`      -- polyhedron in vertex/ray representation
- `solver` -- (optional, default: `default_lp_solver(N)`) the backend used to
              solve the linear program
### Output
`true` iff ``x ∈ P``.
### Algorithm
We check, using linear programming, the definition of a convex polyhedron that a
point is in the set if and only if it is the sum of a convex combination of the vertices
and a conic combination of the rays.
Let ``\\{v_j\\}`` be the ``m`` vertices and `\\{r_k\\}`` be the ``o`` rays of `P`.
Then we solve the following ``m + o``-dimensional linear program.
```math
\\max 0 \\text{ s.t. }
\\bigwedge_{i=1}^n \\sum_{j=1}^m λ_j v_j[i] + \\sum_{k=1}^n μ_k r_k[i] = x[i]
∧ \\sum_{j=1}^m λ_j = 1
∧ \\bigwedge_{j=1}^m λ_j ≥ 0
∧ \\bigwedge_{k=1}^n μ_k ≥ 0
```
"""
function ∈(x::AbstractVector{N}, P::VPolyhedron{N};
           solver=default_lp_solver(N)) where {N}
    vertices = P.vertices
    m = length(vertices)

    rays = P.rays
    o = length(rays)

    # special cases: 0 vertex
    if m == 0
        return false
    end

    n = length(x)
    @assert n == LazySets.dim(P) "a vector of length $(length(x)) cannot be " *
        "contained in a polyhedron of dimension $(LazySets.dim(P))"

    model = Model(solver)

    @variable(model, λ[1:m] ≥ 0)
    @variable(model, μ[1:o] ≥ 0)

    @constraint(model, sum(λ) == 1)
    @constraint(model, sum(*, zip(λ, vertices)) .+ sum(*, zip(μ, rays)) .== x)

    optimize!(model)

    if termination_status(model) == MathOptInterface.OPTIMAL
        return true
    elseif termination_status(model) == MathOptInterface.INFEASIBLE
        return false
    end
    error("LP returned status $(lp.status) unexpectedly")
end

"""
    rand(::Type{VPolyhedron}; [N]::Type{<:Real}=Float64, [dim]::Int=2,
         [rng]::AbstractRNG=GLOBAL_RNG, [seed]::Union{Int, Nothing}=nothing,
         [num_vertices]::Int=-1)
Create a random polyhedron in vertex/ray representation.
### Input
- `VPolyhedron`    -- type for dispatch
- `N`            -- (optional, default: `Float64`) numeric type
- `dim`          -- (optional, default: 2) dimension
- `rng`          -- (optional, default: `GLOBAL_RNG`) random number generator
- `seed`         -- (optional, default: `nothing`) seed for reseeding
- `num_vertices` -- (optional, default: `-1`) upper bound on the number of
                    vertices of the polyhedron (see comment below)
### Output
A random polyhedron in vertex/ray representation.
### Algorithm
All numbers are normally distributed with mean 0 and standard deviation 1.
The number of vertices can be controlled with the argument `num_vertices`.
For a negative value we choose a random number in the range `dim:5*dim` (except
if `dim == 1`, in which case we choose in the range `1:2`).
Note that we do not guarantee that the vertices are not redundant.
"""
function rand(::Type{VPolyhedron};
              N::Type{<:Real}=Float64,
              dim::Int=2,
              rng::AbstractRNG=GLOBAL_RNG,
              num_vertices::Int=-1,
              num_rays::Int=-1)
    rng = reseed(rng, seed)
    if num_vertices < 0
        num_vertices = (dim == 1) ? rand(rng, 1:2) : rand(rng, dim:5*dim)
    end
    if num_rays < 0
        num_rays = rand(rng, 1:dim)
    end
    vertices = [randn(rng, N, dim) for i in 1:num_vertices]
    rays = [randn(rng, N, dim) for i in 1:num_rays]
    return VPolyhedron(vertices, rays)
end

"""
    linear_map(M::AbstractMatrix, P::VPolyhedron; [apply_convex_hull]::Bool=false)
Concrete linear map of a polyhedron in vertex/ray representation.
### Input
- `M` -- matrix
- `P` -- polyhedron in vertex/ray representation
- `apply_convex_hull` -- (optional, default: `false`) flag for applying a convex
                         hull to eliminate redundant vertices
### Output
A polyhedron in vertex/ray representation.
### Algorithm
The linear map ``M`` is applied to each vertex of the given set ``P``, obtaining
a polyhedron in vertex/ray representation. The output type is again a `VPolyhedron`.
"""
function LazySets.linear_map(M::AbstractMatrix, P::VPolyhedron;
                    apply_convex_hull::Bool=false)
    @assert LazySets.dim(P) == size(M, 2) "a linear map of size $(size(M)) cannot be " *
        "applied to a set of dimension $(LazySets.dim(P))"

    return _linear_map_vrep(M, P; apply_convex_hull=apply_convex_hull)
end

@inline function _linear_map_vrep(M::AbstractMatrix, P::VPolyhedron;
                                  apply_convex_hull::Bool=false)
    vlist = broadcast(v -> M * v, P.vertices)
    rlist = broadcast(v -> M * v, P.rays)
    if apply_convex_hull
        convex_hull!(vlist)
        conic_hull!(rlist)
    end
    return VPolyhedron(vlist, rlist)
end

"""
    translate(P::VPolyhedron, v::AbstractVector)
Translate (i.e., shift) a polyhedron in vertex/ray representation by a given vector.
### Input
- `P` -- polyhedron in vertex/ray representation
- `v` -- translation vector
### Output
A translated polyhedron in vertex/ray representation.
### Notes
See [`translate!(::VPolyhedron, ::AbstractVector)`](@ref) for the in-place version.
"""
function LazySets.translate(P::VPolyhedron, v::AbstractVector)
    return LazySets.translate!(deepcopy(P), v)
end

"""
    translate!(P::VPolyhedron, v::AbstractVector)
Translate (i.e., shift) a polyhedron in vertex/ray representation by a given vector,
in-place.
### Input
- `P` -- polyhedron in vertex/ray representation
- `v` -- translation vector
### Output
The polyhedron `P` translated by `v`.
### Notes
See [`translate(::VPolyhedron, ::AbstractVector)`](@ref) for the out-of-place
version.
### Algorithm
We add the vector to each vertex of the polyhedron.
"""
function LazySets.translate!(P::VPolyhedron, v::AbstractVector)
    if isempty(P.vertices)
        return P
    end

    @assert length(v) == LazySets.dim(P) "cannot translate a $(LazySets.dim(P))-dimensional " *
                                "set by a $(length(v))-dimensional vector"
    for x in P.vertices
        x .+= v
    end
    return P
end

"""
    vertices_list(P::VPolyhedron)
Return the list of vertices of a polyhedron in vertex/ray representation.
### Input
- `P` -- polyhedron in vertex/ray representation
### Output
The list of vertices.
"""
function LazySets.vertices_list(P::VPolyhedron)
    return P.vertices
end

"""
    rays_list(P::VPolyhedron)
Return the list of rays of a polyhedron in vertex/ray representation.
### Input``
- `P` -- polyhedron in vertex/ray representation
### Output
The list of rays.
"""
function rays_list(P::VPolyhedron)
    return P.rays
end

"""
    constraints_list(P::VPolyhedron)
Return a list of constraints defining a polyhedron in vertex/ray representation.
### Input
- `P` -- polyhedron in vertex/ray representation
### Output
A list of constraints of the polyhedron.
### Algorithm
We use `tohrep` to compute the constraint representation of `P`.
"""
function LazySets.constraints_list(P::VPolyhedron)
    return constraints_list(tohrep(P))
end

"""
    remove_redundant_vertices(P::VPolyhedron{N};
                              [backend]=nothing,
                              [solver]=nothing) where {N}
Return the polyhedron obtained by removing the redundant vertices of the given
polyhedron in vertex/ray representation.
### Input
- `P`       -- polyhedron in vertex/ray representation
- `backend` -- (optional, default: `nothing`) the backend for polyhedral
               computations; see `default_polyhedra_backend(P)` or
               [Polyhedra's documentation](https://juliapolyhedra.github.io/)
               for further information
- `solver`  -- (optional, default: `nothing`) the linear programming
               solver used in the backend, if needed; see
               `default_lp_solver_polyhedra(N)`
### Output
A new polyhedron such that its vertices are the convex hull of the given polyhedron.
### Notes
The optimization problem associated to removing redundant vertices is handled
by `Polyhedra`.
If the polyhedral computations backend requires an LP solver, but it has not
been specified, we use `default_lp_solver_polyhedra(N)` to define such solver.
Otherwise, the redundancy-removal function of the polyhedral backend is used.
"""
function LazySets.remove_redundant_vertices(P::VPolyhedron{N};
                                   backend=nothing,
                                   solver=nothing) where {N}
    require(@__MODULE__, :Polyhedra; fun_name="remove_redundant_vertices")
    if backend == nothing
        backend = default_polyhedra_backend(P)
    end
    Q = polyhedron(P; backend=backend)
    if Polyhedra.supportssolver(typeof(Q))
        if solver == nothing
            solver = default_lp_solver_polyhedra(N)
        end
        vQ = Polyhedra.vrep(Q)
        Polyhedra.setvrep!(Q, Polyhedra.removevredundancy(vQ, solver))
    else
        Polyhedra.removevredundancy!(Q; ztol=_ztol(N))
    end
    return VPolyhedron(Q)
end

"""
    tohrep(P::VPolyhedron{N};
           [backend]=default_polyhedra_backend(P)) where {N}
Transform a polyhedron in vertex/ray representation to a polyhedron in constraint
representation.
### Input
- `P`       -- polyhedron in vertex/ray representation
- `backend` -- (optional, default: `default_polyhedra_backend(P)`) the
               backend for polyhedral computations; see [Polyhedra's
               documentation](https://juliapolyhedra.github.io/) for further
               information
### Output
A `HPolyhedron` as the constraint representation of `P`.
### Notes
The conversion may not preserve the numeric type (e.g., with `N == Float32`)
depending on the backend.
"""
function LazySets.tohrep(P::VPolyhedron{N};
                backend=default_polyhedra_backend(P)) where {N}
    @assert !isempty(P.vertices) "cannot convert an empty polyhedron in vertex/ray " *
        "representation to constraint representation"
    require(@__MODULE__, :Polyhedra; fun_name="tohrep")
    return convert(HPolyhedron, polyhedron(P; backend=backend))
end

"""
    tovrep(P::VPolyhedron)
Return a vertex/ray representation of the given polyhedron in vertex/ray representation
(no-op).
### Input
- `P` -- polyhedron in vertex/ray representation
### Output
The same polyhedron instance.
"""
function LazySets.tovrep(P::VPolyhedron)
    return P
end

const HPoly = Union{HPolyhedron, HPolytope}

function Base.convert(::Type{VPolyhedron}, P::HPoly; 
        backend=default_polyhedra_backend(P))
    require(@__MODULE__, :Polyhedra; fun_name="convert")
    P = polyhedron(P; backend=backend)
    return VPolyhedron(P)
end

function Base.convert(::Type{VPolyhedron}, ::EmptySet)
    return VPolyhedron()
end

function default_polyhedra_backend(P::LazySet{N}) where {N}
    if LazySets.dim(P) == 1
        return default_polyhedra_backend_1d(N)
    else
        return default_polyhedra_backend_nd(N)
    end
end

function default_polyhedra_backend_1d(N::Type{<:Number}, solver=nothing)
    return Polyhedra.IntervalLibrary{N}()
end

function default_polyhedra_backend_nd(N::Type{<:Number},
                                      solver=default_lp_solver_polyhedra(N))
    return Polyhedra.DefaultLibrary{N}(solver)
end

function default_lp_solver_polyhedra(N::Type{<:AbstractFloat};
                                     presolve::Bool=false)
    if presolve
        return JuMP.optimizer_with_attributes(GLPK.Optimizer,
                                              "presolve" => GLPK.GLP_ON)
    else
        return JuMP.optimizer_with_attributes(GLPK.Optimizer)
    end
end

function default_lp_solver_polyhedra(N::Type{<:Rational};
                                     presolve::Bool=false)
    if presolve
        return JuMP.optimizer_with_attributes(
            () -> GLPK.Optimizer(method=GLPK.EXACT),
            "presolve" => GLPK.GLP_ON)
    else
        return JuMP.optimizer_with_attributes(
            () -> GLPK.Optimizer(method=GLPK.EXACT))
    end
end

"""
isempty(P::VPolyhedron)
Check whether a polyhedron in vertex/ray representation is empty.
### Input

- `P` -- polyhedron in vertex/ray representation

### Output

`true` if the given polyhedron contains no vertices, and `false` otherwise.

### Algorithm

This algorithm checks whether the `vertices_list` of `P` is empty.
"""
function Base.isempty(P::VPolyhedron)
    return isempty(vertices_list(P))
end

"""
    isbounded(P::AbstractPolytope)

Check whether a polyhedron in vertex/ray representation is bounded.

### Input

- `P` -- polyhedron in vertex/ray representation

### Output

`true` if the given polyhedron contains no rays, and `false` otherwise.

### Algorithm

This algorithm checks whether the `rays_list` of `P` is empty.
"""
function LazySets.isbounded(P::VPolyhedron)
    return isempty(rays_list(P))
end

# VPolyhedron from a VRep
function VPolyhedron(P::VRep{N}) where {N}
    vertices = collect(Polyhedra.points(P))
    rays = map(ray -> ray.a, Polyhedra.allrays(P))
    return VPolyhedron(vertices, rays)
end

"""
    polyhedron(P::VPolyhedron;
               [backend]=default_polyhedra_backend(P),
               [relative_dimension]=nothing)
Return a `VRep` polyhedron from `Polyhedra.jl` given a polyhedron in vertex/ray
representation.
### Input
- `P`       -- polyhedron in vertex/ray representation
- `backend` -- (optional, default: `default_polyhedra_backend(P)`) the
               backend for polyhedral computations; see [Polyhedra's
               documentation](https://juliapolyhedra.github.io/) for further
               information
- `relative_dimension` -- (default, optional: `nothing`) an integer representing
                          the (relative) dimension of the polyhedron; this
                          argument is mandatory if the polyhedron is empty
### Output
A `VRep` polyhedron.
### Notes
The *relative dimension* (or just *dimension*) refers to the dimension of the
set relative to itself, independently of the ambient dimension. For example, a
point has (relative) dimension zero, and a line segment has (relative) dimension
one.
In this library, `LazySets.dim` always returns the ambient dimension of the set,
such that a line segment in two dimensions has dimension two. However,
`Polyhedra.dim` will assign a dimension equal to one to a line segment
because it uses a different convention.
"""
function Polyhedra.polyhedron(P::VPolyhedron;
                    backend=default_polyhedra_backend(P),
                    relative_dimension=nothing)
    require(@__MODULE__, :Polyhedra; fun_name="polyhedron")

    vertices = P.vertices
    rays::Vector{Ray{Float64, Vector{Float64}}} = map(Ray, P.rays)
    if isempty(P)
        if relative_dimension == nothing
            error("the conversion to a `Polyhedra.polyhedron` requires the " *
                "(relative) dimension of the `VPolyhedron` to be known, but it " *
                "cannot be inferred from an empty set; use the keyword " *
                "argument `relative_dimension`")
        end
        Pv = Polyhedra.vrep(vertices, Polyhedra.Line{Float64, Vector{Float64}}[], rays, d=relative_dimension)
    else
        Pv = Polyhedra.vrep(vertices, Polyhedra.Line{Float64, Vector{Float64}}[], rays)
    end
    return Polyhedra.polyhedron(Pv, backend)
end

function LazySets.convex_hull(P1::VPolyhedron, P2::VPolyhedron; backend=nothing, prune::Bool=true)
    vunion = [P1.vertices; P2.vertices]
    runion = [P1.rays; P2.rays]
    P = VPolyhedron(vunion, runion)

    if prune
        P = remove_redundant_vertices(P, backend=backend)
    end

    return P
end

function LazySets.convex_hull(cup::UnionSetArray{N, VPolyhedron{N, VNV, VNR}}; backend=nothing, prune::Bool=true) where {N, VNV, VNR}
    vertices = reduce(vcat, (X.vertices for X in array(cup)))
    rays = reduce(vcat, (X.rays for X in array(cup)))
    P = VPolyhedron(vertices, rays)

    if prune
        P = remove_redundant_vertices(P, backend=backend)
    end

    return P
end