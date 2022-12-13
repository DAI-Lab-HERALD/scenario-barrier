using Polyhedra
import GLPK, JuMP

export ConvexHullDynamics, singleton_dynamics, DynamicsRegion, Partition, is_intersection_zeromeasure, index, region, dynamics

# TYPE PIRACY! - the better approach is probably to use dim and let DynamicsRegion derive from LazySet
Base.size(p::AbstractPolyhedron) = (length ∘ constraints_list)(p), LazySets.dim(p)
function Base.size(p::AbstractPolyhedron, d) 
    @assert(1 ≤ d ≤ 2)

    if d == 1
        return (length ∘ constraints_list)(p)
    else
        return LazySets.dim(p)
    end
end


const ConvexHullDynamics{T} = Vector{Tuple{Matrix{T}, Vector{T}}}
singleton_dynamics(d) = [d]

# Dynamics
@kwdef struct DynamicsRegion{T<:Real}
    index::Int
    polytope::AbstractPolyhedron{T}
    # Convex hull of uncertain dynamics
    conv_dynamics::ConvexHullDynamics{T}

    mbr::Union{AbstractHyperrectangle{T}, Nothing}
end
DynamicsRegion(index, polytope, dynamics) = DynamicsRegion(index, polytope, dynamics, LazySets.isbounded(polytope) ? box_approximation(polytope) : nothing)
Spatial.has_mbr(region::DynamicsRegion) = !isnothing(region.mbr)
Spatial.mbr(region::DynamicsRegion) = region.mbr
Spatial.region(region::DynamicsRegion) = region.polytope

Base.hash(p::DynamicsRegion, h::UInt) = hash(p.index, h)
Base.:(==)(p1::DynamicsRegion, p2::DynamicsRegion) = p1.index == p2.index

is_intersection_zeromeasure(p::DynamicsRegion, q::LazySet; kwargs...) = is_intersection_zeromeasure(region(p), q; kwargs...)
is_intersection_zeromeasure(p::LazySet, q::DynamicsRegion; kwargs...) = is_intersection_zeromeasure(q, p; kwargs...)

function is_intersection_zeromeasure(p::LazySet{N}, q::LazySet{N}, solver=nothing) where {N}
    z = HPolyhedron([constraints_list(p); constraints_list(q)])

    A, b = tosimplehrep(z)
    n = size(A, 2)

    if isnothing(solver)
        solver = default_lp_solver(N)
    end

    model = Model(solver)
    @variable(model, x[i=1:n])
    @variable(model, x₀, upper_bound=1.0)

    @objective(model, Max, x₀)

    # Intuitively, if x₀ > 0 then the intersection is full-dimensional.
    @constraint(model, A * x .+ x₀ .≤ b)

    optimize!(model)

    iszeromeasure = objective_value(model) ≤ 0

    return iszeromeasure
end

# default LP solver for floating-point numbers
function default_lp_solver(N::Type{<:AbstractFloat})
    JuMP.optimizer_with_attributes(() -> GLPK.Optimizer(method=GLPK.SIMPLEX))
end

# default LP solver for rational numbers
function default_lp_solver(N::Type{<:Rational})
    JuMP.optimizer_with_attributes(() -> GLPK.Optimizer(method=GLPK.EXACT))
end

Base.show(io::IO, p::DynamicsRegion) = print(io, p.index)
LazySets.tosimplehrep(p::DynamicsRegion) = tosimplehrep(region(p))
index(p::DynamicsRegion) = p.index
region(p::DynamicsRegion) = p.polytope
dynamics(p::DynamicsRegion) = p.conv_dynamics

Partition{T} = Vector{DynamicsRegion{T}}
