using Polyhedra
import HiGHS, GLPK, JuMP

export AffineDynamics, ConvexHullDynamics, singleton_dynamics, DynamicsRegion, Partition, is_intersection_zeromeasure, index, region, dynamics

struct AffineDynamics{T}
    A::AbstractMatrix{T}
    b::AbstractVector{T}
end
Base.iterate(d::AffineDynamics, args...) = iterate((d.A, d.b), args...)

const ConvexHullDynamics{T} = AbstractVector{AffineDynamics{T}}
singleton_dynamics(d::AffineDynamics) = [d]
singleton_dynamics(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T} = singleton_dynamics(AffineDynamics(A, b))

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

is_intersection_zeromeasure(p::DynamicsRegion, q::DynamicsRegion; solver=nothing)::Bool = is_intersection_zeromeasure(region(p), region(q); solver=solver)
is_intersection_zeromeasure(p::DynamicsRegion, q::AbstractPolyhedron; solver=nothing)::Bool = is_intersection_zeromeasure(region(p), q; solver=solver)
is_intersection_zeromeasure(p::AbstractPolyhedron, q::DynamicsRegion; solver=nothing)::Bool = is_intersection_zeromeasure(q, p; solver=solver)

function is_intersection_zeromeasure(p::AbstractPolyhedron, q::AbstractPolyhedron; solver=nothing)::Bool
    N = promote_type(eltype(p), eltype(q))

    z = [constraints_list(p); constraints_list(q)]

    A, b = tosimplehrep(z)
    n = size(A, 2)

    if isnothing(solver)
        solver = default_lp_solver(N)
    end

    model = Model(solver)
    set_string_names_on_creation(model, false)

    @variable(model, x[1:n])
    @variable(model, x₀, upper_bound=1.0)

    @objective(model, Max, 1.0 * x₀)

    # Intuitively, if x₀ > 0 then the intersection is full-dimensional.
    @constraint(model, A * x .+ x₀ .≤ b)

    optimize!(model)

    iszeromeasure = objective_value(model) ≤ 0

    return iszeromeasure
end

# default LP solver for floating-point numbers
function default_lp_solver(N::Type{<:AbstractFloat})
    JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
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
