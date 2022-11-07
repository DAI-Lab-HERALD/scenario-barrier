import Base.@kwdef

using LazySets

using MathOptInterface
const MOI = MathOptInterface

using MosekTools

export Region, Partition

# TYPE PIRACY! - the better approach is probably to use dim and let Region derive from LazySet
Base.size(p::AbstractPolyhedron) = (length ∘ constraints_list)(p), LazySets.dim(p)
function Base.size(p::AbstractPolyhedron, d) 
    @assert(1 ≤ d ≤ 2)

    if d == 1
        return (length ∘ constraints_list)(p)
    else
        return LazySets.dim(p)
    end
end

# Dynamics
RegionDynamics{T} = Tuple{Matrix{T}, Vector{T}}
@kwdef struct Region{I<:Int, T<:Real, P<:AbstractPolyhedron{T}}
    index::I
    polytope::P
    dynamics::RegionDynamics{T}
end

Base.hash(p::Region, h::UInt) = hash(p.index, h)
Base.:(==)(p1::Region, p2::Region) = p1.index == p2.index

is_intersection_zeromeasure(p::Region, q::LazySet) = is_intersection_zeromeasure(region(p), q)
is_intersection_zeromeasure(p::LazySet, q::Region) = is_intersection_zeromeasure(q, p)

is_intersection_zeromeasure(p::LazySet, q::LazySet) = isempty(Intersection(p, q)) || ishyperplane(p, q)
ishyperplane(p, q) = ishyperplane(HPolyhedron([constraints_list(p); constraints_list(q)]))
function ishyperplane(p::AbstractPolyhedron)
    m, n = size(p)
    H, h = tosimplehrep(p)

    # If two constraints are opposite inequalities
    isapproxzero(x) = isapprox(x, 0.0)
    for i in 1:m
        for j in 1:i-1
            if all(isapproxzero.(H[i] + H[j])) && isapproxzero(h[i] + h[j])
                return true
            end
        end
    end

    # For N > 1, it is possible the feasible set is a point without opposite inequalities. 
    # TODO: Check L-inf ball around optimal point is not feasible
    return false
end

Base.show(io::IO, p::Region) = print(io, p.index)
LazySets.intersection(p1::Region, p2::Region) = intersection(p1.polytope, p2.polytope)
LazySets.intersection(p1::Region, p2::AbstractPolyhedron) = intersection(p1.polytope, p2)
LazySets.intersection(p1::AbstractPolyhedron, p2::Region) = intersection(p2, p1)
LazySets.tosimplehrep(p::Region) = tosimplehrep(region(p))
Base.size(p::Region) = size(p.polytope)
Base.size(p::Region, d) = size(p.polytope, d)
region(p::Region) = p.polytope
dynamics(p::Region) = p.dynamics
iszeromeasure(p::Region) = iszeromeasure(region(p))

Partition{I, T, P} = Vector{Region{I, T, P}}
