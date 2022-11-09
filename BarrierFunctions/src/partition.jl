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
@kwdef struct DynamicsRegion{I<:Int, T<:Real, P<:AbstractPolyhedron{T}}
    index::I
    polytope::P
    dynamics::Tuple{Matrix{T}, Vector{T}}
end

Base.hash(p::DynamicsRegion, h::UInt) = hash(p.index, h)
Base.:(==)(p1::DynamicsRegion, p2::DynamicsRegion) = p1.index == p2.index

is_intersection_zeromeasure(p::DynamicsRegion, q::LazySet) = is_intersection_zeromeasure(region(p), q)
is_intersection_zeromeasure(p::LazySet, q::DynamicsRegion) = is_intersection_zeromeasure(q, p)

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

Base.show(io::IO, p::DynamicsRegion) = print(io, p.index)
LazySets.tosimplehrep(p::DynamicsRegion) = tosimplehrep(region(p))
index(p::DynamicsRegion) = p.index
region(p::DynamicsRegion) = p.polytope
dynamics(p::DynamicsRegion) = p.dynamics

Partition{I, T, P} = Vector{DynamicsRegion{I, T, P}}
