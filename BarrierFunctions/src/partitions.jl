import Base.@kwdef

using LazySets

using MathOptInterface
const MOI = MathOptInterface

using MosekTools

const HPoly{N} = Union{HPolytope{N}, HPolyhedron{N}}

Base.size(p::HPoly) = (length ∘ constraints_list)(p), LazySets.dim(p)
function Base.size(p::HPoly, d) 
    @assert(1 ≤ d ≤ 2)

    if d == 1
        return (length ∘ constraints_list)(p)
    else
        return LazySets.dim(p)
    end
end
Base.eltype(p::HPoly) = (eltype ∘ constraints_list)(p)

function iszeromeasure(p::HPoly)
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

    # Otherwise check for infeasiblity
    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => true
    )
    model = Model(solver)
    @variable(model, x[1:n])
    @constraint(model, H * x .≤ h)
    # Default objective is FEASIBLE_SENSE
    
    optimize!(model)
    infeasible = termination_status(model) == MOI.INFEASIBLE

    # For N > 1, it is possible the feasible set is a point without opposite inequalities. 
    # TODO: Check L-inf ball around optimal point is not feasible

    return infeasible
end
iszeromeasure(::EmptySet) = true

PartitionDynamics{T} = Tuple{Matrix{T}, Vector{T}}
@kwdef struct Partition{I<:Int, T<:Real}
    index::I
    polytope::HPolyhedron{T, Vector{T}}
    dynamics::PartitionDynamics{T}
end

Base.hash(p::Partition, h::UInt) = hash(p.index, h)
Base.:(==)(p1::Partition, p2::Partition) = p1.index == p2.index

Base.show(io::IO, p::Partition) = print(io, p.index)
LazySets.intersection(p1::Partition, p2::Partition) = intersection(p1.polytope, p2.polytope)
LazySets.intersection(p1::Partition, p2::AbstractPolyhedron) = intersection(p1.polytope, p2)
LazySets.intersection(p1::AbstractPolyhedron, p2::Partition) = intersection(p2, p1)
Base.size(p::Partition) = size(p.polytope)
Base.size(p::Partition, d) = size(p.polytope, d)
region(p::Partition) = p.polytope
dynamics(p::Partition) = p.dynamics
iszeromeasure(p::Partition) = iszeromeasure(region(p))

PolytopePartitioning{T, I} = Vector{Partition{T, I}}
