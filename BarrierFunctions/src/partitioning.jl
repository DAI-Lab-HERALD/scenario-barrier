import Base.@kwdef

using MathOptInterface
const MOI = MathOptInterface

@kwdef struct ConvexPolytope{T<:Real}
    # Convex polytope defined by Hx ≤ h
    H::Matrix{T}
    h::Vector{T}
end

function Base.show(io::IO, p::ConvexPolytope)
    m, n = size(p.H)

    lines = [
        join(["$(H[j, i])x_$i" for i in 1:n], " + ") * " ≤ $(h[j])"
        for j in 1:m
    ]
    print(io, join(lines, "\n"))
end

Base.intersect(p1::ConvexPolytope, p2::ConvexPolytope) = ConvexPolytope(vcat(p1.H, p2.H), vcat(p1.h, p2.h))
Base.size(p::ConvexPolytope) = size(p.H)
Base.size(p::ConvexPolytope, d) = size(p.H, d)

function Base.isempty(p::ConvexPolytope)
    model = Model()

    n = size(p.H, 2)
    @variable(model, x[1:n])
    @constraint(model, p.H * x ≤ p.h)
    # Default objective is FEASIBLE_SENSE
    
    optimize!(model)
    infeasible = termination_status(model) ∈ [MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE]

    return infeasible
end

@kwdef struct IndexedConvexPolytope{I<:Int, T<:Real}
    index::I
    polytope::ConvexPolytope{T}
end

Base.show(io::IO, p::IndexedConvexPolytope) = print(io, p.index)
Base.intersect(p1::IndexedConvexPolytope, p2::IndexedConvexPolytope) = p1.polytope ∩ p2.polytope
Base.intersect(p1::IndexedConvexPolytope, p2::ConvexPolytope) = p1.polytope ∩ p2
Base.intersect(p1::ConvexPolytope, p2::IndexedConvexPolytope) = p2 ∩ p1
Base.size(p::IndexedConvexPolytope) = size(p.polytope)
Base.size(p::IndexedConvexPolytope, d) = size(p.polytope, d)
Base.isempty(p::IndexedConvexPolytope) = isempty(p.polytope)

PolytopePartitioning{T, I} = Vector{IndexedConvexPolytope{T, I}}
