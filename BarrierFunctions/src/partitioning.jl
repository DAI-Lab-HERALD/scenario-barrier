struct ConvexPolytope{T<:Real}
    # Convex polytope defined by Hx ≤ h
    H::Matrix{T}
    h::Vector{T}
end

function Base.show(io::IO, p::ConvexPolytope)
    m, n = size(p.H)

    lines = [
        join(["$(H[j, i])x_$i", " + " for i in 1:n]) * " ≤ $(h[j])"
        for j in 1:m
    ]
    print(io, join(lines, "\n"))
end

Base.intersect(p1::ConvexPolytope, p2::ConvexPolytope) = ConvexPolytope(vcat(p1.H, p2.H), vcat(p1.h, p2.h))
Base.size(p::ConvexPolytope) = size(p.H)
Base.size(p::ConvexPolytope, d) = size(p.H, d)

function Base.isempty(p::ConvexPolytope)
    #TODO: make LP and test infeasibility
    return false
end

struct IndexedConvexPolytope{T<:Real}
    i::Int
    p::ConvexPolytope{T}
end

Base.show(io::IO, p::IndexedConvexPolytope) = print(io, p.i)
Base.intersect(p1::IndexedConvexPolytope, p2::IndexedConvexPolytope) = p1.p ∩ p2.p
Base.intersect(p1::IndexedConvexPolytope, p2::ConvexPolytope) = p1.p ∩ p2
Base.intersect(p1::ConvexPolytope, p2::IndexedConvexPolytope) = p2 ∩ p1
Base.size(p::IndexedConvexPolytope) = size(p.p)
Base.size(p::IndexedConvexPolytope, d) = size(p.p, d)

PolyhedralPartitioning = Vector{IndexedConvexPolytope}
