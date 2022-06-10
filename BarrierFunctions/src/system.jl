import Base.@kwdef

using Distributions

@kwdef struct System{T<:Real}
    A::AbstractDict{Int, Matrix{T}}
    b::AbstractDict{Int, Vector{T}}
    Q::PolytopePartitioning{T}

    η::Sampleable

    X0::ConvexPolytope{T}
    Xs::ConvexPolytope{T}
    Xu::Vector{ConvexPolytope{T}}

    function System(A::AbstractDict, b::AbstractDict, Q::PolytopePartitioning, η, X0, Xs, Xu)
        n1, n2, n3 = length(A), length(b), size(Q, 1)
        @assert(n1 == n2 && n2 == n3, "Number of partitions and dynamics matrices must match")

        return new{eltype(X0)}(A, b, Q, η, X0, Xs, Xu)
    end
end

function System(A::Vector, b::Vector, Q::Vector, η, X0, Xs, Xu)
    n1, n2, n3 = size(A, 1), size(b, 1), size(Q, 1)
    @assert(n1 == n2 && n2 == n3, "Number of partitions and dynamics matrices must match")

    index_set = 1:n1
    A = Dict(i => Ai for (i, Ai) in zip(index_set, A))
    b = Dict(i => bi for (i, bi) in zip(index_set, b))
    Q = [IndexedConvexPolytope(i, p) for (i, p) in zip(index_set, Q)]

    return System(A, b, Q, η, X0, Xs, Xu)
end

initial_set(s::System) = s.X0
safe_set(s::System) = s.Xs
unsafe_set(s::System) = s.Xu

partitions(system::System) = system.Q

noise(s::System) = s.η
next_state(s::System, q::IndexedConvexPolytope) = s.A[q.index], s.b[q.index]
