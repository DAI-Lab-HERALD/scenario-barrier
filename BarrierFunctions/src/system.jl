import Base.@kwdef

using Distributions

PartitionDynamics{T} = Tuple{Matrix{T}, Vector{T}}

@kwdef struct System{I<:Int, T<:Real, D<:AbstractDict{I, PartitionDynamics{T}}}
    f::D
    Q::PolytopePartitioning{I, T}

    η::Sampleable

    X0::ConvexPolytope{T}
    Xs::ConvexPolytope{T}
    Xu::Vector{ConvexPolytope{T}}

    function System{T, I, D}(f, Q, η, X0, Xs, Xu) where {T, I, D}
        n1, n2 = length(f), size(Q, 1)
        @assert(n1 == n2, "Number of partitions and dynamics matrices must match")
        @assert(n1 >= 1, "The number of partitions must be at least 1")

        return new{T, I, D}(f, Q, η, X0, Xs, Xu)
    end
end

System{T}(f, Q, η, X0, Xs, Xu) where {T} = System{keytype(f), T, valtype(f)}(f, Q, η, X0, Xs, Xu)

function System(A, b, Q, η, X0, Xs, Xu)
    n1, n2, n3 = size(A, 1), size(b, 1), size(Q, 1)
    @assert(n1 == n2 && n2 == n3, "Number of partitions and dynamics matrices must match")
    @assert(n1 >= 1, "The number of partitions must be at least 1")

    index_set = 1:n1
    f = Dict(i => (Ai, bi) for (i, Ai, bi) in zip(index_set, A, b))
    Q = [IndexedConvexPolytope(i, p) for (i, p) in zip(index_set, Q)]

    return System{eltype(A[0])}(f, Q, η, X0, Xs, Xu)
end

initial_set(s::System) = s.X0
safe_set(s::System) = s.Xs
unsafe_set(s::System) = s.Xu

partitions(system::System) = system.Q

noise(s::System) = s.η
next_state(s::System, q::IndexedConvexPolytope) = s.A[q.index], s.b[q.index]
