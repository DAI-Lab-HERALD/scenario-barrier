import Base.@kwdef

using Distributions

PartitionDynamics{T} = Tuple{Matrix{T}, Vector{T}}

@kwdef struct System{N, I<:Int, T<:Real}
    f::AbstractDict{I, PartitionDynamics{T}}
    Q::PolytopePartitioning{I, T}

    η::Sampleable

    X0::ConvexPolytope{T}
    Xs::ConvexPolytope{T}
    Xu::Vector{ConvexPolytope{T}}

    function System{N, T, I}(f, Q, η, X0, Xs, Xu) where {N, T, I}
        n1, n2 = length(f), size(Q, 1)
        @assert(n1 == n2, "Number of partitions and dynamics matrices must match")
        # @assert(n1 >= 1, "The number of partitions must be at least 1")

        return new{N, T, I}(f, Q, η, X0, Xs, Xu)
    end
end

function System{N}(A, b, Q, η, X0, Xs, Xu) where {N}
    n1, n2, n3 = size(A, 1), size(b, 1), size(Q, 1)
    @assert(n1 == n2 && n2 == n3, "Number of partitions and dynamics matrices must match")
    #@assert(n1 >= 1, "The number of partitions must be at least 1")

    index_set = 1:n1
    f = Dict(i => (Ai, bi) for (i, Ai, bi) in zip(index_set, A, b))
    Q = [IndexedConvexPolytope(i, p) for (i, p) in zip(index_set, Q)]

    return System{N, keytype(f), eltype(X0)}(f, Q, η, X0, Xs, Xu)
end


state_space_size(::System{N, I, T}) where {N, I, T} = N
initial_set(s::System) = s.X0
safe_set(s::System) = s.Xs
unsafe_set(s::System) = s.Xu

partitions(system::System) = system.Q

noise(s::System) = s.η
next_state(s::System, q::IndexedConvexPolytope) = s.f[q.index]
