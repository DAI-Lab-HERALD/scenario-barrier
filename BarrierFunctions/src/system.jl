import Base.@kwdef

using LazySets

using Distributions

# TODO: We could do it more generically with <:LazySet{T} for each region field
@kwdef struct System{N, I<:Int, T<:Real, P<:AbstractPolyhedron{T}, VP<:AbstractVector{P}}
    Q::Partition{I, T, P}

    η::Sampleable

    X0::P
    Xs::P
    Xu::VP

    function System{N, I, T, P, VP}(Q, η, X0, Xs, Xu) where {N, I, T, P, VP}
        n = size(Q, 1)
        @assert(n >= 1, "The number of partitions must be at least 1")
        @assert(X0 ⊆ Xs, "The initial set X0 must be a subset of the safe set Xs")

        return new{N, I, T, P, VP}(Q, η, X0, Xs, Xu)
    end
end

function System(Q, A, b, η, X0::P, Xs::P, Xu::VP) where {T<:Real, P<:AbstractPolyhedron{T}, VP<:AbstractVector{P}}
    n1, n2, n3 = size(A, 1), size(b, 1), size(Q, 1)
    @assert(n1 == n2 && n2 == n3, "Number of partitions and dynamics matrices must match")
    @assert(n1 >= 1, "The number of partitions must be at least 1")

    Q = [Region(i, qi, (Ai, bi)) for (i, (qi, Ai, bi)) in enumerate(zip(Q, A, b))]

    return System{LazySets.dim(X0), typeof(Q[1].index), T, P, VP}(Q, η, X0, Xs, Xu)
end


state_space_size(::System{N, I, T, P, VP}) where {N, I, T, P, VP} = N
initial_set(s::System) = s.X0
safe_set(s::System) = s.Xs
unsafe_set(s::System) = s.Xu

partitions(system::System) = system.Q

noise(s::System) = s.η
