using Distributions

export System, autonomous_system
export state_space_size, safe_set, unsafe_set, initial_set, regions, noise, next_state

# TODO: We could do it more generically with <:LazySet{T} for each region field
@kwdef struct System{N, T<:Real, P<:AbstractPolyhedron{T}}
    Q::AbstractSpatialIndex{T, DynamicsRegion{T}}

    η::Sampleable

    X0::P
    Xs::P
    Xu::AbstractVector{P}

    function System{N, T, P}(Q, η, X0, Xs, Xu) where {N, T, P}
        n = length(Q)
        @assert(n >= 1, "The number of regions must be at least 1")
        @assert(X0 ⊆ Xs, "The initial set X0 must be a subset of the safe set Xs")

        return new{N, T, P}(Q, η, X0, Xs, Xu)
    end
end

function System(Q, dyn, η, X0::P, Xs::P, Xu::AbstractVector{P}) where {T<:Real, P<:AbstractPolyhedron{T}}
    n1, n2 = length(dyn), size(Q, 1)
    @assert(n1 == n2, "Number of regions and dynamics matrices must match")
    @assert(n1 >= 1, "The number of regions must be at least 1")

    elems = [DynamicsRegion(i, qi, dyni) for (i, (qi, dyni)) in enumerate(zip(Q, dyn))]

    N = LazySets.dim(X0)
    Q = UnboundedIndex{T, N, DynamicsRegion{T}}()
    bulk_load!(Q, elems)

    return System{N, T, P}(Q, η, X0, Xs, Xu)
end


state_space_size(::System{N, T, P}) where {N, T, P} = N
initial_set(s::System) = s.X0
safe_set(s::System) = s.Xs
unsafe_set(s::System) = s.Xu

regions(system::System) = system.Q

noise(s::System) = s.η
