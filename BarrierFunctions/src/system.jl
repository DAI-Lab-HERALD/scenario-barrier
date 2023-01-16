using Distributions

export PiecewiseSystem
export dimensionality, safe_set, unsafe_set, initial_set, regions, noise, next_state

abstract type AbstractSystem end

@kwdef struct PiecewiseSystem{N, T<:Real, P<:AbstractPolyhedron{T}} <: AbstractSystem
    Q::AbstractSpatialIndex{T, DynamicsRegion{T}}

    η::Sampleable

    X0::P
    Xs::P
    Xu::AbstractVector{P}

    function PiecewiseSystem{N, T, P}(Q, η, X0, Xs, Xu) where {N, T, P}
        n = length(Q)
        @assert(n >= 1, "The number of regions must be at least 1")
        @assert(X0 ⊆ Xs, "The initial set X0 must be a subset of the safe set Xs")

        return new{N, T, P}(Q, η, X0, Xs, Xu)
    end
end

function PiecewiseSystem(Q, dyn, η, X0::P, Xs::P, Xu::AbstractVector{P}) where {T<:Real, P<:AbstractPolyhedron{T}}
    n1, n2 = length(dyn), size(Q, 1)
    @assert(n1 == n2, "Number of regions and dynamics matrices must match")
    @assert(n1 >= 1, "The number of regions must be at least 1")

    elems = [DynamicsRegion(i, qi, dyni) for (i, (qi, dyni)) in enumerate(zip(Q, dyn))]

    N = LazySets.dim(X0)
    Q = UnboundedIndex{T, N, DynamicsRegion{T}}()
    bulk_load!(Q, elems)

    return PiecewiseSystem{N, T, P}(Q, η, X0, Xs, Xu)
end

dimensionality(::PiecewiseSystem{N, T, P}) where {N, T, P} = N
initial_set(s::PiecewiseSystem) = s.X0
safe_set(s::PiecewiseSystem) = s.Xs
unsafe_set(s::PiecewiseSystem) = s.Xu

regions(system::PiecewiseSystem) = system.Q
noise(s::PiecewiseSystem) = s.η

@kwdef struct PolynomialSystem{N, T<:Real} <: AbstractSystem
    vars::Vector{MultivariatePolynomials.AbstractVariable}
    f::AbstractPolynomialLike{T}

    η::Sampleable

    X0::AbstractSemialgebraicSet
    Xs::AbstractSemialgebraicSet
    Xu::Vector{AbstractSemialgebraicSet}
end

PolynomialSystem{N, T}(f, η, X0, Xs, Xu) where {N, T} = PolynomialSystem{N, T}(variables(f), f, η, X0, Xs, Xu)

dimensionality(::PolynomialSystem{N, T}) where {N, T} = N
initial_set(s::PolynomialSystem) = s.X0
safe_set(s::PolynomialSystem) = s.Xs
unsafe_set(s::PolynomialSystem) = s.Xu

variables(s::PolynomialSystem) = s.vars
dynamics(s::PolynomialSystem) = s.f
noise(s::PolynomialSystem) = s.η