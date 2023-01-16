export BarrierFunction, ConstantBarrier, LinearBarrier, QuadraticBarrier, BarrierRegion, PiecewiseBarrier, PolynomialBarrier


abstract type BarrierFunction end

# Simple barriers
@kwdef struct ConstantBarrier{T} <: BarrierFunction
    v::T
end
function (B::ConstantBarrier)(x)
    return B.v
end

@kwdef struct LinearBarrier{T, VT<:AbstractVector{T}} <: BarrierFunction
    p::VT
    v::T
end
function (B::LinearBarrier)(x)
    return B.p' * x + B.v
end

@kwdef struct QuadraticBarrier{T, VT<:AbstractVector{T}, VVT<:AbstractMatrix{T}} <: BarrierFunction
    P::VVT
    p::VT
    v::T
end
function (B::QuadraticBarrier)(x)
    return x' * B.P * x + B.p' * x + B.v
end
matrix(B::QuadraticBarrier) = vcat(hcat(B.P, B.p), hcat(B.p', B.v))

@kwdef struct PolynomialBarrier{T} <: BarrierFunction
    p::AbstractPolynomialLike{T}
end
function (B::PolynomialBarrier)(x)
    return B.p(x)
end

# Piece-wise barrier
@kwdef struct BarrierRegion{T<:Real, B<:BarrierFunction}
    index::Int
    polytope::AbstractPolyhedron{T}
    fun::B
end
index(b::BarrierRegion) = b.index
region(b::BarrierRegion) = b.polytope

function (B::BarrierRegion)(x)
    @assert x ∈ region(B)

    return B.fun(x)
end

struct PiecewiseBarrier <: BarrierFunction
    regions

    function PiecewiseBarrier(kv)
        new(Dict(kv))
    end
end

function (B::PiecewiseBarrier)(x::T) where {T<:Real}
    for (_, Bᵢ) in B.regions
        if [x] ∈ region(Bᵢ)
            return Bᵢ.fun([x])
        end
    end

    error("x not in supp(B)")
end

function (B::PiecewiseBarrier)(x::VT) where {VT<:AbstractVector{<:Real}}
    for (_, Bᵢ) in B.regions
        if x ∈ region(Bᵢ)
            return Bᵢ.fun(x)
        end
    end

    error("x not in supp(B)")
end

Base.getindex(B::PiecewiseBarrier, key...) = getindex(B.regions, key...)
function Base.iterate(B::PiecewiseBarrier) 
    it = iterate(B.regions)
    if isnothing(it)
        return nothing
    end
    (_, region), state = it
    return region, state
end
function Base.iterate(B::PiecewiseBarrier, state)
    it = iterate(B.regions, state)
    if isnothing(it)
        return nothing
    end
    (_, region), state = it
    return region, state
end
