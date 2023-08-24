export BarrierFunction, ConstantBarrier, LinearBarrier, QuadraticBarrier, BarrierRegion, PiecewiseBarrier, PolynomialBarrier


abstract type BarrierFunction end

# Simple barriers
struct LinearBarrier{T, VT<:AbstractVector{T}} <: BarrierFunction
    p::VT
    v::T
end
function (B::LinearBarrier)(x)
    return B.p' * x + B.v
end

struct PolynomialBarrier{T} <: BarrierFunction
    p::AbstractPolynomialLike{T}
    # Store the variables separately, as the order is important
    vars::Vector{<:MultivariatePolynomials.AbstractVariable}
end
function (B::PolynomialBarrier)(x)
    return B.p(x)
end

variables(B::PolynomialBarrier) = B.vars

# Piece-wise barrier
struct BarrierRegion{T<:Real, B<:BarrierFunction}
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

(B::PiecewiseBarrier)(x::T) where {T<:Real} = B([x])

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
