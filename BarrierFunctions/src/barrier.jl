export BarrierFunction, ConstantBarrier, LinearBarrier, BarrierRegion, PiecewiseBarrier, PolynomialBarrier
export BarrierTemplate, ConstantTemplate, LinearTemplate, PolynomialTemplate, PiecewiseTemplate

abstract type BarrierTemplate end

struct ConstantTemplate <: BarrierTemplate end
struct LinearTemplate <: BarrierTemplate end
struct PolynomialTemplate <: BarrierTemplate
    degree::Int
end
struct PiecewiseTemplate <: BarrierTemplate
    underlying_template::BarrierTemplate
end

abstract type BarrierFunction end

# Simple barriers
struct LinearBarrier{T, VT<:AbstractVector{T}} <: BarrierFunction
    p::VT
    v::T
end
function (B::LinearBarrier)(x)
    return B.p' * x + B.v
end

struct ConstantBarrier{T} <: BarrierFunction
    v::T
end
function (B::ConstantBarrier)(x)
    return B.v
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
struct BarrierRegion{B <: BarrierFunction, I <: Integer}
    index::I
    polytope::AbstractPolyhedron
    fun::B
end
index(b::BarrierRegion) = b.index
region(b::BarrierRegion) = b.polytope
Base.print(io::IO, b::BarrierRegion) = print(io, "$(index(b))")

function (B::BarrierRegion)(x)
    @assert x ∈ region(B)

    return B.fun(x)
end

struct PiecewiseBarrier{I <: Integer, R <: BarrierRegion} <: BarrierFunction
    regions::Dict{I, R}
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
Base.length(B::PiecewiseBarrier) = length(B.regions)