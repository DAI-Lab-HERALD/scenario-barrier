export AbstractAffineGenerator, AffineGenerator


abstract type AbstractAffineGenerator{N} end

"""
AffineGenerator{NM, MAT<:AbstractMatrix{NM}, VN<:AbstractVector{NM}} <: AbstractAffineMap{N, S}
Type that represents an affine transformation ``M⋅X ⊕ v`` of a set ``X``, i.e.,
the set
```math
f(X) = \\{ y ∈ \\mathbb{R}^n : y = Mx + v,\\qquad x ∈ X \\}.
```
### Fields
- `M` -- matrix
- `v` -- translation vector
The fields' getter functions are `matrix`, `set` and `vector`, respectively.
### Notes
An affine map is the composition of a linear map and a translation. This type is
parametric in the coefficients of the linear map, `NM`, which may be different
from the numeric type of the wrapped set, `N`. However, the numeric type of the
translation vector should be `NM`.
An affine map preserves convexity: if `X` is convex, then any affine map of `X`
is convex as well.
```
"""
struct AffineGenerator{N, MAT<:AbstractMatrix{N}, VN<:AbstractVector{N}} <: AbstractAffineGenerator{N}
    M::MAT
    v::VN

    # default constructor with dimension-match check
    function AffineGenerator(M::MAT, v::VN) where {N, MAT<:AbstractMatrix{N}, VN<:AbstractVector{N}}
        @assert size(M, 1) == length(v) "a map with output dimension " *
            "$(size(M, 1)) is incompatible with a translation vector of " *
            "dimension $(length(v))"

        return new{N, MAT, VN}(M, v)
    end
end

# convenience constructor from a UniformScaling
function AffineGenerator(M::UniformScaling, v::AbstractVector)
    return AffineGenerator(M.λ, v)
end

# convenience constructor from a scalar
function AffineGenerator(α::N, v::AbstractVector) where {N<:Real}
    return AffineGenerator(Diagonal(fill(α, length(v))), v)
end

function LazySets.matrix(am::AffineGenerator)
    return am.M
end

function LazySets.vector(am::AffineGenerator)
    return am.v
end

function LazySets.affine_map(am::AffineGenerator, X)
    return affine_map(matrix(am), X, vector(am))
end