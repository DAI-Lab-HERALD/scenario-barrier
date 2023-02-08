export UncertainAffineMap

"""
    UncertainAffineMap{N, S<:LazySet{N}, NM, GEN<:AffineGenerator{NM}} <: LazySet{N}
Type that represents an uncertain affine transformation ``M(\\lambda)⋅X ⊕ v`` of a set ``X``, i.e.,
the set
```math
Y = \\{ y ∈ \\mathbb{R}^n : y = M(\\lambda)x + v(\\lambda),\\qquad x ∈ X ,\\qquad M(\\lambda) = \\sum_{i=1}^m M_i \\lambda_i,\\qquad v(\\lambda) = \\sum_{i=1}^m v_i \\lambda_i, \\qquad \\lambda \\in \\mathbb{R}^m_{\\geq 0}, \\qquad \\sum_{i=1}^m \\lambda_i = 1 \\}.
```
If ``X`` is ``n``-dimensional, then ``M(\\lambda)`` should be an ``m × n`` matrix and
``v`` should be an ``m``-dimensional vector.
### Fields
- `generators` -- affine generators for the uncertain affine transformation
- `X` -- set
### Notes
An affine map is the composition of a linear map and a translation. This type is
parametric in the generators, `NM`, which may be different
from the numeric type of the wrapped set, `N`.
An affine map preserves convexity: if `X` is convex, then any affine map of `X`
is convex as well.
"""
struct UncertainAffineMap{N, S<:LazySet{N}, NM, GEN<:AbstractAffineGenerator{NM}} <: LazySet{N}
    generators::Vector{GEN}
    X::S

    # default constructor with dimension-match check
    function AffineMap(generators::Vector{GEN}, X::S) where {N, S<:LazySet{N}, NM, GEN<:AbstractAffineGenerator{NM}}

        @assert length(generators) >= 1 "an uncertain affine map cannot consist of no generators"

        @assert all(g -> dim(X) == input_dim(g), generators) "a generators with input dimension $(input_dim.(generators)) cannot be " *
            "applied to a set of dimension $(dim(X))"

        return new{N, S, NM, GEN}(generators, X)
    end
end

isoperationtype(::Type{<:UncertainAffineMap}) = true

# EmptySet is absorbing for AffineMap
function UncertainAffineMap(generators::Vector, ∅::EmptySet)
    return ∅
end

function generators(am::UncertainAffineMap)
    return am.generators
end

function set(am::UncertainAffineMap)
    return am.X
end

function concretize(am::UncertainAffineMap{N, S, NM, GEN}) where {N, S<:LazySet{N}, NM, GEN<:AbstractAffineGenerator{NM}}
    X = concretize(am.X)
    cha = ConvexHullArray{N, S}(S[affine_map(g, X) for g in generators(am)])

    return concretize(cha)
end

function isboundedtype(::Type{<:UncertainAffineMap{N, S, NM, GEN}}) where {N, S<:LazySet{N}, NM, GEN<:AbstractAffineGenerator{NM}}
    return isboundedtype(S)
end