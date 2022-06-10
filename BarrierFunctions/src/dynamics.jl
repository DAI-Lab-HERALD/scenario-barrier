
abstract type AbstractDynamics end

struct VectorDynamics{V} <: AbstractDynamics
    fx::AbstractVector{V}
end

dynamics(fx::AbstractPolynomialLike) = dynamics([fx])
dynamics(fx::AbstractVector{V}) where {V} = VectorDynamics(fx)

next_state(d::VectorDynamics{V}, q) where {V} = d.fx