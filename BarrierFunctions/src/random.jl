using Random, Distributions

export UnitSimplex

struct UnitSimplex <: Sampleable{Multivariate, Continuous}
    dim::Int
end

Base.length(s::UnitSimplex) = s.dim
function Distributions._rand!(rng::AbstractRNG, s::UnitSimplex, x::AbstractVector{T}) where {T<:Real}
    exp_dist = Exponential()
    rand!(rng, exp_dist, x)
    total = sum(x)
    x ./= total
    
    return x
end