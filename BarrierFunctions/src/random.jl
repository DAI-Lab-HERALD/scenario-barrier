using Random, Distributions

export ProbabilitySimplex

struct ProbabilitySimplex <: Sampleable{Multivariate, Continuous}
    dim::Int
end

Base.length(s::ProbabilitySimplex) = s.dim
function Distributions._rand!(rng::AbstractRNG, s::ProbabilitySimplex, x::AbstractVector{T}) where {T<:Real}
    exp_dist = Exponential()
    rand!(rng, exp_dist, x)
    total = sum(x)
    x ./= total
    
    return x
end