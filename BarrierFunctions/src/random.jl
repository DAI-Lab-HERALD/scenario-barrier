using Random, Distributions

struct UnitSimplex <: Sampleable{Multivariate, Continuous}
    dim::Int
end

Base.length(s::UnitSimplex) = s.dim
function Distributions._rand!(rng::AbstractRNG, s::UnitSimplex, x::AbstractVector{T}) where {T<:Real}
    exp_dist = Exponential()
    y = rand(rng, exp_dist, length(s))
    y = [0; cumsum(y) ./ sum(y)]
    y = y[2:end] - y[1:end-1]

    shuffle!(rng, y)
    copyto!(x, y)
    
    return x
end