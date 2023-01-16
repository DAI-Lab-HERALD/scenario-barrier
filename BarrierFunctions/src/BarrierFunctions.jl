module BarrierFunctions
    using JuMP, PolyJuMP, SumOfSquares

    import MathOptInterface
    const MOI = MathOptInterface

    using LazySets
    import Base.@kwdef

    using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

    include("utils/index.jl")
    include("utils/split.jl")
    include("utils/random.jl")

    include("partition.jl")
    include("system.jl")

    include("utils/plot.jl")

    include("barrier.jl")
    include("expectation.jl")
    include("verify.jl")
    include("monte_carlo.jl")
end