module BarrierFunctions
    using JuMP, PolyJuMP, SumOfSquares

    import MathOptInterface
    const MOI = MathOptInterface

    using LazySets, ReachabilityBase.Require
    import Base.@kwdef

    using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets
    using LinearAlgebra, Distributions

    include("sets/VPolyhedron.jl")
    include("operations/AffineGenerator.jl")
    include("operations/UncertainAffineMap.jl")

    include("utils/index.jl")
    include("utils/split.jl")
    include("utils/random.jl")
    include("utils/convex_hull.jl")

    include("partition.jl")
    include("system.jl")

    include("utils/plot.jl")

    include("barrier.jl")
    include("expectation.jl")
    include("verify.jl")
    include("monte_carlo.jl")
end