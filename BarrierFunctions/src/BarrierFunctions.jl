module BarrierFunctions
    using JuMP

    import MathOptInterface
    const MOI = MathOptInterface

    using LazySets
    import Base.@kwdef

    include("utils/index.jl")
    include("utils/split.jl")
    include("utils/random.jl")

    include("partition.jl")
    include("system.jl")

    include("utils/plot.jl")

    include("barrier.jl")
    include("split.jl")
    include("expectation.jl")
    include("verify.jl")
    include("random.jl")
    include("monte_carlo.jl")
    include("plot.jl")
end