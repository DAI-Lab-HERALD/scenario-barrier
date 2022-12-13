module BarrierFunctions
    using JuMP

    import MathOptInterface
    const MOI = MathOptInterface

    using LazySets
    import Base.@kwdef

    include("index.jl")
    include("partition.jl")
    include("system.jl")
    include("barrier.jl")
    include("split.jl")
    include("verify.jl")
    include("random.jl")
    include("monte_carlo.jl")
    include("plot.jl")
end