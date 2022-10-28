module BarrierFunctions
    using JuMP

    include("partition.jl")
    export Region, Partition

    include("system.jl")
    export System, autonomous_system
    export state_space_size, safe_set, unsafe_set, initial_set, partitions, noise, next_state

    include("split.jl")
    export grid_split

    include("verify.jl")
    export verify

    include("monte_carlo.jl")
    include("plot.jl")
    export monte_carlo, plot_monte_carlo
end