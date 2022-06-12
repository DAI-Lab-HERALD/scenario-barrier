module BarrierFunctions
    using JuMP

    include("partitioning.jl")
    export ConvexPolytope, IndexedConvexPolytope, PolytopePartitioning

    include("system.jl")
    export System, autonomous_system
    export state_space_size, safe_set, unsafe_set, initial_set, partitions, noise, next_state

    include("verify.jl")
    export verify
end