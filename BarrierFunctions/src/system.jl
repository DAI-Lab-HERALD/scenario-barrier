using SemialgebraicSets

struct System{V <: AbstractSystemVariables, D <: AbstractDynamics, P <: AbstractPartitioning}
    X::AbstractSemialgebraicSet
    Xu::AbstractVector{<:AbstractSemialgebraicSet}
    Xi::AbstractVector{<:AbstractSemialgebraicSet}
    Xs::Union{<:AbstractSemialgebraicSet, Nothing}

    v::V
    d::D
    p::P
end

system(fx, x, z, X, Xu, Xi, Xs, p) = System(X, Xu, Xi, Xs, variables(x, z), dynamics(fx), p)

System(X, Xu::AbstractSemialgebraicSet, Xi, Xs, v, d, c, p) = System(X, [Xu], Xi, Xs, v, d, c, p)
System(X, Xu, Xi::AbstractSemialgebraicSet, Xs, v, d, c, p) = System(X, Xu, [Xi], Xs, v, d, c, p)
System(X, Xu::AbstractSemialgebraicSet, Xi::AbstractSemialgebraicSet, Xs, v, d, p) = System(X, [Xu], [Xi], Xs, v, d, p)

state_space(s::System) = s.X
unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System{V, D, <:AbstractPartitioning}) where {V, D} = partition_iterator(system.p)

state(s::System) = state(s.v)
noise(s::System) = noise(s.v)
next_state(s::System, p) = next_state(s.d, p)

control(s::System) = control(s.c)
action_space(s::System) = action_space(s.c)
