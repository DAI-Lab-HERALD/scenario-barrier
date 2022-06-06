using SemialgebraicSets

struct System{V <: AbstractSystemVariables, D <: AbstractDynamics}
    X::AbstractSemialgebraicSet
    Xu::AbstractVector{<:AbstractSemialgebraicSet}
    Xi::ConvexPolytope
    Xs::Union{<:AbstractSemialgebraicSet, Nothing}

    v::V
    d::D
    p::PolyhedralPartitioning
end

system(fx, x, z, X, Xu, Xi, Xs, p) = System(X, Xu, Xi, Xs, variables(x, z), dynamics(fx), p)

System(X, Xu::AbstractSemialgebraicSet, Xi, Xs, v, d, c, p) = System(X, [Xu], Xi, Xs, v, d, c, p)

state_space(s::System) = s.X
unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System) = system.p

state(s::System) = state(s.v)
noise(s::System) = noise(s.v)
next_state(s::System, p) = next_state(s.d, p)

control(s::System) = control(s.c)
action_space(s::System) = action_space(s.c)
