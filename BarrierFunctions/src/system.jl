using SemialgebraicSets

struct System{V <: AbstractSystemVariables, D <: AbstractDynamics}
    Xu::AbstractVector{<:AbstractSemialgebraicSet}
    Xi::ConvexPolytope
    Xs::Union{<:AbstractSemialgebraicSet, Nothing}

    v::V
    d::D
    Q::PolytopePartitioning
end

system(fx, x, z, Xu, Xi, Xs, Q) = System(Xu, Xi, Xs, variables(x, z), dynamics(fx), Q)

System(Xu::AbstractSemialgebraicSet, Xi, Xs, v, d, c, p) = System([Xu], Xi, Xs, v, d, c, Q)

unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System) = system.Q

state(s::System) = state(s.v)
noise(s::System) = noise(s.v)
next_state(s::System, q) = next_state(s.d, q)
