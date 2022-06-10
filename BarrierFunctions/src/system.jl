using Distributions

struct System{T<:Real}
    A::AbstractDict{Int, Matrix{T}}
    Q::PolytopePartitioning{T}

    η::Sampleable

    Xu::ConvexPolytope{T}
    Xi::ConvexPolytope{T}
    Xs::ConvexPolytope{T}
end

System(A, Q, Xu::AbstractSemialgebraicSet, Xi, Xs) = System(A, Q, [Xu], Xi, Xs)

unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System) = system.Q

noise(s::System) = s.η
next_state(s::System, q::IndexedConvexPolytope) = s.A[q.index]
