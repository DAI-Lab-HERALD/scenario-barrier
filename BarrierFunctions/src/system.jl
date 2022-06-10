using Distributions

struct System{T<:Real}
    A::AbstractDict{Int, Matrix{T}}
    Q::PolytopePartitioning{T}

    η::Sampleable

    Xu::ConvexPolytope{T}
    Xi::ConvexPolytope{T}
    Xs::ConvexPolytope{T}

    function System(A, Q, Xu, Xi, Xs)
        n1, n2 = length(A), size(Q, 1)
        @assert(n1 == n2, "Number of partitions and dynamics matrices must match")

        return new(A, Q, Xu, Xi, Xs)
    end
end

function System(A::Vector{Matrix}, Q::Vector{ConvexPolytope}, Xu, Xi, Xs) 
    n1, n2 = size(A, 1), size(Q, 1)
    @assert(n1 == n2, "Number of partitions and dynamics matrices must match")

    index_set = 1:n1
    A = Dict(i => Ai for (i, Ai) in zip(index_set, A))
    Q = [IndexedConvexPolytope(i, p) for (i, p) in zip(index_set, Q)]

    return System(A, Q, Xu, Xi, Xs)
end

unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System) = system.Q

noise(s::System) = s.η
next_state(s::System, q::IndexedConvexPolytope) = s.A[q.index]
