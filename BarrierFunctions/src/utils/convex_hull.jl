using InvertedIndices, LinearAlgebra
using Polyhedra, CDDLib

using JuMP
# import MathOptInterface
# const MOI = MathOptInterface

### Assume
# - points are stored in a [d x n] Matrix (column major for speed)
# - the convex hull is non-degenerate
function quick_convex_hull(A::AbstractMatrix{T}, solver) where {T <: Real}
    d = size(A, 1)
    if d <= 2
        A = convex_hull!([copy(p) for p in eachcol(A)])
        return reduce(hcat, A)
    end

    initial_points, A = find_initial_points(A)
    
    remaining_points = [initial_points]

    A, halfspace_index = prune_interior(initial_points, A, solver)

    for index in unique(halfspace_index)
        Asubset = A[:, halfspace_index .== index]

        if size(Asubset, 2) >= d + 1  # The amount of points necessary for a d-simplex.
            Asubset = convex_hull!([copy(p) for p in eachcol(Asubset)]; backend=CDDLib.Library())
            Asubset = reduce(hcat, Asubset)
        end
        push!(remaining_points, Asubset)
    end

    A = reduce(hcat, remaining_points)

    A = convex_hull!([copy(p) for p in eachcol(A)]; backend=CDDLib.Library())
    return reduce(hcat, A)
end

function find_initial_points(A::AbstractMatrix{T}) where {T <: Real}
    d = size(A, 1)

    # Find axis polytope and prune
    f_axis = [direction(i, sign) for i in 1:d, sign in [1, -1]][:]
    f_diagonal = [direction(i, j, sign_i, sign_j) for i in 1:d, j in 1:d, sign_i in [1, -1], sign_j in [1, -1] if i < j]
    f_all = vcat(f_axis, f_diagonal)
    initial_idx = [argmax(map(f, eachcol(A))) for f in f_all]
    unique!(initial_idx)

    initial_points, A = A[:, initial_idx], A[:, Not(initial_idx)]

    return initial_points, A
end

direction(i::Int, sign) = x -> sign * x[i]
direction(i::Int, j::Int, sign_i, sign_j) = x -> sign_i * x[i] + sign_j * x[j]

function prune_interior(polygon::AbstractMatrix{T}, A::AbstractMatrix{T}, solver) where {T <: Real}
    n = size(polygon, 1)

    v = polyhedron(vrep([copy(p) for p in eachcol(polygon)]))
    h = hrep(v)

    inellipsoid = inscribed_ellipsoid(h, n, solver)

    keepidx = [!inellipsoid(p) for p in eachcol(A)]
    A = A[:, keepidx]

    halfspace_indices = [halfspace_index(h, p) for p in eachcol(A)]
    keepidx = halfspace_indices .!= 0
    A = A[:, keepidx]
    halfspace_indices = halfspace_indices[keepidx]

    return A, halfspace_indices
end

function halfspace_index(h, p)
    for (i, halfspace) in enumerate(halfspaces(h))
        if !ininterior(p, halfspace)
            return i
        end
    end

    return 0
end

function inscribed_ellipsoid(h, n, solver)
    model = Model(solver)

    @variable(model, B[1:n, 1:n], PSD)
    @variable(model, d[1:n])

    for halfspace in halfspaces(h)
        @constraint(model, [halfspace.β - dot(halfspace.a, d); B * halfspace.a] in SecondOrderCone())
    end

    @variable(model, t)
    upper_tri = [B[i, j] for j in 1:n for i in 1:j]
    @constraint(model, [t; upper_tri] in MOI.RootDetConeTriangle(n))

    @objective(model, Max, t)

    optimize!(model)

    B = value.(B)
    d = value.(d)

    Binv = inv(B)

    inellipsoid(p) = norm(Binv * (p - d)) <= 1

    return inellipsoid
end