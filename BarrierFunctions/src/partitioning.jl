import Base.@kwdef

using MathOptInterface
const MOI = MathOptInterface

using MosekTools

@kwdef struct ConvexPolytope{T<:Real}
    # Convex polytope defined by Hx ≤ h
    H::Matrix{T}
    h::Vector{T}
end

function Base.show(io::IO, p::ConvexPolytope)
    m, n = size(p.H)

    lines = [
        join(["$(H[j, i])x_$i" for i in 1:n], " + ") * " ≤ $(h[j])"
        for j in 1:m
    ]
    print(io, join(lines, "\n"))
end

Base.intersect(p1::ConvexPolytope, p2::ConvexPolytope) = ConvexPolytope(vcat(p1.H, p2.H), vcat(p1.h, p2.h))
Base.size(p::ConvexPolytope) = size(p.H)
Base.size(p::ConvexPolytope, d) = size(p.H, d)
Base.eltype(p::ConvexPolytope) = eltype(p.H)

function iszeromeasure(p::ConvexPolytope)
    m, n = size(p.H)

    # If two constraints are opposite inequalities
    isapproxzero(x) = isapprox(x, 0.0)
    for i in 1:m
        for j in 1:i-1
            if all(isapproxzero.(p.H[i] + p.H[j])) && isapproxzero(p.h[i] + p.h[j])
                return true
            end
        end
    end

    # Otherwise check for infeasiblity
    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => true
    )
    model = Model(solver)
    @variable(model, x[1:n])
    @constraint(model, p.H * x .≤ p.h)
    # Default objective is FEASIBLE_SENSE
    
    optimize!(model)
    infeasible = termination_status(model) == MOI.INFEASIBLE

    return infeasible
end

function Base.issubset(l::ConvexPolytope, r::ConvexPolytope) 
    m, n = size(l.H)

    for i in 1:m
        # Otherwise check for infeasiblity
        solver = optimizer_with_attributes(Mosek.Optimizer, 
            "QUIET" => true
        )
        model = Model(solver)
        @variable(model, x[1:n])
        @constraint(model, r.H * x .≤ r.h)
        @objective(model, Min, sum(l.H[i] .* x) - l.h[i])
        
        optimize!(model)

        # For an LP, the primal is unbounded iff the dual is infeasible
        unbounded = termination_status(model) == MOI.DUAL_INFEASIBLE
        if unbounded
            return true
        else
            value = objective_value(model)
            if value < 0.0
                return false
            end
        end
    end

    return true
end

PartitionDynamics{T} = Tuple{Matrix{T}, Vector{T}}
@kwdef struct Partition{I<:Int, T<:Real}
    index::I
    polytope::ConvexPolytope{T}
    dynamics::PartitionDynamics{T}
end

Base.hash(p::Partition, h::UInt) = hash(p.index, h)
Base.:(==)(p1::Partition, p2::Partition) = p1.index == p2.index

Base.show(io::IO, p::Partition) = print(io, p.index)
Base.intersect(p1::Partition, p2::Partition) = p1.polytope ∩ p2.polytope
Base.intersect(p1::Partition, p2::ConvexPolytope) = p1.polytope ∩ p2
Base.intersect(p1::ConvexPolytope, p2::Partition) = p2 ∩ p1
Base.size(p::Partition) = size(p.polytope)
Base.size(p::Partition, d) = size(p.polytope, d)
region(p::Partition) = p.polytope
dynamics(p::Partition) = p.dynamics
iszeromeasure(p::Partition) = iszeromeasure(region(p))

PolytopePartitioning{T, I} = Vector{Partition{T, I}}
