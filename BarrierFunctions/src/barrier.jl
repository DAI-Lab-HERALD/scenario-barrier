@kwdef struct BarrierRegion{I<:Int, P<:AbstractPolyhedron{<:Real}, B}
    index::I
    polytope::P
    fun::B
end
index(b::BarrierRegion) = b.index
region(b::BarrierRegion) = b.polytope

struct Barrier
    regions

    function Barrier(kv)
        new(Dict(kv))
    end
end

function (B::Barrier)(x)
    for (_, Bᵢ) in B.regions
        if x ∈ region(Bᵢ)
            return Bᵢ.fun(x)
        end
    end

    throw("x not in supp(B)")
end

Base.getindex(B::Barrier, key...) = getindex(B.regions, key...)
Base.iterate(B::Barrier) = iterate(B.regions)
Base.iterate(B::Barrier, state) = iterate(B.regions, state)