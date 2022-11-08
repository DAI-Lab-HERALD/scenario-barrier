using Spatial

struct UnboundedIndex{T, D, E} <: AbstractSpatialIndex{T, E}
    bounded_index    # Should be efficient
    unbounded_index  # Should accomodate unbounded regions (possibly inefficient)

    function UnboundedIndex{T, D, E}() where {T, D, E}
        return new(RTreeIndex{T, D, E}(), LinearIndex{T, E}())
    end
end

Base.isempty(index::UnboundedIndex) = isempty(index.bounded_index) && isempty(index.unbounded_index)
Base.length(index::UnboundedIndex) = length(index.bounded_index) + length(index.unbounded_index)
Base.eltype(::UnboundedIndex{T, E}) where {T, E} = E

struct UnboundedIndexIteratorState
    iterate_bounded
    substate
end

function Base.iterate(index::UnboundedIndex)
    if !isempty(index.bounded_index)
        elem, state = iterate(index.bounded_index)
        return elem, UnboundedIndexIteratorState(true, state)
    end

    it = iterate(index.unbounded_index)
    if isnothing(it)
        return nothing
    end

    elem, state = it
    return elem, UnboundedIndexIteratorState(false, state)
end

function Base.iterate(index::UnboundedIndex, state)
    if state.iterate_bounded
        it = iterate(index.bounded_index, state.substate)

        if !isnothing(it)
            elem, state = it
            return elem, UnboundedIndexIteratorState(true, state)
        end
        
        it = iterate(index.unbounded_index)
    else
        it = iterate(index.unbounded_index, state.substate)
    end

    if !isnothing(it)
        elem, state = it
        return elem, UnboundedIndexIteratorState(false, state)
    end

    return nothing
end

function Spatial.findfirst(query, index::UnboundedIndex)
    res = Spatial.findfirst(query, index.bounded_index)
    if isnothing(res)
        res = Spatial.findfirst(query, index.unbounded_index)
    end

    return res
end

function Spatial.findall(query, index::UnboundedIndex)
    bounded_res = Spatial.findall(query, index.bounded_index)
    unbounded_res = Spatial.findfirst(query, index.unbounded_index)

    append!(bounded_res, unbounded_res)

    return bounded_res
end

function Base.insert!(index::UnboundedIndex, elem)
    if has_mbr(elem)
        insert!(index.bounded_index, elem)
    else
        insert!(index.unbounded_index, elem)
    end
end

function Base.delete!(query, index::UnboundedIndex)
    delete!(query, index.bounded_index)
    delete!(query, index.unbounded_index)
end
    
function Spatial.bulk_load!(index::UnboundedIndex, data)
    mask = has_mbr.(data)
    bounded = data[mask]
    unbounded = data[.!mask]  # Elementwise negation of mask

    bulk_load!(index.bounded_index, bounded)
    bulk_load!(index.unbounded_index, unbounded)
end