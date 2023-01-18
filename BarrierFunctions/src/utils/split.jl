export grid_split

function grid_split(Q, dyn, num_subregions)
    n = length(Q)
    @assert n == length(dyn) && n == size(num_subregions, 2) "Number of subpartition specifications do no match number of partitions"

    Qnew, dyn_new = [], []

    for (qᵢ, dynᵢ, split_spec) in zip(Q, dyn, eachcol(num_subregions))
        for rect in split_region(qᵢ, split_spec)
            push!(Qnew, rect)
            push!(dyn_new, dynᵢ)
        end
    end

    return Qnew, dyn_new
end

function split_region(q::AbstractHyperrectangle{N}, split_spec) where {N}
    new_rects = split(q, split_spec)

    return new_rects
end

function split_region(q, split_spec)
    @assert all(split_spec .== 1) "Cannot split an unbounded region into a grid"

    return [q]
end