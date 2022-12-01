export grid_split

function grid_split(Q, dyn, num_subregions)
    n = length(Q)
    @assert n == length(dyn) && n == length(num_subregions) "Number of subpartition specifications do no match number of partitions"

    Qnew, dyn_new = [], []

    for (qᵢ, dynᵢ, split_spec) in zip(Q, dyn, num_subregions)
        if LazySets.isbounded(qᵢ)
            rect = box_approximation(qᵢ)
            new_rects = split(rect, split_spec)
            
            for rect in new_rects
                push!(Qnew, rect)
                push!(dyn_new, dynᵢ)
            end
        else
            @assert all(split_spec .== 1) "Cannot split an unbounded region into a grid"

            push!(Qnew, qᵢ)
            push!(dyn_new, dynᵢ)
        end
    end

    return Qnew, dyn_new
end
