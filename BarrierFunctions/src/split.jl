export grid_split

function grid_split(Q, A, b, num_subregions)
    n = length(Q)
    @assert n == length(A) && n == length(b) && n == length(num_subregions) "Number of subpartition specifications do no match number of partitions"

    Qnew, Anew, bnew = [], [], []

    for (qᵢ, Aᵢ, bᵢ, split_spec) in zip(Q, A, b, num_subregions)
        if LazySets.isbounded(qᵢ)
            rect = box_approximation(qᵢ)
            new_rects = split(rect, split_spec)
            
            for rect in new_rects
                push!(Qnew, rect)
                push!(Anew, Aᵢ)
                push!(bnew, bᵢ)
            end
        else
            @assert all(split_spec .== 1) "Cannot split an unbounded region into a grid"

            push!(Qnew, qᵢ)
            push!(Anew, Aᵢ)
            push!(bnew, bᵢ)
        end
    end

    return Qnew, Anew, bnew
end
