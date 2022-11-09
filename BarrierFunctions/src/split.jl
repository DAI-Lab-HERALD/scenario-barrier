export grid_split

function grid_split(system::System{N, T, P}, num_subregions) where {N, T, P}
    Q = regions(system)
    n, m = length(Q), length(num_subregions)
    @assert(n == m, "Number of subpartition specifications do no match number of partitions")

    next_partition_index = maximum(q.index for q in Q) + 1

    Qnew = DynamicsRegion{T}[]
    for (q, split_spec) in zip(Q, num_subregions)
        if (LazySets.isbounded ∘ region)(q)
            rect = box_approximation(region(q))
            new_rects = split(rect, split_spec)
            
            for rect in new_rects
                push!(Qnew, DynamicsRegion(next_partition_index, convert(P, rect), dynamics(q)))
                next_partition_index += 1
            end
        else
            push!(Qnew, q)
        end
    end

    Qnew = LinearIndex{T}(Qnew)

    return System{N, T, P}(Qnew, noise(system), initial_set(system), safe_set(system), unsafe_set(system))
end
