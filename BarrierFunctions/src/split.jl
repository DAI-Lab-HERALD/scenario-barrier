using LazySets

function grid_split(system::System{N, I, T, P, VP}, num_subpartitions) where {N, I, T, P, VP}
    Q = partitions(system)
    n, m = length(Q), length(num_subpartitions)
    @assert(n == m, "Number of subpartition specifications do no match number of partitions")

    next_partition_index = maximum(q.index for q in Q) + 1

    Qnew = Partition{I, T, P}()
    for (q, split_spec) in zip(Q, num_subpartitions)
        if (LazySets.isbounded ∘ region)(q)
            rect = box_approximation(region(q))
            new_rects = split(rect, split_spec)
            
            for rect in new_rects
                push!(Qnew, Region(next_partition_index, convert(P, rect), dynamics(q)))
                next_partition_index += 1
            end
        else
            push!(Qnew, q)
        end
    end

    return System{N, I, T, P, VP}(Qnew, noise(system), initial_set(system), safe_set(system), unsafe_set(system))
end
