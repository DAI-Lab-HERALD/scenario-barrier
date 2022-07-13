

function grid_split(system::System{N, I, T}, num_subpartitions) where {N, I, T}
    P = partitions(system)
    n, m = length(P), length(num_subpartitions)
    @assert(n == m, "Number of subpartition specifications do no match number of partitions")

    next_partition_index = maximum(q.index for q in P)

    Pnew = Partition{I, T}[]
    for (q, spec) in zip(P, num_subpartitions)
        @assert(region(q) )

        min = minimum()
        max = maximum()

        LinRange()
    end
end