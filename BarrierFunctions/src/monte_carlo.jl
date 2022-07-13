using LazySets


function monte_carlo(system, steps, particles)
    X0 = initial_set(system)
    η = noise(system)

    x0 = LazySets.sample(X0, particles)
    traj = [copy(x0)]

    for t in 1:steps
        ηs = rand(η, particles)

        x1 = [step(system, x, η) for (x, η) in zip(x0, eachcol(ηs))]
        push!(traj, x1)

        x0 = x1
    end

    traj = map(row -> reduce(hcat, row), traj)
    traj = reduce((a, b) -> cat(a, b, dims=3), traj)
    traj = permutedims(traj, (1, 3, 2))
    return traj
end

function step(system, x, η)
    for q in partitions(system)
        if x ∈ region(q)
            A, b = dynamics(q)
            return A * x .+ b .+ η
        end
    end
    @assert(false, "x should always belong to one partition")
end