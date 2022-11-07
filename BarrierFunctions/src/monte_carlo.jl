using LazySets

export monte_carlo, step

# TODO: Distinguish between initial conditions and particles,
# and take the max over initials of the average over particles.
function monte_carlo(system, steps, particles)
    X0 = initial_set(system)
    η = noise(system)

    x0 = LazySets.sample(X0, particles)
    traj = [copy(x0)]
    unsafe = falses(size(x0))

    for t in 1:steps
        ηs = rand(η, particles)

        next_step = [step(system, x, η) for (x, η) in zip(x0, eachcol(ηs))]
        x1 = [x for (x, _) in next_step]
        push!(traj, x1)

        u = [u for (_, u) in next_step]
        unsafe = unsafe .| u

        x0 = x1
    end
    
    println("Estimated safety prob: $(1 - count(unsafe) / particles)")

    traj = map(row -> reduce(hcat, row), traj)
    traj = reduce((a, b) -> cat(a, b, dims=3), traj)
    traj = permutedims(traj, (1, 3, 2))
    return traj
end

function step(system, x, η)
    for q in unsafe_set(system)
        if x ∈ q
            return x, true
        end
    end

    for q in partitions(system)
        if x ∈ region(q)
            A, b = dynamics(q)
            return A * x .+ b .+ η, false
        end
    end
    @assert(false, "x should always belong to one partition")
end