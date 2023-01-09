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

    query = PointQuery(x)
    q = Spatial.findall(query, regions(system))

    @assert length(q) == 1 || pairwise_zeromeasure(q) "x should always belong to exactly one region"

    q = q[1]

    A, b = mix_dynamics(q)
    return A * x .+ b .+ η, false
end

pairwise_zeromeasure(Q) = all(qs -> is_intersection_zeromeasure(qs...), zip(Q[1:end - 1], Q[2:end]))

function mix_dynamics(q)
    dyn = dynamics(q)
    dist = ProbabilitySimplex(length(dyn))
    α = rand(dist)
    
    convex_combination(x) = sum(((αᵢ, xᵢ), ) -> αᵢ * xᵢ, zip(α, x))
    A = convex_combination([Aᵢ for (Aᵢ, bᵢ) in dyn])
    b = convex_combination([bᵢ for (Aᵢ, bᵢ) in dyn])

    return A, b
end