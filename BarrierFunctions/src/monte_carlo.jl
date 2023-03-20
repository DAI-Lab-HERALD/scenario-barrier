export monte_carlo, step

# TODO: Distinguish between initial conditions and particles,
# and take the max over initials of the average over particles.
function monte_carlo(system, steps, initial_conditions, particles)
    X0 = initial_set(system)
    η = noise(system)

    x0 = LazySets.sample(X0, initial_conditions)
    x0 = reshape(x0, initial_conditions, 1, :)
    x0 = repeat(x0, 1, particles, 1)

    traj = [copy(x0)]
    unsafe = falses((initial_conditions, particles))

    for t in 1:steps
        ηs = rand(η, (initial_conditions, particles))

        x1 = copy(x0)
        for j = axes(x0, 2)
            for i = axes(x0, 1)
                if !unsafe[i, j]
                    x, u = step(system, x0[i, j], ηs[i, j])

                    x1[i, j] = x
                    unsafe[i, j] = u
                end
            end
        end

        push!(traj, x1)
        x0 = x1
    end

    unsafety_prob = map(unsafe_particles -> count(unsafe_particles) / particles, eachcol(unsafe))
    unsafety_prob = maximum(unsafety_prob)
    
    println("Estimated safety prob: $(1 - unsafety_prob)")

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