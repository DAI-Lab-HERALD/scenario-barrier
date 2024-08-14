export monte_carlo, dynstep

function monte_carlo(system, steps, alphas, initial_conditions, particles)
    X0 = initial_set(system)
    η = noise(system)
    nalpha = length(alphas)

    x0 = LazySets.sample(X0, initial_conditions)
    x0 = reshape(x0, 1, initial_conditions, 1, :)
    x0 = repeat(x0, nalpha, 1, particles, 1)

    unsafe = falses((nalpha, initial_conditions, particles))

    for t in 1:steps
        ηs = rand(η, (nalpha, initial_conditions, particles))

        x1 = copy(x0)
        for k in axes(x0, 3)
            for j in axes(x0, 2)
                for i in axes(x0, 1)
                    if !unsafe[i, j, k]
                        x, u = dynstep(system, x0[i, j, k], alphas[i], ηs[i, j, k])

                        x1[i, j, k] = x
                        unsafe[i, j, k] = u
                    end
                end
            end
        end

        x0 = x1
    end

    unsafety_prob = mapslices(U -> count(U) / particles, unsafe, dims=3)
    println(size(unsafe))
    unsafety_prob = maximum(unsafety_prob)
    
    println("Estimated safety prob: $(1.0 - unsafety_prob)")

    return 1.0 - unsafety_prob
end

function dynstep(system, x, alpha, η)
    for q in unsafe_set(system)
        if x ∈ q
            return x, true
        end
    end

    query = PointQuery(x)
    q = Spatial.findall(query, regions(system))

    @assert length(q) == 1 || pairwise_zeromeasure(q) "x should always belong to exactly one region"

    q = q[1]

    A, b = mix_dynamics(q, alpha)
    return A * x .+ b .+ η, false
end

pairwise_zeromeasure(Q) = all(qs -> is_intersection_zeromeasure(qs...), zip(Q[1:end - 1], Q[2:end]))

function mix_dynamics(q, alpha)
    dyn = dynamics(q)
    A₁, b₁ = dyn[1]
    A₂, b₂ = dyn[2]
    
    A = A₁ * (1 - alpha) + A₂ * alpha
    b = b₁ * (1 - alpha) + b₂ * alpha

    return A, b
end