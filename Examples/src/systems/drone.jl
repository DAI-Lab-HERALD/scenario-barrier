using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

function drone(uncertain=true)
    # Only modeling longitudinal dynamics
    # x[1] = position
    # x[2] = velocity
    vars = Symbolics.@variables x[1:2]

    Q = [
        Hyperrectangle(low=[0, -10], high=[20, 10]),
        HPolyhedron([x[1] ≤ 0, x[2] ≥ -10, x[2] ≤ -6], vars),
        HPolyhedron([x[1] ≤ 0, x[2] ≥ -6, x[2] ≤ -2], vars),
        HPolyhedron([x[1] ≤ 0, x[2] ≥ -2, x[2] ≤ 2], vars),
        HPolyhedron([x[1] ≤ 0, x[2] ≥ 2, x[2] ≤ 6], vars),
        HPolyhedron([x[1] ≤ 0, x[2] ≥ 6, x[2] ≤ 10], vars),
        HPolyhedron([x[1] ≥ 20, x[2] ≥ -10, x[2] ≤ -6], vars),
        HPolyhedron([x[1] ≥ 20, x[2] ≥ -6, x[2] ≤ -2], vars),
        HPolyhedron([x[1] ≥ 20, x[2] ≥ -2, x[2] ≤ 2], vars),
        HPolyhedron([x[1] ≥ 20, x[2] ≥ 2, x[2] ≤ 6], vars),
        HPolyhedron([x[1] ≥ 20, x[2] ≥ 6, x[2] ≤ 10], vars),
        HPolyhedron([x[2] ≥ 10], vars),
        HPolyhedron([x[2] ≤ -10], vars),
    ]

    m₋, m₊ = 0.75, 1.25
    m = (m₋ + m₊) / 2
    τ = 1.0

    if uncertain
        Ab = [
            AffineDynamics{Float64}([
                1 τ;
                0 (m₋ - 0.1 * τ) / m₋
            ], Float64[0, 0]),
            AffineDynamics{Float64}([
                1 τ;
                0 (m₊ - 0.1 * τ) / m₊
            ], Float64[0, 0])]
    else
        Ab = [AffineDynamics{Float64}([
                1 τ;
                0 (m - 0.1 * τ) / m
            ], Float64[0, 0])]
    end
    dyn = [Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [0.5m, 1.0m] × [-0.5m/s, 0.5m/s]
    X0 = Hyperrectangle(low=[0.5, -0.5], high=[1.0, 0.5])

    # On the road is safe
    Xs = HPolyhedron([x[2] ≤ 10, x[2] ≥ -10], vars)

    # Outside the road is dangerous
    Xu = [
        HPolyhedron([x[2] ≥ 10], vars),
        HPolyhedron([x[2] ≤ -10], vars)
    ]

    split_spec = Int64[
        [5 1 1 1 1 1 1 1 1 1 1 1 1]
        [5 1 1 1 1 1 1 1 1 1 1 1 1]
    ]
    Q, dyn = grid_split(Q, dyn, split_spec)

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_drone_uncertain = Problem("drone_uncertain", drone(true), 10, ScenarioApproach(0.005/10, 1e-9))
prob_drone_exact = Problem("drone_exact", drone(false), 10, ScenarioApproach(0.005/10, 1e-9))

probs_drone = [
    prob_drone_uncertain,
    prob_drone_exact
];