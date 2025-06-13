using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats
using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

function drone(uncertain=true)
    # Only modeling longitudinal dynamics
    # x[1] = position
    # x[2] = velocity
    vars = Symbolics.@variables x[1:2]

    Q = [
        Hyperrectangle(low=[-10, -10], high=[14, 10]),
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
    dyn = [Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [0.5m, 1.0m] × [-0.5m/s, 0.5m/s]
    X0 = Hyperrectangle(low=[0.5, -0.5], high=[1.0, 0.5])

    # Low velocity is safe
    Xs = Hyperrectangle(low=[-10, -10], high=[14, 10])

    # High velocity is dangerous
    Xu = [
        HPolyhedron([x[1] ≥ 14], vars),
        HPolyhedron([x[1] ≤ -10], vars),
        HPolyhedron([x[2] ≥ 10], vars),
        HPolyhedron([x[2] ≤ -10], vars)
    ]

    split_spec = Int64[
        [24]
        [20]
    ]
    Q, dyn = grid_split(Q, dyn, split_spec)

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_drone_exact = Problem("drone_exact", drone(false), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

probs_drone = [
    prob_drone_exact
];

prob_drone_1e_2 = Problem("drone_1e_2", drone(false), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-2))
prob_drone_1e_3 = Problem("drone_1e_3", drone(false), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-3))
prob_drone_1e_4 = Problem("drone_1e_4", drone(false), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-4))
prob_drone_1e_9 = Problem("drone_1e_9", drone(false), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

probs_drone_1e = [
    prob_drone_1e_2,
    prob_drone_1e_3,
    prob_drone_1e_4,
    prob_drone_1e_9
];

function polynomial_drone(uncertain=true)
    # Only modeling longitudinal dynamics
    # x[1] = position
    # x[2] = velocity
    vars = Symbolics.@variables x[1:2]

    Q = [
        Hyperrectangle(low=[-10, -10], high=[14, 10]),
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
    dyn = [Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [-0.5m, -0.5m] × [-0.5m/s, 0.5m/s]
    X0 = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])

    # Low velocity is safe
    Xs = Hyperrectangle(low=[-10, -10], high=[14, 10])

    # High velocity is dangerous
    Xu = [
        Complement(Xs)
    ]

    split_spec = Int64[
        [24]
        [20]
    ]
    Q, dyn = grid_split(Q, dyn, split_spec)

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_drone_1e_2_saa = Problem("drone_1e_2_saa", polynomial_drone(false), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-2))
prob_drone_1e_3_saa = Problem("drone_1e_3_saa", polynomial_drone(false), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-3))
prob_drone_1e_4_saa = Problem("drone_1e_4_saa", polynomial_drone(false), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-4))
prob_drone_1e_9_saa = Problem("drone_1e_9_saa", polynomial_drone(false), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-9))

probs_drone_1e_saa = [
    prob_drone_1e_2_saa,
    prob_drone_1e_3_saa,
    prob_drone_1e_4_saa,
    prob_drone_1e_9_saa
];


function cdc_drone()
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

    Ab = [AffineDynamics{Float64}([
            1 τ;
            0 (m - 0.1 * τ) / m
        ], Float64[0, 0])]
    dyn = [Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab, Ab]

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [0.5m, 1.0m] × [-0.5m/s, 0.5m/s]
    X0 = Hyperrectangle(low=[0.5, -0.5], high=[1.0, 0.5])

    # Low velocity is safe
    Xs = HPolyhedron([x[2] ≤ 10, x[2] ≥ -10], vars)

    # High velocity is dangerous
    Xu = [
        HPolyhedron([x[2] ≥ 10], vars),
        HPolyhedron([x[2] ≤ -10], vars)
    ]

    split_spec = Int64[
        [5 1 1 1 1 1 1 1 1 1 1 1 1]
        [5 1 1 1 1 1 1 1 1 1 1 1 1]
    ]
    Q, dyn = grid_split(Q, dyn, split_spec)

    elems = [DynamicsRegion(i, qi, dyni) for (i, (qi, dyni)) in enumerate(zip(Q, dyn))]

    Q = LinearIndex{Float64, DynamicsRegion{Float64}}()
    bulk_load!(Q, elems)

    return PiecewiseSystem{2, Float64}(Q, η, X0, Xs, Xu)
end

cdc_prob_drone_exact = Problem("cdc_drone_exact", cdc_drone(), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

function cdc_drone_saa()
    # Only modeling longitudinal dynamics
    # x[1] = position
    # x[2] = velocity
    @polyvar x[1:2]

    m₋, m₊ = 0.75, 1.25
    m = (m₋ + m₊) / 2
    τ = 1.0

    A = [
            1 τ;
            0 (m - 0.1 * τ) / m
        ]
    f = A * x

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [0.5m, 1.0m] × [-0.5m/s, 0.5m/s]
    X0 = @set (x[1] - 0.75)^2 ≤ 0.25^2 && x[2]^2 ≤ 0.5^2

    # Low velocity is safe
    Xs = @set x[2]^2 ≤ 10^2

    # High velocity is dangerous
    Xu = [
        @set x[2]^2 ≥ 10^2
    ]

    return PolynomialSystem{2, Float64}(f, η, X0, Xs, Xu)
end

cdc_prob_drone_exact_saa = Problem("cdc_drone_exact_saa", cdc_drone_saa(), 10, PolynomialTemplate(2), SampleAverage(0.005/10, 1e-6))