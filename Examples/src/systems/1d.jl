using BarrierFunctions
using LazySets, Symbolics
using Spatial
using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

using Distributions, PDMats, Random

function martingale1d(uncertain=true, gaussian=true)
    vars = Symbolics.@variables x

    Q = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars),
        LazySets.Interval(1.5, 2.5),
        LazySets.Interval(0.5, 1.5),
        LazySets.Interval(-0.5, 0.5),
        LazySets.Interval(-1.5, -0.5),
        LazySets.Interval(-2.5, -1.5)
    ]

    if uncertain
        Ab = [AffineDynamics(Float64[1][:,:], [-0.05]), AffineDynamics(Float64[1][:,:], [0.05])]
    else
        Ab = singleton_dynamics(Float64[1][:,:], [0.0])
    end
    dyn = [Ab for _ in 1:length(Q)]

    split_spec = Int64[1 1 5 5 5 5 5][:,:]
    Q, dyn = grid_split(Q, dyn, split_spec)

    if gaussian
        μ = [0]
        Σ = PDiagMat([0.01^2])  # Square to make std.dev. to variance.
        η = DiagNormal(μ, Σ)
    else
        μ = [-5.0]
        Σ = PDiagMat([0.1^2])  # Square to make std.dev. to variance.
        η = MvLogNormal(DiagNormal(μ, Σ))
    end

    # Start in a box [-0.5, 0.5]
    X0 = LazySets.Interval(-0.5, 0.5)

    # Center is safe
    Xs = LazySets.Interval(-2.5, 2.5)

    # Outside the center is dangerous
    Xu = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars)
    ]

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_linear1d_uncertain = Problem("linear1d_uncertain", martingale1d(true, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))
prob_linear1d_exact = Problem("linear1d_exact", martingale1d(false, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

probs_linear1d = [
    prob_linear1d_uncertain,
    prob_linear1d_exact
];

prob_linear1d_uncertain_constant = Problem("linear1d_uncertain_constant", martingale1d(true, true), 10, PiecewiseTemplate(ConstantTemplate()), ScenarioApproach(0.005/10, 1e-9))
prob_linear1d_exact_constant = Problem("linear1d_exact_constant", martingale1d(false, true), 10, PiecewiseTemplate(ConstantTemplate()), ScenarioApproach(0.005/10, 1e-9))

probs_linear1d_constant = [
    prob_linear1d_uncertain_constant,
    prob_linear1d_exact_constant
];

function polynomial_martingale1d(uncertain=true, gaussian=true)
    vars = Symbolics.@variables x

    Q = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars),
        LazySets.Interval(1.5, 2.5),
        LazySets.Interval(0.5, 1.5),
        LazySets.Interval(-0.5, 0.5),
        LazySets.Interval(-1.5, -0.5),
        LazySets.Interval(-2.5, -1.5)
    ]

    if uncertain
        Ab = [AffineDynamics(Float64[1][:,:], [-0.05]), AffineDynamics(Float64[1][:,:], [0.05])]
    else
        Ab = singleton_dynamics(Float64[1][:,:], [0.0])
    end
    dyn = [Ab for _ in 1:length(Q)]

    split_spec = Int64[1 1 5 5 5 5 5][:,:]
    Q, dyn = grid_split(Q, dyn, split_spec)

    if gaussian
        μ = [0]
        Σ = PDiagMat([0.01^2])  # Square to make std.dev. to variance.
        η = DiagNormal(μ, Σ)
    else
        μ = [-5.0]
        Σ = PDiagMat([0.1^2])  # Square to make std.dev. to variance.
        η = MvLogNormal(DiagNormal(μ, Σ))
    end

    # Start in a box [-0.5, 0.5]
    X0 = LazySets.Interval(-0.5, 0.5)

    # Center is safe
    Xs = LazySets.Interval(-2.5, 2.5)

    # Outside the center is dangerous
    Xu = [
        Complement(Xs)
    ]

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_linear1d_uncertain_saa = Problem("linear1d_uncertain_saa", polynomial_martingale1d(true, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 0.01))
prob_linear1d_exact_saa = Problem("linear1d_exact_saa", polynomial_martingale1d(false, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 0.01))

probs_linear1d_saa = [
    prob_linear1d_uncertain_saa,
    prob_linear1d_exact_saa
];

prob_linear_1d_uncertain_1e_2 = Problem("linear1d_uncertain_1e_2", martingale1d(true, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-2))
prob_linear_1d_uncertain_1e_3 = Problem("linear1d_uncertain_1e_3", martingale1d(true, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-3))
prob_linear_1d_uncertain_1e_4 = Problem("linear1d_uncertain_1e_4", martingale1d(true, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-4))
prob_linear_1d_uncertain_1e_9 = Problem("linear1d_uncertain_1e_9", martingale1d(true, true), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

prob_linear_1d_uncertain_1e_2_saa = Problem("linear1d_uncertain_1e_2_saa", polynomial_martingale1d(true, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-2))
prob_linear_1d_uncertain_1e_3_saa = Problem("linear1d_uncertain_1e_3_saa", polynomial_martingale1d(true, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-3))
prob_linear_1d_uncertain_1e_4_saa = Problem("linear1d_uncertain_1e_4_saa", polynomial_martingale1d(true, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-4))
prob_linear_1d_uncertain_1e_9_saa = Problem("linear1d_uncertain_1e_9_saa", polynomial_martingale1d(true, true), 10, PolynomialTemplate(4), SampleAverage(0.005, 1e-9))

probs_linear1d_1e = [
    prob_linear_1d_uncertain_1e_2,
    prob_linear_1d_uncertain_1e_3,
    prob_linear_1d_uncertain_1e_4,
    prob_linear_1d_uncertain_1e_9
];

probs_linear1d_1e_saa = [
    prob_linear_1d_uncertain_1e_2_saa,
    prob_linear_1d_uncertain_1e_3_saa,
    prob_linear_1d_uncertain_1e_4_saa,
    prob_linear_1d_uncertain_1e_9_saa
];

function sanity_check()
    system = martingale1d(false)
    solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER)

    Random.seed!(1596743182)
    _, _, convhull_prob = verify(solver, system, 2,  ScenarioApproach(0.005/2, 1e-4, true))

    Random.seed!(1596743182)
    _, _, no_convhull_prob = verify(solver, system, 2,  ScenarioApproach(0.005/2, 1e-4, false))

    println("Found safety probabilities $convhull_prob / $no_convhull_prob, difference: $(convhull_prob - no_convhull_prob)")
end

function cdc_martingale1d()
    vars = Symbolics.@variables x

    Q = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars),
        LazySets.Interval(1.5, 2.5),
        LazySets.Interval(0.5, 1.5),
        LazySets.Interval(-0.5, 0.5),
        LazySets.Interval(-1.5, -0.5),
        LazySets.Interval(-2.5, -1.5)
    ]

    Ab = singleton_dynamics(Float64[1][:,:], [0.0])
    dyn = [Ab for _ in 1:length(Q)]

    split_spec = Int64[1 1 1 1 1 1 1][:,:]
    Q, dyn = grid_split(Q, dyn, split_spec)

    μ = [0]
    Σ = PDiagMat([0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [-0.5, 0.5]
    X0 = LazySets.Interval(-0.5, 0.5)

    # Center is safe
    Xs = LazySets.Interval(-2.5, 2.5)

    # Outside the center is dangerous
    Xu = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars)
    ]

    elems = [DynamicsRegion(i, qi, dyni) for (i, (qi, dyni)) in enumerate(zip(Q, dyn))]

    Q = LinearIndex{Float64, DynamicsRegion{Float64}}()
    bulk_load!(Q, elems)

    return PiecewiseSystem{1, Float64}(Q, η, X0, Xs, Xu)
end

cdc_prob_linear1d = Problem("cdc_linear1d", cdc_martingale1d(), 10, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

function cdc_martingale1d_polynomial()
    @polyvar x

    f = [x + 0.0]  # Add 0 to ensure that f is interpreted as a polynomial

    μ = [0]
    Σ = PDiagMat([0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    X0 = @set x^2 ≤ 0.5^2
    Xs = @set x^2 ≤ 2.5^2

    Xu = [
        @set x^2 ≥ 2.5^2
    ]

    return PolynomialSystem{1, Float64}(f, η, X0, Xs, Xu)
end

cdc_prob_linear1d_saa = Problem("cdc_linear1d_ssa", cdc_martingale1d_polynomial(), 10, PolynomialTemplate(2), SampleAverage(0.005, 0.01))