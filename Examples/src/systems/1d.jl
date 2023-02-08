using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats, Random

function martingale1d(uncertain=true)
    vars = Symbolics.@variables x

    Q = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars),
        LazySets.Interval(0.5, 2.5),
        LazySets.Interval(-0.5, 0.5),
        LazySets.Interval(-2.5, 0.5)
    ]

    if uncertain
        Ab = [AffineDynamics(Float64[1][:,:], [-0.05]), AffineDynamics(Float64[1][:,:], [0.05])]
    else
        Ab = singleton_dynamics(Float64[1][:,:], [0.0])
    end
    dyn = [Ab, Ab, Ab, Ab, Ab]

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

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end


prob_linear1d_uncertain = Problem("linear1d_uncertain", martingale1d(true), 10, ScenarioApproach(0.005/10, 1e-9))
prob_linear1d_exact = Problem("linear1d_exact", martingale1d(false), 10, ScenarioApproach(0.005/10, 1e-9))

probs_linear1d = [
    prob_linear1d_uncertain,
    prob_linear1d_exact
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