using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats

function martingale1d(uncertain=true)
    vars = Symbolics.@variables x

    Q = [
        HPolyhedron([x ≥ 2.5], vars),
        HPolyhedron([x ≤ -2.5], vars),
        HPolyhedron([x ≤ 2.5, x ≥ 0.5], vars),
        HPolyhedron([x ≤ 0.5, x ≥ -0.5], vars),
        HPolyhedron([x ≤ -0.5, x ≥ -2.5], vars)
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
    X0 = HPolyhedron([x ≤ 0.5, x ≥ -0.5], vars)

    # Center is safe
    Xs = HPolyhedron([x ≤ 2.5, x ≥ -2.5], vars)

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