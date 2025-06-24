using BarrierFunctions
using LazySets, Symbolics

using Distributions, PDMats
using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

function room_temp()
    vars = Symbolics.@variables x[1:3]

    Q = [
        Hyperrectangle(low=[17.0, 17.0, 17.0], high=[30.0, 30.0, 30.0]),
    ]

    Te = 10.0
    αₑ = 8e-3
    α = 6.2e-3

    τ = 5.0

    Ab = [
        AffineDynamics{Float64}([
            (1 - τ * (α + αₑ))  (τ * α) 0.0;
            (τ * α) (1 - τ * (2α + αₑ)) (τ * α);
            0.0 (τ * α) (1 - τ * (α + αₑ))
        ], Float64[τ * αₑ * Te, τ * αₑ * Te, τ * αₑ * Te]),
        AffineDynamics{Float64}([
            (1 - τ * (α + αₑ))  (τ * α) 0.0;
            (τ * α) (1 - τ * (2α + αₑ)) (τ * α);
            0.0 (τ * α) (1 - τ * (α + αₑ))
        ], Float64[τ * αₑ * Te, τ * αₑ * Te, τ * αₑ * Te])
    ]
    dyn = [Ab]

    μ = [0, 0, 0]
    Σ = PDiagMat([0.01^2, 0.01^2, 0.01^2])  # Square to make std.dev. to variance.
    η = DiagNormal(μ, Σ)

    # Start in a box [17ᵒC, 18ᵒC]^3
    X0 = Hyperrectangle(low=[17.0, 17.0, 17.0], high=[18.0, 18.0, 18.0])

    # Low velocity is safe
    Xs = Hyperrectangle(low=[17.0, 17.0, 17.0], high=[29.0, 29.0, 29.0])

    Xu = [
        HPolyhedron([x[1] ≥ 29.0], vars),
        HPolyhedron([x[2] ≥ 29.0], vars),
        HPolyhedron([x[3] ≥ 29.0], vars),
    ]

    split_spec = Int64[
        [13]
        [13]
        [13]
    ]
    Q, dyn = grid_split(Q, dyn, split_spec)

    return PiecewiseSystem(Q, dyn, η, X0, Xs, Xu)
end

prob_room_temp = Problem("room_temp", room_temp(), 3, PiecewiseTemplate(LinearTemplate()), ScenarioApproach(0.005/10, 1e-9))

probs_room_temp = [
    prob_room_temp
];
