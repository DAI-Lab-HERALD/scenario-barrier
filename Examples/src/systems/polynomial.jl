using BarrierFunctions
using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

using Distributions, PDMats

function polynomial()
    @polyvar x[1:2]

    dt = 0.1

    f = [x[2] * dt, (-x[1] - x[2] + x[1]^3 / 3) * dt]

    μ = [0, 0]
    Σ = PDiagMat([0.1^2, 0.1^2])  # Square to make std.dev. to variance.
    η = MvNormal(μ, Σ)

    X0 = @set (x[1] - 1.5)^2 + x[2]^2 ≤ 0.5^2
    Xs = @set (x[1] + 1)^2 + (x[2] + 1)^2 ≥ 0.4^2

    Xu = [
        @set (x[1] + 1)^2 + (x[2] + 1)^2 ≤ 0.4^2
    ]

    return PolynomialSystem{2, Float64}(f, η, X0, Xs, Xu)
end

prob_polynomial_scenario_approach = Problem("prob_polynomial_scenario_approach", polynomial(), 1, PolynomialTemplate(2), ScenarioApproach(0.005/1, 1e-4, false))
prob_polynomial_sample_average = Problem("prob_polynomial_sample_average", polynomial(), 1, PolynomialTemplate(2), SampleAverage(0.005/1, 1e-4))

probs_polynomial = [
    prob_polynomial_scenario_approach,
    prob_polynomial_sample_average
];