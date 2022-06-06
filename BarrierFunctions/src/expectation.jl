using Distributions
using SemialgebraicSets, DynamicPolynomials, MultivariatePolynomials

abstract type AbstractExpectation end
struct AnalyticExpectation <: AbstractExpectation end

function expectation(::AnalyticExpectation, system::System{V, VectorDynamics{V}}, B, σ, p) where {V}
    x, z = state(system), noise(system)
    B_next = subs(B, x => next_state(system, p))

    𝔼Bfx = 0
    for term in terms(B_next)
        for z_i in z
            z_power = degree(term, z_i)

            if z_power == 0
                z_factor = 1
            elseif z_power % 2 == 1
                z_factor = 0
            else
                z_factor = prod(1:2:(z_power-1)) * σ^z_power
            end

            # Replace z^n with z_factor by first replacing z with 1 and then multiplying by z_factor
            term = subs(term, z_i => 1) * z_factor
        end
        𝔼Bfx += term
    end

    return 𝔼Bfx
end