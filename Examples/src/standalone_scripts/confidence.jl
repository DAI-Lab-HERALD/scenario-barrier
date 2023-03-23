using Revise
using Plots

βs = [10^(-exponent) for exponent in 1:0.1:6]

# SSA
δ = 0.015
M = 1
ssa_num_samples = Int64.(ceil.(M ./ (δ^2 .* βs)))

# Scenario approach
d = 380
ϵ = 0.005

function required_sample(d, ϵ, β)
    lower_bound = d + 1
    upper_bound = ceil(Int, (ℯ / (ℯ - 1)) * (1 / ϵ) * (d + log(1 / β)))  # The funny ℯ is Euler's constant

    beta(N) = sum(i -> binomial(big(N), i) * ϵ^i * (1 - ϵ)^(N - i), 0:(d - 1))

    while lower_bound + 1 < upper_bound
        mid = floor(Int64, (lower_bound + upper_bound) / 2)
        b = beta(mid)

        if b > β
            lower_bound = mid
        else
            upper_bound = mid
        end
    end

    return upper_bound
end

scenario_approach_num_samples = [required_sample(d, ϵ, β) for β in βs]


println(βs)
println(ssa_num_samples)
println(scenario_approach_num_samples)

plot(βs, ssa_num_samples, label="Sample average approximation", xscale=:log10, xflip=true, ylabel="Number of samples", xlabel="β", minorgrid=true)
plot!(βs, scenario_approach_num_samples, label="Scenario approach")


plot!(βs, ssa_num_samples, inset=(1, bbox(0.2, 0.25, 0.5, 0.4, :top)), ylim=(20000, 100000), label=nothing, xscale=:log10, xflip=true, minorgrid=true, framestyle=:box, subplot=2)
plot!(βs, scenario_approach_num_samples, label=nothing, subplot=2)

savefig("figures/number_of_samples.pdf")