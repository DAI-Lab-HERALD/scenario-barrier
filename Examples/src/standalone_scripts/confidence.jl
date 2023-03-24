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


colors = palette(:seaborn_colorblind6)

plot(βs, ssa_num_samples, linewidth=3, linecolor=colors[1], label="Sample average approximation", xscale=:log10, xflip=true, ylabel="Number of samples", xlabel="β", minorgrid=true, xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12, legendfontsize=12)
plot!(βs, scenario_approach_num_samples, linewidth=3, linecolor=colors[3], label="Scenario approach")


plot!(βs, ssa_num_samples, linewidth=3, linecolor=colors[1], inset=(1, bbox(0.25, 0.3, 0.5, 0.4, :top)), ylim=(20000, 100000), label=nothing, xscale=:log10, xflip=true, minorgrid=true, framestyle=:box, subplot=2, xtickfontsize=12, ytickfontsize=12)
plot!(βs, scenario_approach_num_samples, linewidth=3, linecolor=colors[3], label=nothing, subplot=2)

savefig("figures/number_of_samples.pdf")