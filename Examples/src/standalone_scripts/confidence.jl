using Revise
using Distributions
using Plots, ColorSchemes
pgfplotsx()

function required_sample(d, ϵ, β)

    lower_bound = d + 1
    upper_bound = ceil((2 / ϵ) * log(1 / β) + 2 * d + (2 * d / ϵ) * log(2 / ϵ))

    while lower_bound + 1 < upper_bound
        mid = floor((lower_bound + upper_bound) / 2)

        η = Binomial(mid, ϵ)
        b = cdf(η, d - 1)

        if b > β
            lower_bound = mid
        else
            upper_bound = mid
        end
    end

    return upper_bound
end

function plot_confidence()
    βs = [10^(-exponent) for exponent in 1:0.1:6]

    # SSA
    δ = 0.015
    M = 1
    ssa_num_samples = Int64.(ceil.(M ./ (δ^2 .* βs)))

    # Scenario approach
    d = 380
    ϵ = 0.0005

    scenario_approach_num_samples = [required_sample(d, ϵ, β) for β in βs]

    println(βs)
    println(ssa_num_samples)
    println(scenario_approach_num_samples)


    colors = ColorSchemes.seaborn_colorblind6

    xticks = reverse([10.0^(-exponent) for exponent in 1:1.0:6]), reverse(["\$10^{-$(round(Int, exponent))}\$" for exponent in 1:1.0:6])
    yticks = [[0]; [x * 10.0^9 for x in 1:4]], [["0"]; ["\$$(round(Int, x)) \\cdot 10^{9}\$" for x in 1:4]]

    p = plot(βs, ssa_num_samples, linewidth=3, linecolor=colors[1], label="Sample average approximation", legend=:topleft, xticks=xticks, yticks=yticks, xscale=:log10, xflip=true, ylabel="Number of samples", xlabel="β", minorgrid=true, xtickfontsize=20, ytickfontsize=20, xguidefontsize=20, yguidefontsize=20, legendfontsize=20)
    plot!(p, βs, scenario_approach_num_samples, linewidth=3, linecolor=colors[3], label="Scenario approach", size=(800, 600))


    yticks = [[0]; [x * 10.0^5 for x in 2:2:8]; [10.0^6]], [["0"]; ["\$$(round(Int, x)) \\cdot 10^{5}\$" for x in 2:2:8]; ["\$1 \\cdot 10^{6}\$"]]

    plot!(p, βs[1:15], ssa_num_samples[1:15], linewidth=3, linecolor=colors[1], inset=(1, bbox(0.15, 0.25, 0.6, 0.5, :top)), ylim=(0, 1100000), label=nothing, xticks=xticks, yticks=yticks, xscale=:log10, xflip=true, minorgrid=true, framestyle=:box, subplot=2, xtickfontsize=20, ytickfontsize=20)
    plot!(p, βs, scenario_approach_num_samples, linewidth=3, linecolor=colors[3], label=nothing, subplot=2)

    savefig(p, "figures/number_of_samples.pdf")
    display(p)

    return p
end