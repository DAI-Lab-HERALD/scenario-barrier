using LinearAlgebra
using Distributions, Random

using JuMP, MosekTools, Mosek

# All economic values are per unit
newsvendor_optimal_inventory(solver, demand_dist, cost, sales_price, salavage_value, ϵ, β) = newsvendor_optimal_inventory(solver, demand_dist, sales_price - cost, cost - salavage_value, ϵ, β) 

function newsvendor_optimal_inventory(solver, demand_dist, profit, loss, ϵ, β)
    # In the original problem, we only have a single variable
    # All others are purely to implement a max into LP and turn the problem into epigraph from
    num_samples = required_samples(1, ϵ, β)

    model = Model(solver)

    @variable(model, x, lower_bound = 0.0, upper_bound = 1500.0)
    @variable(model, t)

    δ = 1500.0 * ϵ

    demand_realizations = rand(demand_dist, num_samples)

    for demand in demand_realizations        
        units_unsold = @variable(model, lower_bound = 0.0)
        demand_unmet = @variable(model, lower_bound = 0.0)

        @constraint(model, units_unsold ≥ x - demand)
        @constraint(model, demand_unmet ≥ demand - x)

        @constraint(model, loss * units_unsold + profit * demand_unmet + (δ - t) / (1 - ϵ) ≤ 0)
    end

    @objective(model, Min, t)

    optimize!(model)

    return value(x), objective_value(model)
end


function required_samples(d, ϵ, β)
    lower_bound = d + 1
    upper_bound = ceil(Int, (ℯ / (ℯ - 1)) * (1 / ϵ) * (d + log(1 / β)))  # The funny ℯ is Euler's constant

    beta(N) = sum(i -> binomial(big(N), i) * ϵ^i * (1 - ϵ)^(N - i), 0:(d - 1))

    while lower_bound + 1 < upper_bound
        mid = floor(Int, (lower_bound + upper_bound) / 2)
        b = beta(mid)

        if b > β
            lower_bound = mid
        else
            upper_bound = mid
        end
    end

    return upper_bound
end


solver = optimizer_with_attributes(Mosek.Optimizer, "INTPNT_BASIS" => MSK_BI_NEVER, "QUIET" => true)
demand_dist = truncated(Normal(1000.0, 100.0), lower = 0.0, upper = 1500.0)
cost = 0.005
sales_price = 0.08
salavage_value = 0.003
ϵ = 0.05
β = 1e-6

results = []
for i in 1:100
    println("Iteration $i")

    res = newsvendor_optimal_inventory(solver, demand_dist, cost, sales_price, salavage_value, ϵ, β)
    push!(results, res)
end

println(results)