using Revise
using BarrierFunctions

include("1d.jl")

H = 10
traj = monte_carlo(linear1d, H, 1000)
# plot_monte_carlo(car_with_sidewind, traj)

ϵ = 0.05 / H
succ, B, prob = verify(linear1d, H=H, ϵ=ϵ, num_samples=100)
print("$succ, $(sort(collect(B), by = x->x[1].index)), $prob")
