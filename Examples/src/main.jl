using Revise
using BarrierFunctions

include("car_with_sidewind.jl")

H = 10
# traj = monte_carlo(car_with_sidewind, H, 100)
# plot_monte_carlo(car_with_sidewind, traj)

ϵ = 0.05 / H
succ, B, prob = verify(car_with_sidewind, H=H, ϵ=ϵ, num_samples=1)
print("$succ, $B, $prob")
