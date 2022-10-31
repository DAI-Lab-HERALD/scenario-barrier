using Revise
using BarrierFunctions

include("1d.jl")
include("car_with_sidewind.jl")

H = 10
traj = monte_carlo(linear1d, H, 1000)
# plot_monte_carlo(car_with_sidewind, traj)

ϵ = 0.005 / H
succ, B, prob = verify(linear1d, H=H, ϵ=ϵ, num_samples=100)
println("Success: $succ")
println("Barrier $(sort(collect(B), by = x->x[1].index))")
println("Safety prob: $prob")
