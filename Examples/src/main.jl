using Revise
using BarrierFunctions

include("1d.jl")
include("car_with_sidewind.jl")

system = linear1d
H = 10
traj = monte_carlo(system, H, 1000)
# plot_monte_carlo(car_with_sidewind, traj)

ϵ = 0.005 / H
succ, B, prob = verify(system, H=H, ϵ=ϵ, num_samples=100)
println("Success: $succ")
println("Barrier $(sort(collect(B), by = x->x[1].index))")
println("Safety prob: $prob")

x = LinRange(-3.0, 3.0, 100)
# plot_barrier_1d(x, B, system)
