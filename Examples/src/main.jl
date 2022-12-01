using Revise
using BarrierFunctions

using Plots
plotly()

include("1d.jl")
include("car_with_sidewind.jl")

system = linear1d
H = 10
traj = monte_carlo(system, H, 10000)
# plot_monte_carlo_2d(system, traj, plot_safe=false)
# plot_forward_reachable_set_2d(system, 1000, plot_safe=false, xlims=[-10, 160], ylims=[-10, 10])


ϵ = 0.05 / H
succ, B, prob = verify(system, H=H, ϵ=ϵ, β=1e-9)
println("Success: $succ")
# println("Barrier $B")
println("Safety prob: $prob")

# x = LinRange(-3.0, 3.0, 1000)
# plot_barrier_1d(x, B, system)
