using Revise
using BarrierFunctions
using Gurobi, MosekTools, GLPK

using Plots, PythonPlot
pythonplot()
pygui(true)

include("1d.jl")
include("car_with_sidewind.jl")
include("drone.jl")
include("nndm.jl")

system = car_with_sidewind
H = 10
traj = monte_carlo(system, H, 10000)
# plot_monte_carlo_2d(system, traj, plot_safe=false)
# plot_forward_reachable_set_2d(system, 1000, plot_safe=false, xlims=[-10, 160], ylims=[-10, 10])


ϵ = 0.05 / H
succ, B, prob = verify(Mosek.Optimizer, system, H=H, ϵ=ϵ, β=1e-9)
println("Success: $succ")
# println("Barrier $B")
println("Safety prob: $prob")

plot_barrier(system, B)
