using Revise
using BarrierFunctions
using MosekTools, Mosek, JuMP

using Plots
import PythonPlot: pygui

using Logging, LoggingExtras
using ProgressLogging

using CSV, DataFrames

pythonplot()
pygui(true)

include("problem.jl")
include("systems/systems.jl")
include("runner.jl")
# system = nndm_system("pendulum")
# H = 1
# system = car_with_sidewind(split="latlon")
# H = 10
# traj = monte_carlo(system, H, 10000)
# plot_monte_carlo_2d(system, traj, plot_safe=false)
# plot_forward_reachable_set_2d(system, 1000, plot_safe=false, xlims=[-10, 160], ylims=[-10, 10])

run_experiments(probs_car_with_sidewind)
run_experiments(probs_drone)
run_experiments(probs_linear1d)
run_experiments(probs_nndm_pendulum_1layer)
run_experiments(probs_nndm_pendulum_2layer)
run_experiment(prob_dubins_car_1000regions)
run_experiment(prob_room_temp)
run_experiment(prob_lane_keeping_2520regions_181410)

run_experiments(probs_drone_1e)
run_experiments(probs_drone_1e_saa)
run_experiments(probs_room_temp_1e)
run_experiments(probs_room_temp_1e_saa)
run_experiments(probs_nndm_pendulum_1layer_1e)
run_experiments(probs_nndm_pendulum_1layer_1e_saa)
