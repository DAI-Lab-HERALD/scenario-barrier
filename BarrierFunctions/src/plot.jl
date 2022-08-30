using Plots


function plot_monte_carlo(system, traj, plot_regions=false)
    plot(initial_set(system), label="X0", fillcolor=:blue, xlim=[-5, 160], ylim=[-5, 5])
    plot!(safe_set(system), label="Xs", fillcolor=:green)
    plot!(unsafe_set(system), label="Xu", fillcolor=:red)

    if plot_regions
        plot!([region(q) for q in partitions(system)])
    end

    plot!(traj[1, :, :], traj[2, :, :], label=false)
end