using Plots

export plot_monte_carlo_2d, plot_barrier_1d, plot_barrier_2d


function plot_monte_carlo_2d(system, traj, plot_regions=false, xlim=nothing, ylim=nothing)
    plot(initial_set(system), label="X0", fillcolor=:blue, xlim=xlim, ylim=ylim)
    if plot_safe
        plot!(safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(unsafe_set(system), label="Xu", fillcolor=:red)
    end

    if plot_regions
        plot!([region(q) for q in partitions(system)])
    end

    plot!(traj[1, :, :], traj[2, :, :], label=false)
end

function plot_barrier_1d(x, barrier, system, plot_safe=true)
    f(z) = barrier(z)
    y = f.(x)
    xlim = [minimum(x), maximum(x)]
    ylim = [minimum(y), maximum(y)]

    plot(initial_set(system), label="X0", fillcolor=:blue, xlim=xlim, ylim=ylim)
    if plot_safe
        plot!(safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(unsafe_set(system), label="Xu", fillcolor=:red)
    end

    plot!(x, f)
end

function plot_barrier_2d(x, y, barrier, system, plot_safe=true)
    # TODO: Plot surface (and figure out how to plot regions on top)
    f(z) = barrier(z)
    y = f.(x)
    xlim = [minimum(x), maximum(x)]
    ylim = [minimum(y), maximum(y)]

    plot(initial_set(system), label="X0", fillcolor=:blue, xlim=xlim, ylim=ylim)
    if plot_safe
        plot!(safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(unsafe_set(system), label="Xu", fillcolor=:red)
    end

    plot!(x, f)
end