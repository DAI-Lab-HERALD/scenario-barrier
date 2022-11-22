using Plots

export plot_monte_carlo_2d, plot_barrier_1d, plot_barrier_2d


function plot_monte_carlo_2d(system, traj; plot_safe=true, plot_regions=false, xlims=:auto, ylims=:auto)
    p = plot(traj[1, :, :], traj[2, :, :], label=false, xlims=xlims, ylims=ylims)

    plot!(p, initial_set(system), label="X0", fillcolor=:blue)
    if plot_safe
        plot!(p, safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(p, unsafe_set(system), label="Xu", fillcolor=:red)
    end

    if plot_regions
        plot!(p, [region(q) for q in partitions(system)])
    end

    display(p)
end

function plot_barrier_1d(x, barrier, system, plot_safe=true)
    f(z) = barrier(z)
    y = f.(x)
    xlims = [minimum(x), maximum(x)]
    ylims = [minimum(y), maximum(y)]

    plot(initial_set(system), label="X0", fillcolor=:blue, xlims=xlims, ylims=ylims)
    if plot_safe
        plot!(safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(unsafe_set(system), label="Xu", fillcolor=:red)
    end

    plot!(x, f, label="B", ylims=(0,Inf))
end

function plot_barrier_2d(x, y, barrier, system, plot_safe=true)
    # TODO: Plot surface (and figure out how to plot regions on top)
    f(z) = barrier(z)
    y = f.(x)
    xlims = [minimum(x), maximum(x)]
    ylims = [minimum(y), maximum(y)]

    plot(initial_set(system), label="X0", fillcolor=:blue, xlims=xlims, ylims=ylims)
    if plot_safe
        plot!(safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(unsafe_set(system), label="Xu", fillcolor=:red)
    end

    plot!(x, f, label="B")
end