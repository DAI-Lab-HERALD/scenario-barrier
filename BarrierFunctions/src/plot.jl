using Plots

export plot_monte_carlo_2d, plot_forward_reachable_set_2d, plot_barrier_1d, plot_barrier_2d


function plot_monte_carlo_2d(system, traj; plot_safe=true, plot_regions=false, xlims=:auto, ylims=:auto)
    p = plot(traj[1, :, :], traj[2, :, :], label=false, xlims=xlims, ylims=ylims)

    plot!(p, initial_set(system), label="X0", fillcolor=:blue)
    if plot_safe
        plot!(p, safe_set(system), label="Xs", fillcolor=:green)
    else
        plot!(p, unsafe_set(system), label="Xu", fillcolor=:red)
    end

    if plot_regions
        plot!([region(q) for q in regions(system)])
    end

    display(p)
end

# function plot_forward_reachable_set_2d(system, num_samples; plot_safe=true, xlims=nothing, ylims=nothing)
#     distribution = noise(system)
#     ηs = rand(distribution, num_samples)
#     conv_hull_η = VPolytope(convex_hull([copy(η) for η in eachcol(ηs)]))
#     conv_hull_η = tohrep(conv_hull_η)

#     for q in regions(system)
#         p = plot(initial_set(system), label="X0", fillcolor=:blue, xlims=xlims, ylims=ylims)
#         if plot_safe
#             plot!(p, safe_set(system), label="Xs", fillcolor=:green)
#         else
#             plot!(p, unsafe_set(system), label="Xu", fillcolor=:red)
#         end

#         Aᵢ, bᵢ = dynamics(q)
#         q_next = minkowski_sum(Aᵢ * region(q) + bᵢ, conv_hull_η, prune=false)
#         plot!(p, region(q), label="q", fillcolor=:goldenrod)
#         plot!(p, q_next, label="x(k+1) for q", fillcolor=:maroon)

#         query = RegionIntersectsQuery(q_next)
#         next_regions = Spatial.findall(query, regions(system))

#         for qⱼ in next_regions
#             plot!(p, region(qⱼ), label="overlapping", fillcolor=:yellow)
#         end

#         display(p)
#     end
# end

function plot_forward_reachable_set_2d(system, num_samples; plot_safe=true, xlims=nothing, ylims=nothing)
    distribution = noise(system)
    η = rand(distribution)

    for q in regions(system)
        Aᵢ, bᵢ = dynamics(q)
        q_next = concretize(Aᵢ * region(q) + bᵢ + η)

        query = RegionIntersectsQuery(q_next)
        next_regions = Spatial.findall(query, regions(system))

        for qⱼ in next_regions
            p = plot(initial_set(system), label="X0", fillcolor=:blue, xlims=xlims, ylims=ylims)
            # if plot_safe
            #     plot!(p, safe_set(system), label="Xs", fillcolor=:green)
            # else
            #     plot!(p, unsafe_set(system), label="Xu", fillcolor=:red)
            # end
            plot!(p, region(q), label="q", fillcolor=:goldenrod)
            plot!(p, q_next, label="x(k+1) for q", fillcolor=:red)
            plot!(p, region(qⱼ), label="overlapping", fillcolor=:yellow)

            Hⱼ, hⱼ = tosimplehrep(region(qⱼ))
            input_set = HPolyhedron(Hⱼ * Aᵢ, hⱼ - Hⱼ * (bᵢ + η))
            input_set = intersection(input_set, region(q))
            plot!(p, input_set, label="input_set", fillcolor=:green)

            display(p)
        end
    end
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

    plot!(x, f, label="B")
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

    plot!(x, f, label="B")
end