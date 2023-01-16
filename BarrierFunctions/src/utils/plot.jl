using Plots
import ColorSchemes

export plot_monte_carlo_2d, plot_forward_reachable_set_2d, plot_barrier


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
    ηs = rand(distribution, num_samples)
    conv_hull_η = VPolytope(convex_hull([copy(η) for η in eachcol(ηs)]))
    conv_hull_η = tohrep(conv_hull_η)

    for q in regions(system)
        dyn = dynamics(q)
        Aᵢ, bᵢ = dyn[1]  # TODO: Deal with convex hull dynamics
        q_next = minkowski_sum(Aᵢ * region(q) + bᵢ, conv_hull_η, prune=false)

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
            plot!(p, q_next, label="noisy x(k+1) for q", fillcolor=:red)
            plot!(p, concretize(Aᵢ * region(q) + bᵢ), label="nominal x(k+1) for q", fillcolor=:orange)
            plot!(p, region(qⱼ), label="overlapping", fillcolor=:yellow)

            # Hⱼ, hⱼ = tosimplehrep(region(qⱼ))
            # input_set = HPolyhedron(Hⱼ * Aᵢ, hⱼ - Hⱼ * (bᵢ + η))
            # input_set = intersection(input_set, region(q))
            # plot!(p, input_set, label="input_set", fillcolor=:green)

            display(p)
        end
    end
end

function plot_barrier(system::PiecewiseSystem{1, T, P}, B; xlims=nothing, plot_safe=true) where {T, P}
    if isnothing(xlims)
        xbounds = bounded_region_lims(system)
        xext = 0.2 .* radius_hyperrectangle(xbounds)[1]  # Add a little space on either side of the MBR
        xlims = (low(xbounds, 1) - xext, high(xbounds, 1) + xext)
    end
    ylims = (-0.05, 1.2)

    p = plot(initial_set(system), label="X0", color=ColorSchemes.tab10[1], xlims=xlims, ylims=ylims, linewidth=4)
    if plot_safe
        plot!(p, safe_set(system), label="Xs", color=ColorSchemes.tab10[3], linewidth=4)
    else
        plot!(p, unsafe_set(system), label="Xu", color=ColorSchemes.tab10[4], linewidth=4)
    end

    # Iterate over barrier regions, as the barrier is continuous for each region, but discontinuous across
    xbounds = Hyperrectangle(low=[xlims[1]], high=[xlims[2]])
    for (i, Bᵢ) in enumerate(B)
        q = intersection(region(Bᵢ), xbounds)   # Intersection with plotting box to account for unbounded regions
        x = LinRange(low(q, 1), high(q, 1), 2)
        label = i == 1 ? "B" : nothing          # Only attach label to first region for legend

        plot!(p, x, z -> Bᵢ(z), label=label, color=ColorSchemes.tab10[2], ylims=ylims)
    end
    return p
end

function plot_barrier(system::PiecewiseSystem{2, T, P}, B; xylims=nothing, plot_safe=true) where {T, P}
    if isnothing(xylims)
        xybounds = bounded_region_lims(system)
        xyext = 0.2 .* radius_hyperrectangle(xybounds)  # Add a little space on either side of the MBR
        xylims = (low(xybounds) .- xyext, high(xybounds) .+ xyext)
    end
    xlims = (xylims[1][1], xylims[2][1])
    ylims = (xylims[1][2], xylims[2][2])
    zlims = [-0.05, 1.2]

    # p = plot(initial_set(system), z=0, st=:surface, label="X0", color=ColorSchemes.tab10[1], xlims=xlims, ylims=ylims, zlims=zlims, linewidth=4)
    # if plot_safe
    #     plot!(p, safe_set(system), z=0, st=:surface, label="Xs", color=ColorSchemes.tab10[3], linewidth=4)
    # else
    #     plot!(p, unsafe_set(system), z=0, st=:surface, label="Xu", color=ColorSchemes.tab10[4], linewidth=4)
    # end

    p = plot()
    f(z) = B(z)

    # Iterate over barrier regions, as the barrier is continuous for each region, but discontinuous across
    xybounds = Hyperrectangle(low=xylims[1], high=xylims[2])
    for (i, Bᵢ) in enumerate(B)
        q = intersection(region(Bᵢ), xybounds)   # Intersection with plotting box to account for unbounded regions
        
        xs = LinRange(low(q, 1), high(q, 1), 5)
        ys = LinRange(low(q, 2), high(q, 2), 5)

        xy = [(x, y) for x in xs for y in ys if [x, y] ∈ q]
        xs = [x for (x, _) in xy]
        ys = [y for (_, y) in xy]

        verts = vertices_list(q)
        vertx = [v[1] for v in verts]
        verty = [v[2] for v in verts]

        append!(xs, vertx)
        append!(ys, verty)

        label = i == 1 ? "B" : nothing           # Only attach label to first region for legend

        plot!(p, xs, ys, (x, y) -> Bᵢ([x, y]), st=:surface, label=label, xlims=xlims, ylims=ylims, zlims=zlims, clims=(0, 1))
    end
    return p
end

function bounded_region_lims(system)
    mbrs = [mbr(region) for region in regions(system) if has_mbr(region)]
    return join_mbr(mbrs)
end