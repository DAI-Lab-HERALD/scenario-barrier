using Revise
using LazySets, Polyhedra, JuMP, CDDLib

using LinearAlgebra

using Plots, ColorSchemes, LaTeXStrings
# pgfplotsx()
# plotlyjs()
gr()

N = Float64
A = N[0.5 1; -1 0]

qᵢ = LazySets.Hyperrectangle(low=N[-3, 1], high=N[-2, 2.5])
lower, upper = low(qᵢ), high(qᵢ)
qⱼ = LazySets.Hyperrectangle(low=N[1, 1], high=N[2, 2.5])

f(x) = [x[2], x[1]^3 / 10 + 2 * x[2]]
finv(y) = [cbrt(10 * y[2] - 20 * y[1]), y[1]]

f̂(x, α) = ([0 1; 1.9 2] * x + [0.0; 3.0]) * α + ([0 1; 1.875 2] * x + [0; 3.125]) * (1 - α)
f̂inv(y, α) = InverseLinearMap([0 1; 1.9 2], y + (-[0.0; 3.0])) * α + InverseLinearMap([0 1; 1.875 2], y + (-[0; 3.125])) * (1 - α)

## Check that the linear relaxations are sound.
function check_linear_relaxations()
    x_range = LinRange(lower[1], upper[1], 100)
    y_range = LinRange(lower[2], upper[2], 100)

    f1(x, y) = f([x, y])[1]
    f2(x, y) = f([x, y])[2]

    f̂1lower(x, y) = f̂([x, y], 0)[1]
    f̂2lower(x, y) = f̂([x, y], 0)[2]
    f̂1upper(x, y) = f̂([x, y], 1)[1]
    f̂2upper(x, y) = f̂([x, y], 1)[2]

    red_cd = cgrad([:red, :red])
    green_cd = cgrad([:green, :green])
    p = Plots.surface(x_range, y_range, f2, xlabel=L"x_1", ylabel=L"x_2", color=red_cd, colorbar=false)
    surface!(p, x_range, y_range, f̂2lower, color=green_cd)
    surface!(p, x_range, y_range, f̂2upper, color=green_cd)
    display(p)
end

## Prepare plot
qᵢ_next_α0 = f̂(qᵢ, 0)
qᵢ_next_α1 = f̂(qᵢ, 1)
qᵢ_next = convex_hull(qᵢ_next_α0, qᵢ_next_α1)

forward = intersection(qᵢ_next, qⱼ)

Qᵢⱼω_α = [f̂inv(forward, α) for α in 0:0.1:1]

Qᵢⱼω_α0 = f̂inv(forward, 0)
Qᵢⱼω_α1 = f̂inv(forward, 1)
Qᵢⱼω = intersection(convex_hull(Qᵢⱼω_α0, Qᵢⱼω_α1), qᵢ)

Qᵢⱼω_next_α0 = f̂(Qᵢⱼω, 0)
Qᵢⱼω_next_α1 = f̂(Qᵢⱼω, 1)
Qᵢⱼω_next = convex_hull(Qᵢⱼω_next_α0, Qᵢⱼω_next_α1)

bar_Qij_omega = box_approximation(Qᵢⱼω)
# bar_Qij_omega_next = VPolytope(f.(vertices_list(bar_Qij_omega)))

bar_Qᵢⱼω_next_α0 = f̂(bar_Qij_omega, 0)
bar_Qᵢⱼω_next_α1 = f̂(bar_Qij_omega, 1)
bar_Qᵢⱼω_next = convex_hull(bar_Qᵢⱼω_next_α0, bar_Qᵢⱼω_next_α1)

colors = ColorSchemes.seaborn_colorblind

p = plot(qᵢ, label=L"$q_i$", xlabel=L"$x_1$", ylabel=L"$x_2$", fillcolor=colors[1], fillstyle = :/, legend=:outertop, legendcolumns=2, xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfontsize=14, size=(660, 660))
plot!(p, qᵢ_next, label=L"$\mathrm{conv}(F(q_i) + v)$", fillcolor=colors[10], fillstyle = :/)
plot!(p, qⱼ, label=L"$q_j$", fillcolor=colors[3], fillstyle = :|)
plot!(p, Qᵢⱼω_next, label=L"$\mathrm{conv}(F(q_{ij}(v)) + v)$", fillcolor=colors[4], alpha=0.5)
plot!(p, Qᵢⱼω, label=L"$q_{ij}(v)$", fillcolor=colors[4], alpha=0.5)
plot!(p, bar_Qᵢⱼω_next, label=L"$\mathrm{conv}(F(\overline{q}_{ij}(v)) + v)$", fillcolor=colors[5], fillstyle = :\)
plot!(p, bar_Qij_omega, label=L"$\overline{q}_{ij}(v)$", fillcolor=colors[5], fillstyle = :\)

# for Qᵢⱼω in Qᵢⱼω_α
#     plot!(p, Qᵢⱼω, fillcolor=colors[5], alpha=0.8)
# end

savefig(p, "figures/preimage.pdf")
display(p)
