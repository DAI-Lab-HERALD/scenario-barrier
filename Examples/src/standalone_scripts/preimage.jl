using LazySets, Polyhedra, JuMP

using LinearAlgebra

using Plots, ColorSchemes, LaTeXStrings
pgfplotsx()

N = Float64
A = N[0.5 1; -1 0]

qᵢ = LazySets.Hyperrectangle(low=N[-3, 1], high=N[-2, 3])
qᵢ_next = A * qᵢ
qⱼ = LazySets.Hyperrectangle(low=N[1, 1], high=N[2, 3])

preimage = intersection(qᵢ, concretize(InverseLinearMap(A, qⱼ)))

forward = A * preimage

colors = ColorSchemes.seaborn_colorblind6

p = plot(qᵢ, label=L"$\bar{P}_i$", xlabel=L"$x_1$", ylabel=L"$x_2$", fillcolor=colors[1], legend=:bottom, xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfontsize=14)
plot!(p, qⱼ, label=L"$\bar{P}_j$", fillcolor=colors[2])
plot!(p, qᵢ_next, label=L"$f(\bar{P}_i) + \eta(\omega)$", fillcolor=colors[3])
plot!(p, preimage, label=L"$Q_{ij}(\omega)$", fillcolor=colors[4])
plot!(p, forward, label=L"$f(Q_{ij}(\omega)) + \eta(\omega)$", fillcolor=colors[5])

savefig(p, "figures/preimage.pdf")
display(p)
