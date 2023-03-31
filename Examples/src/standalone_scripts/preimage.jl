using LazySets, Polyhedra, JuMP

using LinearAlgebra

using Plots, ColorSchemes
pgfplotsx()

N = Float64
A = N[0.5 1; -1 0]

qᵢ = LazySets.Hyperrectangle(low=N[-3, 1], high=N[-2, 3])
qᵢ_next = A * qᵢ
qⱼ = LazySets.Hyperrectangle(low=N[1, 1], high=N[2, 3])

preimage = intersection(qᵢ, concretize(InverseLinearMap(A, qⱼ)))

forward = A * preimage

colors = ColorSchemes.seaborn_colorblind6

p = plot(qᵢ, label="P̄ᵢ", xlabel="x₁", ylabel="x₂", fillcolor=colors[1], legend=:bottom, xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfontsize=14)
plot!(p, qⱼ, label="P̄ⱼ", fillcolor=colors[2])
plot!(p, qᵢ_next, label="f(P̄ᵢ) + η(ω)", fillcolor=colors[3])
plot!(p, preimage, label="Qᵢⱼ(ω)", fillcolor=colors[4])
plot!(p, forward, label="f(Qᵢⱼ(ω)) + η(ω)", fillcolor=colors[5])

savefig(p, "figures/preimage.pdf")
display(p)
