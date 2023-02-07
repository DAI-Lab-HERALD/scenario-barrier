using Revise
using Plots, PythonPlot
using JuMP, MosekTools
using LinearAlgebra

pythonplot()
pygui(true)

function counterexample_barrier(solver, w, b)
    model = Model(solver)

    @variable(model, x[1:2])

    A = [1 w; 0 b]
    B(x) = dot(Float64[1, 2], x)

    Ax = A * x

    @constraint(model, [i=1:2], -0.5 ≤ Ax[i] ≤ 0.5)
    @constraint(model, -0.1 ≤ x[1] ≤ 0.1)
    @constraint(model, -1 ≤ x[2] ≤ 1)

    @objective(model, Max, B(Ax))

    optimize!(model)
    # println(solution_summary(model))

    return objective_value(model)
end

function run_example()
    ws = LinRange(-1, 1, 100)
    bs = LinRange(0.5, 1.5, 100)

    product = [(w, b) for w in ws, b in bs]
    ws = [w for (w, b) in product]
    bs = [b for (w, b) in product]

    solver = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    fw = [counterexample_barrier(solver, w, b) for (w, b) in product]

    p = surface(ws, bs, fw, xlabel = "w", ylabel = "b")
    display(p)
end