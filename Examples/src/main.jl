using Revise
using BarrierFunctions

include("car_with_sidewind.jl")

succ, B, prob = verify(car_with_sidewind, H=10, ϵ=0.05, num_samples=100)
print("$succ, $B, $prob")