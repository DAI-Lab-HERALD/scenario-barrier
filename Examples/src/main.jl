using Revise
using BarrierFunctions

include("car_with_sidewind.jl")

H = 10
ϵ = 0.05 / H
succ, B, prob = verify(car_with_sidewind, H=H, ϵ=ϵ, num_samples=100)
print("$succ, $B, $prob")