struct Problem
    name::String
    system::AbstractSystem
    horizon::Int64

    barrier_type::Type{<:BarrierFunction}
    expectation_method::ExpectationMethod
end