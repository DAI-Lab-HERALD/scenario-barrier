struct Problem
    name::String
    system::AbstractSystem
    horizon::Int64

    barrier_type::BarrierTemplate
    expectation_method::ExpectationMethod
end