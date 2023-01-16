struct Problem
    name::String
    system::AbstractSystem
    horizon::Int64

    expectation_method::ExpectationMethod
end