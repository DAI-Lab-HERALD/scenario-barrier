using BarrierFunctions

using Distributions, PDMats

# x[1] = longitudinal coordinate in meters
# x[2] = lateral coordinate in meters

A = Matrix{Float64}[
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ],
    [
        1 0;
        0 1;
    ]
]
b = Vector{Float64}[
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [50 * (1000 / 3600), 0], # 50 km/h in m/s (assuming 1s time steps)
    [50 * (1000 / 3600), 0]
]

Q = ConvexPolytope{Float64}[
    ConvexPolytope(
        H = Float64[0 -1],
        h = [-2.5]
    ),
    ConvexPolytope(
        H = Float64[0 1],
        h = [-2.5]
    ),
    ConvexPolytope(
        H = Float64[
            0  1;
            0 -1;
            1  0
        ],
        h = [2.5, 2.5, 0.0]
    ),
    ConvexPolytope(
        H = Float64[
             0  1;
             0 -1;
            -1  0
        ],
        h = [2.5, 2.5, -100]
    ),
    ConvexPolytope(
        H = Float64[
             0  1;
             0 -1;
             1  0;
            -1  0
        ],
        h = [2.5, 2.5, 10, 0]
    ),
    ConvexPolytope(
        H = Float64[
             0  1;
             0 -1;
             1  0;
            -1  0
        ],
        h = [2.5, 2.5, 100, -10]
    ),
]

μ = [0, 0]
Σ = PDiagMat([0.1, 0.1])
η = DiagNormal(μ, Σ)

# Start in a box [0m, 10m] × [-2.5m, 2.5m]
X0 = ConvexPolytope(
    H = Float64[
        1  0;
        -1 0;
        0  1;
        0 -1
    ],
    h = [10, 0, 2.5, 2.5]
)

# On the road is safe
Xs = ConvexPolytope(
    H = Float64[
        0  1;
        0 -1
    ],
    h = [2.5, 2.5]
)

# Outside the road is dangerous
Xu = [
    ConvexPolytope(
        H = Float64[0 -1],
        h = [-2.5]
    ),
    ConvexPolytope(
        H = Float64[0 1],
        h = [-2.5]
    )
]

car_with_sidewind = System{2}(A, b, Q, η, X0, Xs, Xu)