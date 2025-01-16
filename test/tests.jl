using TestItems

@testitem "basic rates" begin
    for b in 0:0.2:1
        @test SIRLT.P₁₁([[1, 2], [1, 2], [1, 2]], b) ≈ 1
        @test SIRLT.P₁₁([[2, 1], [1, 2]], b) ≈ 1-b
    end
    @test SIRLT.P₁₁([[2, 1], [2], [3, 2, 1], [5,5,5,6]], 0.5) ≈ 0.25

    for a in 0:0.2:1
        @test SIRLT.linearC([[2, 2]], a) ≈ 1
        @test SIRLT.linearC([[0, 2]], a) ≈ 1-a
        @test SIRLT.linearC([[2, 3]], a) ≈ a
    end
    @test SIRLT.linearC([[1, 2], [2,2], [2, 2], [2,2]], 0.25) ≈ 0.9375
end

@testitem "β estimation" begin
    using Random
    using Graphs: barabasi_albert, grid
    using DifferentialEquations: DiscreteProblem, solve, FunctionMap


    Random.seed!(1234)
    # Failure in these tests may both be due to a mistake in β_estimation or
    # in the dynamics that incorporate α and Θ.
    p = (
        β=0.3,
        μ=0.1,
        ν=0.,
        Θ=1.0,
        α=0.5
    )
    n = 50
    gg = barabasi_albert(n^2, 3)
    #gg = grid((n, n); periodic=true)
    nd = SIRLT_dynamics(gg);
    x₀ = init(n^2; nb_I=3, prop_A=0.5);

    prob = DiscreteProblem(nd, x₀, (1, 100), p)
    sol = solve(prob, FunctionMap(); callback=SIR_termination_cb)
    @test isapprox(β_estimation(sol, gg), p.β; atol=0.01)

    p = (
        β=0.3,
        μ=0.1,
        ν=0.,
        Θ=0.0,
        α=0.5
    )

    prob = DiscreteProblem(nd, x₀, (1, 100), p)
    sol = solve(prob, FunctionMap(); callback=SIR_termination_cb)
    @test β_estimation(sol, gg) ≈ 0.0

    p = (
        β=0.3,
        μ=0.1,
        ν=0.,
        Θ=0.5,
        α=0.0
    )

    x₀ = init(n^2; nb_I=3, prop_A=0);
    prob = DiscreteProblem(nd, x₀, (1, 100), p)
    sol = solve(prob, FunctionMap(); callback=SIR_termination_cb)
    @test isapprox(β_estimation(sol, gg), .28; atol=0.01)
    p = (
        β=0.3,
        μ=0.1,
        ν=0.,
        Θ=0.5,
        α=0.0
    )

    x₀ = init(n^2; nb_I=3, prop_A=1);
    prob = DiscreteProblem(nd, x₀, (1, 100), p)
    sol = solve(prob, FunctionMap(); callback=SIR_termination_cb)
    @test β_estimation(sol, gg) ≈ 0.
end
