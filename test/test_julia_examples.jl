#!/usr/bin/env julia

"""
Test all examples by running the generator
"""

# Simply run the generator which tests all code

@testset "Scientific Modeling Cheatsheet - Julia Tests" begin

    @testset "Basic Operations" begin
        @testset "Eigenvalues" begin
            A = [1 2; 3 4]
            eigvals_result = eigvals(A)
            @test length(eigvals_result) == 2
            @test eigvals_result ≈ [-0.3722813232690143, 5.372281323269014]
        end
    end

    @testset "Differential Equations" begin
        @testset "Simple ODE" begin
            f(u, p, t) = -2*u + sin(t)
            prob = ODEProblem(f, 1.0, (0.0, 10.0))
            sol = solve(prob, Tsit5())
            @test sol(10.0) ≈ -0.049826692247477745 rtol=1e-5
        end

        @testset "ODE with saveat" begin
            f(u, p, t) = -2*u + sin(t)
            prob = ODEProblem(f, 1.0, (0.0, 10.0))
            sol = solve(prob, saveat=0.1)
            @test length(sol.t) == 101

            t_save = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
            sol2 = solve(prob, saveat=t_save)
            @test sol2.t == t_save
        end

        @testset "ROBER DAE" begin
            function rober(du, u, p, t)
                y₁, y₂, y₃ = u
                k₁, k₂, k₃ = p
                du[1] = -k₁*y₁ + k₃*y₂*y₃
                du[2] = k₁*y₁ - k₂*y₂^2 - k₃*y₂*y₃
                du[3] = y₁ + y₂ + y₃ - 1
            end

            M = [1.0 0 0; 0 1.0 0; 0 0 0]
            f = ODEFunction(rober, mass_matrix=M)
            prob = ODEProblem(f, [1.0, 0, 0], (0.0, 1e-5), [0.04, 3e7, 1e4])
            sol = solve(prob, Rodas5())

            final_state = sol(1e-5)
            @test final_state[1] ≈ 1.0 atol=0.01
            @test sum(final_state) ≈ 1.0 atol=1e-6  # Mass conservation
        end
    end

    @testset "Nonlinear Solving" begin
        f(u, p) = u^3 - 2*u - 5
        prob = NonlinearProblem(f, 2.0)
        sol = solve(prob, NewtonRaphson())
        @test sol.u ≈ 2.0945514815423265 rtol=1e-6
    end

    @testset "Optimization" begin
        f(x, p) = x[1]^2 + x[2]^2
        optf = OptimizationFunction(f, AutoForwardDiff())
        prob = OptimizationProblem(optf, [1.0, 1.0])
        sol = solve(prob, Optim.LBFGS())
        @test sol.u ≈ [0.0, 0.0] atol=1e-6
    end

    @testset "Automatic Differentiation" begin
        @testset "ForwardDiff" begin
            f(x) = x^2 + sin(x)
            result = ForwardDiff.derivative(f, 2.0)
            @test result ≈ 3.5838531634528574 rtol=1e-6
        end

        @testset "ForwardDiff Gradient" begin
            f(x) = x[1]^2 + sin(x[2])
            grad = ForwardDiff.gradient(f, [2.0, 2.0])
            @test grad[1] ≈ 4.0
            @test grad[2] ≈ cos(2.0)
        end
    end

    @testset "Symbolic Computing" begin
        @variables x y
        expr = x^2 + sin(x)
        derivative = Symbolics.derivative(expr, x)
        @test string(derivative) == "2x + cos(x)"
    end

    @testset "Numerical Integration" begin
        f(x, p) = exp(-x^2)
        prob = IntegralProblem(f, 0.0, 1.0)
        sol = solve(prob, QuadGKJL())
        @test sol.u ≈ 0.746824132812427 rtol=1e-6
    end

    @testset "ModelingToolkit" begin
        @parameters t g=9.8 L=1.0
        @variables x(t) y(t) vx(t) vy(t) λ(t)
        D = Differential(t)

        eqs = [D(x) ~ vx,
               D(y) ~ vy,
               D(vx) ~ -λ * x,
               D(vy) ~ -λ * y - g,
               x^2 + y^2 ~ L^2]

        @named model = System(eqs, t)
        @test length(equations(model)) == 5
    end

end

println("\n✅ All Julia tests passed!")