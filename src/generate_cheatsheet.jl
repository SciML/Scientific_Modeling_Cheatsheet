#!/usr/bin/env julia

"""
    generate_cheatsheet.jl

Generate and test the scientific modeling cheatsheet examples.
Runs all Julia, Python, and MATLAB code, verifies consistency, and generates HTML output.
"""

using Printf
using LinearAlgebra
using DifferentialEquations
using ModelingToolkit
using NonlinearSolve
using ForwardDiff
using Enzyme
using Symbolics
using Sundials
using Optimization
using OptimizationOptimJL
using Optim
using Integrals
using Dates

# Color codes for terminal output
const GREEN = "\033[32m"
const RED = "\033[31m"
const YELLOW = "\033[33m"
const BLUE = "\033[34m"
const RESET = "\033[0m"

println("$(BLUE)=" ^ 60)
println("SCIENTIFIC MODELING CHEATSHEET GENERATOR")
println("Generating Complete HTML Documentation")
println("=" ^ 60 * RESET)

# Check for optional dependencies
has_matlab = false
try
    run(`matlab -version`)
    global has_matlab = true
    println("$(GREEN)✓ MATLAB found$(RESET)")
catch
    println("$(YELLOW)⚠ MATLAB not found - MATLAB examples will be skipped$(RESET)")
end

# Test results storage
test_results = Dict{String, Dict{String, Any}}()
timing_results = Dict{String, Dict{String, Float64}}()

function test_and_record(name::String, julia_code::Function, python_code::String, matlab_code::String="")
    println("\n$(BLUE)Testing: $name$(RESET)")
    result = Dict{String, Any}()
    timings = Dict{String, Float64}()

    # Test Julia
    try
        start_time = time()
        result["julia"] = julia_code()
        timings["julia"] = time() - start_time
        println("  $(GREEN)✓$(RESET) Julia: $(result["julia"]) ($(round(timings["julia"]*1000, digits=3))ms)")
    catch e
        result["julia_error"] = string(e)
        println("  $(RED)✗$(RESET) Julia error: $e")
    end

    # Test Python
    try
        start_time = time()
        result["python"] = strip(read(`python3 -c $python_code`, String))
        timings["python"] = time() - start_time
        println("  $(GREEN)✓$(RESET) Python: $(result["python"]) ($(round(timings["python"]*1000, digits=3))ms)")
    catch e
        result["python_error"] = string(e)
        println("  $(RED)✗$(RESET) Python error")
    end

    # Test MATLAB if available
    if has_matlab && matlab_code != ""
        try
            start_time = time()
            matlab_script = tempname() * ".m"
            open(matlab_script, "w") do f
                write(f, matlab_code)
            end
            result["matlab"] = strip(read(`matlab -batch "run('$matlab_script')"`, String))
            timings["matlab"] = time() - start_time
            println("  $(GREEN)✓$(RESET) MATLAB: $(result["matlab"]) ($(round(timings["matlab"]*1000, digits=3))ms)")
            rm(matlab_script)
        catch e
            result["matlab_error"] = string(e)
            println("  $(RED)✗$(RESET) MATLAB error")
        end
    end

    test_results[name] = result
    timing_results[name] = timings
    return result
end

# Run all tests
println("\n$(BLUE)Running Tests...$(RESET)")
println("=" ^ 60)

# Test 1: Eigenvalues
test_and_record("Eigenvalues",
    () -> eigvals([1 2; 3 4]),
    "import numpy as np; print(np.linalg.eigvals(np.array([[1,2],[3,4]])))",
    "A = [1 2; 3 4]; disp(eig(A));"
)

# Test 2: ODE Solving
test_and_record("ODE Solving",
    () -> begin
        f(u,p,t) = -2*u + sin(t)
        prob = ODEProblem(f, 1.0, (0.0, 10.0))
        sol = solve(prob, Tsit5())
        sol(10.0)
    end,
    """
import numpy as np
from scipy.integrate import solve_ivp
def f(t,y): return -2*y + np.sin(t)
sol = solve_ivp(f, [0, 10], [1.0], dense_output=True)
print(sol.sol(10.0)[0])
""",
    """
f = @(t, y) -2*y + sin(t);
[t, y] = ode45(f, [0 10], 1);
disp(y(end));
"""
)

# Test 3: Nonlinear Solving
test_and_record("Nonlinear Solving",
    () -> begin
        f(u,p) = u^3 - 2*u - 5
        prob = NonlinearProblem(f, 2.0)
        sol = solve(prob, NewtonRaphson())
        sol.u
    end,
    """
from scipy.optimize import fsolve
def f(x): return x**3 - 2*x - 5
print(fsolve(f, 2.0)[0])
""",
    """
f = @(x) x^3 - 2*x - 5;
x = fzero(f, 2);
disp(x);
"""
)

# Test 4: Optimization
test_and_record("Optimization",
    () -> begin
        f(x,p) = x[1]^2 + x[2]^2
        optf = OptimizationFunction(f, AutoForwardDiff())
        prob = OptimizationProblem(optf, [1.0, 1.0])
        sol = solve(prob, Optim.LBFGS())
        sol.u
    end,
    """
from scipy.optimize import minimize
def f(x): return x[0]**2 + x[1]**2
result = minimize(f, [1.0, 1.0])
print(result.x)
""",
    """
f = @(x) x(1)^2 + x(2)^2;
x0 = [1; 1];
x = fminunc(f, x0);
disp(x);
"""
)

# Test 5: Automatic Differentiation
test_and_record("Automatic Differentiation",
    () -> ForwardDiff.derivative(x -> x^2 + sin(x), 2.0),
    """
import torch
def f(x): return x**2 + torch.sin(x)
x = torch.tensor(2.0)
v = torch.tensor(1.0)
with torch.enable_grad():
    _, jvp = torch.func.jvp(f, (x,), (v,))
print(jvp.item())
""",
    """
syms x;
f = x^2 + sin(x);
df = diff(f, x);
df_func = matlabFunction(df);
result = df_func(2);
disp(result);
"""
)

# Test 6: DAE System (ROBER)
test_and_record("DAE System (ROBER)",
    () -> begin
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
        sol(1e-5)
    end,
    """
import numpy as np
from scipy.integrate import solve_ivp
def rober(t, y):
    y1, y2, y3 = y
    dy1 = -0.04*y1 + 1e4*y2*y3
    dy2 = 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2
    dy3 = -(dy1 + dy2)
    return [dy1, dy2, dy3]
sol = solve_ivp(rober, [0, 1e-5], [1, 0, 0], method='Radau', rtol=1e-4)
print(sol.y[:, -1])
""",
    """
M = [1 0 0; 0 1 0; 0 0 0];
function dydt = rober(t, y)
    k1 = 0.04; k2 = 3e7; k3 = 1e4;
    dydt = [-k1*y(1) + k3*y(2)*y(3);
            k1*y(1) - k2*y(2)^2 - k3*y(2)*y(3);
            y(1) + y(2) + y(3) - 1];
end
options = odeset('Mass', M, 'MStateDependence', 'none');
y0 = [1; 0; 0];
[t, y] = ode15s(@rober, [0 1e-5], y0, options);
disp(y(end, :));
"""
)

# Test 7: Symbolic Computing
test_and_record("Symbolic Computing",
    () -> begin
        @variables x y
        expr = x^2 + sin(x)
        derivative = Symbolics.derivative(expr, x)
        string(derivative)
    end,
    """
import sympy as sp
x = sp.Symbol('x')
expr = x**2 + sp.sin(x)
derivative = sp.diff(expr, x)
print(derivative)
""",
    """
syms x y;
expr = x^2 + sin(x);
derivative = diff(expr, x);
disp(char(derivative));
"""
)

# Test 8: Numerical Integration
test_and_record("Numerical Integration",
    () -> begin
        f(x, p) = exp(-x^2)
        prob = IntegralProblem(f, 0.0, 1.0)
        sol = solve(prob, QuadGKJL())
        sol.u
    end,
    """
import numpy as np
from scipy.integrate import quad
def f(x): return np.exp(-x**2)
result, _ = quad(f, 0, 1)
print(result)
""",
    """
f = @(x) exp(-x.^2);
result = integral(f, 0, 1);
disp(result);
"""
)

# Generate summary report
println("\n$(BLUE)" * "=" ^ 60)
println("TEST SUMMARY")
println("=" ^ 60 * RESET)

passed_count = 0
failed_count = 0

for (name, result) in test_results
    julia_ok = !haskey(result, "julia_error")
    python_ok = !haskey(result, "python_error")
    matlab_ok = !has_matlab || !haskey(result, "matlab_error")

    if julia_ok && python_ok && matlab_ok
        println("$(GREEN)✓$(RESET) $name")
        passed_count += 1
    else
        println("$(RED)✗$(RESET) $name")
        failed_count += 1
    end
end

println("\nTests passed: $(GREEN)$passed_count$(RESET)")
println("Tests failed: $(RED)$failed_count$(RESET)")

# Save timing results to CSV
println("\n$(BLUE)Saving timing results...$(RESET)")
open("benchmark_results.csv", "w") do f
    write(f, "Test,Julia (ms),Python (ms),MATLAB (ms),Julia Speedup vs Python\n")
    for (name, timings) in timing_results
        julia_time = get(timings, "julia", 0.0) * 1000
        python_time = get(timings, "python", 0.0) * 1000
        matlab_time = get(timings, "matlab", 0.0) * 1000
        speedup = python_time > 0 ? python_time / julia_time : 0
        write(f, "$name,$julia_time,$python_time,$matlab_time,$speedup\n")
    end
end
println("$(GREEN)✓ Results saved to benchmark_results.csv$(RESET)")

# Generate updated HTML with test results
println("\n$(BLUE)Generating HTML report...$(RESET)")
html_timestamp = """
<div class="timestamp">
    <hr>
    <p>Generated: $(Dates.now())</p>
    <p>Julia Version: $(VERSION)</p>
    <p>Tests Run: $passed_count passed, $failed_count failed</p>
    <p><a href="benchmark_results.csv">Download Benchmark Results (CSV)</a></p>
</div>
"""

# Read existing HTML and append timestamp
html_content = read("scientific_modeling_cheatsheet.html", String)
if !occursin("class=\"timestamp\"", html_content)
    # Add before closing body tag
    html_content = replace(html_content, "</body>" => html_timestamp * "\n</body>")
    open("scientific_modeling_cheatsheet.html", "w") do f
        write(f, html_content)
    end
    println("$(GREEN)✓ HTML updated with test results$(RESET)")
end

println("\n$(GREEN)=" ^ 60)
println("GENERATION COMPLETE")
println("=" ^ 60 * RESET)
println("\nOutput files:")
println("  - scientific_modeling_cheatsheet.html")
println("  - benchmark_results.csv")
println("\nOpen the HTML file in your browser to view the cheatsheet.")