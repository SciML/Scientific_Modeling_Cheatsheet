#!/usr/bin/env julia

"""
Scientific Modeling Cheatsheet Generator

This script contains all code examples for MATLAB, Python, and Julia.
It tests each example, times execution, and generates the complete HTML cheatsheet.
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
using BenchmarkTools

# Color codes for terminal output
const GREEN = "\033[32m"
const RED = "\033[31m"
const YELLOW = "\033[33m"
const BLUE = "\033[34m"
const RESET = "\033[0m"

# Structure to hold example code
struct CodeExample
    section::String
    subsection::String
    description::String
    matlab::String
    python::String
    julia::String
end

# Store all examples
examples = CodeExample[]

# ============================================================================
# SECTION 1: BASIC OPERATIONS
# ============================================================================

push!(examples, CodeExample(
    "Basic Operations",
    "Vector/Matrix Creation",
    "Create arrays and matrices",
    # MATLAB
    """A = [1 2; 3 4]
v = [1; 2; 3]
zeros(3, 3)
ones(2, 4)
eye(3)""",
    # Python
    """import numpy as np
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2, 3])
np.zeros((3, 3))
np.ones((2, 4))
np.eye(3)""",
    # Julia
    """A = [1 2; 3 4]
v = [1, 2, 3]
zeros(3, 3)
ones(2, 4)
I(3)  # or Matrix(I, 3, 3)"""
))

push!(examples, CodeExample(
    "Basic Operations",
    "Eigenvalues",
    "Compute eigenvalues of a matrix",
    # MATLAB
    """A = [1 2; 3 4];
eigenvalues = eig(A)""",
    # Python
    """import numpy as np
A = np.array([[1, 2], [3, 4]])
eigenvalues = np.linalg.eigvals(A)""",
    # Julia
    """using LinearAlgebra
A = [1 2; 3 4]
eigenvalues = eigvals(A)"""
))

# ============================================================================
# SECTION 2: DIFFERENTIAL EQUATIONS
# ============================================================================

push!(examples, CodeExample(
    "Differential Equations",
    "Solving ODEs",
    "Solve dy/dt = -2y + sin(t)",
    # MATLAB
    """f = @(t, y) -2*y + sin(t);
[t, y] = ode45(f, [0 10], 1);""",
    # Python
    """from scipy.integrate import solve_ivp
import numpy as np

def f(t, y):
    return -2*y + np.sin(t)

sol = solve_ivp(f, [0, 10], [1.0])
t, y = sol.t, sol.y[0]""",
    # Julia
    """using DifferentialEquations

f(u, p, t) = -2*u + sin(t)
prob = ODEProblem(f, 1.0, (0.0, 10.0))
sol = solve(prob, Tsit5())"""
))

push!(examples, CodeExample(
    "Differential Equations",
    "Specifying Save Points",
    "Save solution at specific times",
    # MATLAB
    """tspan = 0:0.1:10;
[t, y] = ode45(f, tspan, 1);""",
    # Python
    """import numpy as np
t_eval = np.arange(0, 10.1, 0.1)
sol = solve_ivp(f, [0, 10], [1.0], t_eval=t_eval)""",
    # Julia
    """sol = solve(prob, saveat=0.1)
# Or with specific points:
t_save = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
sol = solve(prob, saveat=t_save)"""
))

push!(examples, CodeExample(
    "Differential Equations",
    "DAE Systems (ROBER Problem)",
    "Solve differential-algebraic equations with mass matrix",
    # MATLAB
    """% ROBER problem
M = [1 0 0; 0 1 0; 0 0 0];
function dydt = rober(t, y)
    k1 = 0.04; k2 = 3e7; k3 = 1e4;
    dydt = [-k1*y(1) + k3*y(2)*y(3);
            k1*y(1) - k2*y(2)^2 - k3*y(2)*y(3);
            y(1) + y(2) + y(3) - 1];
end
options = odeset('Mass', M, 'RelTol', 1e-4);
y0 = [1; 0; 0];
[t, y] = ode15s(@rober, [0 1e-5], y0, options);""",
    # Python
    """# Not directly supported in SciPy
# Reformulate as stiff ODE
from scipy.integrate import solve_ivp

def rober(t, y):
    y1, y2, y3 = y
    dy1 = -0.04*y1 + 1e4*y2*y3
    dy2 = 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2
    dy3 = -(dy1 + dy2)  # Algebraic constraint
    return [dy1, dy2, dy3]

sol = solve_ivp(rober, [0, 1e-5], [1, 0, 0],
                method='Radau', rtol=1e-4)""",
    # Julia
    """using DifferentialEquations

function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] = k₁*y₁ - k₂*y₂^2 - k₃*y₂*y₃
    du[3] = y₁ + y₂ + y₃ - 1
end

M = [1.0 0 0; 0 1.0 0; 0 0 0]
f = ODEFunction(rober, mass_matrix=M)
prob = ODEProblem(f, [1.0, 0, 0], (0.0, 1e-5),
                  [0.04, 3e7, 1e4])
sol = solve(prob, Rodas5())"""
))

# ============================================================================
# SECTION 3: NONLINEAR SOLVING
# ============================================================================

push!(examples, CodeExample(
    "Nonlinear Solving",
    "Root Finding",
    "Find root of x³ - 2x - 5 = 0",
    # MATLAB
    """f = @(x) x^3 - 2*x - 5;
x = fzero(f, 2)""",
    # Python
    """from scipy.optimize import fsolve

def f(x):
    return x**3 - 2*x - 5

x = fsolve(f, 2.0)[0]""",
    # Julia
    """using NonlinearSolve

f(u, p) = u^3 - 2*u - 5
prob = NonlinearProblem(f, 2.0)
sol = solve(prob, NewtonRaphson())
x = sol.u"""
))

# ============================================================================
# SECTION 4: OPTIMIZATION
# ============================================================================

push!(examples, CodeExample(
    "Optimization",
    "Minimize Function",
    "Minimize f(x) = x₁² + x₂²",
    # MATLAB
    """f = @(x) x(1)^2 + x(2)^2;
x0 = [1; 1];
x = fminunc(f, x0)""",
    # Python
    """from scipy.optimize import minimize

def f(x):
    return x[0]**2 + x[1]**2

x0 = [1.0, 1.0]
result = minimize(f, x0)
x = result.x""",
    # Julia
    """using Optimization, OptimizationOptimJL

f(x, p) = x[1]^2 + x[2]^2
optf = OptimizationFunction(f, AutoForwardDiff())
prob = OptimizationProblem(optf, [1.0, 1.0])
sol = solve(prob, Optim.LBFGS())
x = sol.u"""
))

# ============================================================================
# SECTION 5: AUTOMATIC DIFFERENTIATION
# ============================================================================

push!(examples, CodeExample(
    "Automatic Differentiation",
    "Forward Mode",
    "Compute derivative of f(x) = x² + sin(x) at x = 2",
    # MATLAB
    """% Use automatic differentiation toolbox
% or symbolic differentiation
syms x
f = x^2 + sin(x);
df = diff(f, x);
df_func = matlabFunction(df);
result = df_func(2)""",
    # Python
    """import torch

def f(x):
    return x**2 + torch.sin(x)

x = torch.tensor(2.0)
v = torch.tensor(1.0)
with torch.enable_grad():
    _, jvp = torch.func.jvp(f, (x,), (v,))
result = jvp.item()""",
    # Julia
    """using ForwardDiff

f(x) = x^2 + sin(x)
result = ForwardDiff.derivative(f, 2.0)"""
))

push!(examples, CodeExample(
    "Automatic Differentiation",
    "Reverse Mode (Gradients)",
    "Compute gradient of f(x) = x₁² + sin(x₂)",
    # MATLAB
    """% For multivariable functions
f = @(x) x(1)^2 + sin(x(2));
x0 = [2; 2];
% Use automatic differentiation
% or finite differences""",
    # Python
    """import torch

x = torch.tensor([2.0, 2.0], requires_grad=True)
y = x[0]**2 + torch.sin(x[1])
y.backward()
gradient = x.grad.numpy()""",
    # Julia
    """using ForwardDiff

f(x) = x[1]^2 + sin(x[2])
gradient = ForwardDiff.gradient(f, [2.0, 2.0])"""
))

# ============================================================================
# SECTION 6: SYMBOLIC COMPUTING
# ============================================================================

push!(examples, CodeExample(
    "Symbolic Computing",
    "Basic Symbolic Operations",
    "Symbolic differentiation",
    # MATLAB
    """syms x y
expr = x^2 + sin(x)
derivative = diff(expr, x)
expanded = expand((x + y)^3)
simplified = simplify(expr)""",
    # Python
    """import sympy as sp

x, y = sp.symbols('x y')
expr = x**2 + sp.sin(x)
derivative = sp.diff(expr, x)
expanded = sp.expand((x + y)**3)
simplified = sp.simplify(expr)""",
    # Julia
    """using Symbolics

@variables x y
expr = x^2 + sin(x)
derivative = Symbolics.derivative(expr, x)
expanded = expand((x + y)^3)
simplified = simplify(expr)"""
))

push!(examples, CodeExample(
    "Symbolic Computing",
    "Generate Functions",
    "Convert symbolic expressions to functions",
    # MATLAB
    """syms x y
expr = x^2 + y^2
f = matlabFunction(expr)""",
    # Python
    """import sympy as sp
import numpy as np

x, y = sp.symbols('x y')
expr = x**2 + y**2
f = sp.lambdify([x, y], expr, 'numpy')""",
    # Julia
    """using Symbolics

@variables x y
expr = x^2 + y^2
f_expr = build_function(expr, [x, y])
f = eval(f_expr[1])  # Out-of-place
f! = eval(f_expr[2])  # In-place"""
))

# ============================================================================
# SECTION 7: COMPONENT-BASED MODELING
# ============================================================================

push!(examples, CodeExample(
    "Component-Based Modeling",
    "System Definition",
    "Define a pendulum system",
    # MATLAB
    """% Use Simulink or write equations manually
function dydt = pendulum(t, y, g, L)
    x = y(1); y_pos = y(2);
    vx = y(3); vy = y(4);
    lambda = (vx^2 + vy^2 + g*y_pos)/L^2;

    dydt = [vx; vy; -lambda*x;
            -lambda*y_pos - g];
end""",
    # Python
    """import casadi as ca

# Define variables
x = ca.MX.sym('x')
y = ca.MX.sym('y')
vx = ca.MX.sym('vx')
vy = ca.MX.sym('vy')

# Parameters
g = 9.8
L = 1.0

# Would need integrator setup
# with algebraic constraints""",
    # Julia
    """using ModelingToolkit

@parameters t g=9.8 L=1.0
@variables x(t) y(t) vx(t) vy(t) λ(t)
D = Differential(t)

eqs = [D(x) ~ vx,
       D(y) ~ vy,
       D(vx) ~ -λ * x,
       D(vy) ~ -λ * y - g,
       x^2 + y^2 ~ L^2]

@named model = System(eqs, t)"""
))

push!(examples, CodeExample(
    "Component-Based Modeling",
    "Simplify and Compile",
    "Perform index reduction and simplification",
    # MATLAB
    """% Manual simplification or use
% Symbolic Math Toolbox""",
    # Python
    """# Manual index reduction required
# or use specialized DAE solvers""",
    # Julia
    """# Simplify system (index reduction)
simplified = mtkcompile(model)

# The simplified system can now be used
# with standard ODE solvers"""
))

# ============================================================================
# SECTION 8: NUMERICAL INTEGRATION
# ============================================================================

push!(examples, CodeExample(
    "Numerical Integration",
    "Quadrature",
    "Integrate exp(-x²) from 0 to 1",
    # MATLAB
    """f = @(x) exp(-x.^2);
result = integral(f, 0, 1)""",
    # Python
    """import numpy as np
from scipy.integrate import quad

def f(x):
    return np.exp(-x**2)

result, error = quad(f, 0, 1)""",
    # Julia
    """using Integrals

f(x, p) = exp(-x^2)
prob = IntegralProblem(f, 0.0, 1.0)
sol = solve(prob, QuadGKJL())
result = sol.u"""
))

# ============================================================================
# TEST AND TIMING FUNCTIONS
# ============================================================================

function test_julia_code(code::String)
    """Test and time Julia code snippet"""
    try
        # Clean up code for evaluation
        clean_code = replace(code, r"^\s*using\s+.*$"m => "")

        # Time the execution
        t_start = time()
        result = eval(Meta.parse("begin\n$clean_code\nend"))
        t_elapsed = time() - t_start

        return (success=true, result=result, time=t_elapsed)
    catch e
        return (success=false, error=string(e), time=0.0)
    end
end

function test_python_code(code::String)
    """Test and time Python code snippet"""
    try
        # Write to temp file and execute
        tmpfile = tempname() * ".py"
        open(tmpfile, "w") do f
            write(f, code)
            write(f, "\n# Print last result for capture\n")
            write(f, "import sys\n")
            write(f, "if 'result' in locals(): print(result)\n")
            write(f, "elif 'x' in locals(): print(x)\n")
            write(f, "elif 'sol' in locals() and hasattr(sol, 'y'): print(sol.y[:,-1] if len(sol.y.shape) > 1 else sol.y[-1])\n")
            write(f, "elif 'eigenvalues' in locals(): print(eigenvalues)\n")
            write(f, "elif 'gradient' in locals(): print(gradient)\n")
            write(f, "elif 'derivative' in locals(): print(derivative)\n")
        end

        t_start = time()
        output = read(`python3 $tmpfile`, String)
        t_elapsed = time() - t_start

        rm(tmpfile)
        return (success=true, result=strip(output), time=t_elapsed)
    catch e
        return (success=false, error=string(e), time=0.0)
    end
end

function test_matlab_code(code::String)
    """Test and time MATLAB code snippet"""
    # Check if MATLAB is available
    try
        run(`matlab -version`)
    catch
        return (success=false, error="MATLAB not available", time=0.0)
    end

    try
        tmpfile = tempname() * ".m"
        open(tmpfile, "w") do f
            write(f, code)
        end

        t_start = time()
        output = read(`matlab -batch "run('$tmpfile')"`, String)
        t_elapsed = time() - t_start

        rm(tmpfile)
        return (success=true, result=strip(output), time=t_elapsed)
    catch e
        return (success=false, error=string(e), time=0.0)
    end
end

# ============================================================================
# HTML GENERATION
# ============================================================================

function generate_html(examples::Vector{CodeExample}, test_results)
    """Generate the complete HTML cheatsheet"""

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MATLAB – Python – Julia Quick Reference for Scientific Modeling</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        .code-block {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }
        .code-block h4 {
            margin-top: 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        pre {
            margin: 0;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            line-height: 1.4;
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }
        code {
            background-color: #f8f8f8;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .matlab { border-left: 4px solid #ff6b00; }
        .python { border-left: 4px solid #3776ab; }
        .julia { border-left: 4px solid #9558b2; }
        .note {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .warning {
            background-color: #f8d7da;
            border: 1px solid #dc3545;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 40px;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
        }
        .success { color: #27ae60; }
        .failure { color: #e74c3c; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>MATLAB – Python – Julia Quick Reference for Scientific Modeling</h1>

    <div class="note">
        <strong>Note:</strong> This cheatsheet compares implementations across MATLAB, Python (NumPy/SciPy/PyTorch), and Julia for common scientific computing tasks.
        Generated on $(Dates.now())
    </div>
"""

    # Group examples by section
    sections = Dict{String, Vector{CodeExample}}()
    for ex in examples
        if !haskey(sections, ex.section)
            sections[ex.section] = CodeExample[]
        end
        push!(sections[ex.section], ex)
    end

    # Generate HTML for each section
    for section in sort(collect(keys(sections)))
        html *= "\n    <h2>$(section)</h2>\n"

        # Group by subsection
        subsections = Dict{String, Vector{CodeExample}}()
        for ex in sections[section]
            if !haskey(subsections, ex.subsection)
                subsections[ex.subsection] = CodeExample[]
            end
            push!(subsections[ex.subsection], ex)
        end

        for subsection in sort(collect(keys(subsections)))
            html *= "\n    <h3>$(subsection)</h3>\n"

            for ex in subsections[subsection]
                html *= """
    <p><em>$(ex.description)</em></p>
    <div class="comparison-grid">
        <div class="code-block matlab">
            <h4>MATLAB</h4>
            <pre>$(ex.matlab)</pre>
        </div>
        <div class="code-block python">
            <h4>Python</h4>
            <pre>$(ex.python)</pre>
        </div>
        <div class="code-block julia">
            <h4>Julia</h4>
            <pre>$(ex.julia)</pre>
        </div>
    </div>
"""
            end
        end
    end

    # Add warnings and notes
    html *= """

    <div class="warning">
        <h3>⚠️ Python Compatibility Warning</h3>
        <ul>
            <li><strong>SymPy</strong> symbolic objects are incompatible with NumPy arrays and PyTorch tensors</li>
            <li><strong>PyTorch</strong>, <strong>TensorFlow</strong>, and <strong>JAX</strong> use incompatible array types</li>
            <li><strong>SciPy</strong> lacks native DAE support (use Assimulo or CasADi for DAEs)</li>
            <li>Each automatic differentiation system is isolated from others</li>
        </ul>
    </div>

    <div class="note">
        <h3>📝 Julia Unified Ecosystem</h3>
        <ul>
            <li>ModelingToolkit integrates with all DifferentialEquations.jl solvers</li>
            <li>Automatic differentiation works seamlessly across packages</li>
            <li>Symbolic and numeric computing can be mixed naturally</li>
            <li>Component-based modeling with automatic index reduction</li>
        </ul>
    </div>

    <div class="note">
        <h3>🔄 ModelingToolkit v10 Changes</h3>
        <ul>
            <li>All system types unified as <code>System</code> (no more <code>ODESystem</code>, <code>NonlinearSystem</code>)</li>
            <li>Use <code>mtkcompile()</code> instead of <code>structural_simplify()</code></li>
            <li>Automatic index reduction for high-index DAEs</li>
        </ul>
    </div>
"""

    # Add test results summary
    if !isempty(test_results)
        passed = count(r -> r.julia_success || r.python_success, values(test_results))
        total = length(test_results)

        html *= """

    <div class="timestamp">
        <h3>Test Results Summary</h3>
        <p>Tests passed: <span class="success">$(passed)</span> / $(total)</p>
        <table>
            <thead>
                <tr>
                    <th>Example</th>
                    <th>Julia</th>
                    <th>Python</th>
                    <th>MATLAB</th>
                    <th>Julia Time (ms)</th>
                    <th>Python Time (ms)</th>
                </tr>
            </thead>
            <tbody>
"""

        for (name, result) in test_results
            julia_status = result.julia_success ? "✓" : "✗"
            python_status = result.python_success ? "✓" : "✗"
            matlab_status = get(result, :matlab_success, false) ? "✓" : "N/A"

            julia_class = result.julia_success ? "success" : "failure"
            python_class = result.python_success ? "success" : "failure"

            julia_time = round(result.julia_time * 1000, digits=2)
            python_time = round(result.python_time * 1000, digits=2)

            html *= """
                <tr>
                    <td>$(name)</td>
                    <td class="$(julia_class)">$(julia_status)</td>
                    <td class="$(python_class)">$(python_status)</td>
                    <td>$(matlab_status)</td>
                    <td>$(julia_time)</td>
                    <td>$(python_time)</td>
                </tr>
"""
        end

        html *= """
            </tbody>
        </table>
        <hr>
        <p>Generated: $(Dates.now())</p>
        <p>Julia Version: $(VERSION)</p>
    </div>
"""
    end

    html *= """
</body>
</html>
"""

    return html
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    println("$(BLUE)=" ^ 60)
    println("SCIENTIFIC MODELING CHEATSHEET GENERATOR")
    println("=" ^ 60 * RESET)

    test_results = Dict{String, Any}()

    # Test each example
    println("\n$(BLUE)Testing Examples...$(RESET)")
    for (i, example) in enumerate(examples)
        name = "$(example.section): $(example.subsection)"
        println("\n[$i/$(length(examples))] Testing: $name")

        result = Dict{Symbol, Any}()

        # Test Julia code
        print("  Julia... ")
        julia_result = test_julia_code(example.julia)
        result[:julia_success] = julia_result.success
        result[:julia_time] = julia_result.time
        if julia_result.success
            println("$(GREEN)✓$(RESET) ($(round(julia_result.time*1000, digits=2))ms)")
        else
            println("$(RED)✗$(RESET) $(julia_result.error)")
        end

        # Test Python code
        print("  Python... ")
        python_result = test_python_code(example.python)
        result[:python_success] = python_result.success
        result[:python_time] = python_result.time
        if python_result.success
            println("$(GREEN)✓$(RESET) ($(round(python_result.time*1000, digits=2))ms)")
        else
            println("$(RED)✗$(RESET)")
        end

        # Test MATLAB code (if available)
        matlab_result = test_matlab_code(example.matlab)
        if matlab_result.success || matlab_result.error != "MATLAB not available"
            print("  MATLAB... ")
            result[:matlab_success] = matlab_result.success
            result[:matlab_time] = matlab_result.time
            if matlab_result.success
                println("$(GREEN)✓$(RESET) ($(round(matlab_result.time*1000, digits=2))ms)")
            else
                println("$(RED)✗$(RESET)")
            end
        end

        test_results[name] = result
    end

    # Generate HTML
    println("\n$(BLUE)Generating HTML...$(RESET)")
    html_content = generate_html(examples, test_results)

    # Save HTML file
    open("scientific_modeling_cheatsheet.html", "w") do f
        write(f, html_content)
    end

    println("$(GREEN)✓ HTML generated: scientific_modeling_cheatsheet.html$(RESET)")

    # Summary
    passed = count(r -> r.julia_success && r.python_success, values(test_results))
    total = length(test_results)

    println("\n$(BLUE)=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60 * RESET)
    println("Tests passed: $(GREEN)$passed/$total$(RESET)")

    if passed < total
        println("\n$(YELLOW)Failed tests:$(RESET)")
        for (name, result) in test_results
            if !result.julia_success || !result.python_success
                println("  - $name")
            end
        end
        return 1
    end

    return 0
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end