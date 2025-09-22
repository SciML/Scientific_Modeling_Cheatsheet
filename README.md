# Scientific Modeling Cheatsheet

A comprehensive quick reference guide comparing MATLAB, Python, and Julia for scientific computing and modeling tasks.

## 📚 Overview

This cheatsheet provides side-by-side comparisons of common scientific computing operations across three major platforms:
- **MATLAB** - The traditional choice for engineering and scientific computing
- **Python** - Using NumPy, SciPy, PyTorch, and SymPy for scientific computing
- **Julia** - Modern high-performance scientific computing with unified ecosystem

## 🚀 Quick Start

### View the Cheatsheet
Open `scientific_modeling_cheatsheet.html` in your web browser for the full interactive reference guide.

### Topics Covered

#### Basic Operations
- Vector/Matrix creation and manipulation
- Linear algebra operations (eigenvalues, decompositions)
- Array indexing and slicing
- Mathematical operations

#### Scientific Computing
- **Differential Equations**
  - ODEs (Ordinary Differential Equations)
  - DAEs (Differential-Algebraic Equations)
  - Mass matrix formulations
  - Stiff systems

- **Nonlinear Solving**
  - Root finding
  - Systems of nonlinear equations

- **Optimization**
  - Unconstrained optimization
  - Gradient-based methods

- **Automatic Differentiation**
  - Forward mode
  - Reverse mode (gradients)
  - Comparison of different AD systems

- **Symbolic Computing**
  - Symbolic math operations
  - Code generation from symbolic expressions

- **Component-Based Modeling**
  - System definition with ModelingToolkit (Julia)
  - Automatic simplification and index reduction

- **Numerical Integration**
  - Quadrature methods
  - Adaptive integration

## 🔧 Installation

### Julia Packages
```julia
using Pkg
Pkg.add([
    "DifferentialEquations",
    "ModelingToolkit",
    "NonlinearSolve",
    "Optimization",
    "OptimizationOptimJL",
    "ForwardDiff",
    "Enzyme",
    "Symbolics",
    "Sundials",
    "Integrals"
])
```

### Python Packages
```bash
pip install numpy scipy sympy torch matplotlib
```

### MATLAB
Requires MATLAB with:
- Optimization Toolbox
- Symbolic Math Toolbox (optional)

## ⚠️ Important Notes

### Python Ecosystem Fragmentation
The Python scientific computing ecosystem has compatibility issues between different libraries:
- **SymPy** symbolic objects are incompatible with NumPy arrays and PyTorch tensors
- **PyTorch**, **TensorFlow**, and **JAX** use incompatible array types
- **SciPy** lacks native DAE support (use Assimulo or CasADi for DAEs)
- Each automatic differentiation system is isolated from others

### Julia Unified Ecosystem
Julia provides a more unified experience:
- ModelingToolkit integrates with all DifferentialEquations.jl solvers
- Automatic differentiation works seamlessly across packages
- Symbolic and numeric computing can be mixed naturally

### ModelingToolkit v10 Changes
Recent updates to ModelingToolkit (v10) include:
- All system types unified as `System` (no more `ODESystem`, `NonlinearSystem`)
- Use `mtkcompile()` instead of `structural_simplify()`
- Automatic index reduction for high-index DAEs

## 📊 Examples

### Simple ODE Example

**Julia:**
```julia
using DifferentialEquations
f(u, p, t) = -2*u + sin(t)
prob = ODEProblem(f, 1.0, (0.0, 10.0))
sol = solve(prob, Tsit5())
```

**Python:**
```python
from scipy.integrate import solve_ivp
import numpy as np

def f(t, y):
    return -2*y + np.sin(t)

sol = solve_ivp(f, [0, 10], [1.0])
```

**MATLAB:**
```matlab
f = @(t, y) -2*y + sin(t);
[t, y] = ode45(f, [0 10], 1);
```

### DAE Example (ROBER Problem)

The ROBER problem is a standard stiff DAE benchmark:

**Julia:**
```julia
function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] = k₁*y₁ - k₂*y₂^2 - k₃*y₂*y₃
    du[3] = y₁ + y₂ + y₃ - 1  # Algebraic constraint
end

M = [1.0 0 0; 0 1.0 0; 0 0 0]  # Singular mass matrix
f_ode = ODEFunction(rober, mass_matrix=M)
prob = ODEProblem(f_ode, [1.0, 0, 0], (0.0, 1e-5), [0.04, 3e7, 1e4])
sol = solve(prob, Rodas5())
```

## 🧪 Testing

Test scripts are available in the `test/` directory:
- `test_julia_examples.jl` - Tests all Julia code examples
- `test_python_examples.py` - Tests all Python code examples

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with:
- Additional examples
- Corrections or improvements
- New topics relevant to scientific computing

## 📄 License

MIT License

## 🔗 Resources

- [Julia Documentation](https://docs.julialang.org)
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
- [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MATLAB Documentation](https://www.mathworks.com/help/matlab/)

## 📈 Benchmarks

Performance comparisons show Julia typically achieves:
- 10-1000x speedup over Python for numerical operations
- Performance comparable to or better than MATLAB
- Native performance for automatic differentiation

Exact benchmarks depend on specific operations and problem sizes.

## 🏗️ Roadmap

Future additions planned:
- [ ] Partial differential equations (PDEs)
- [ ] Machine learning integration
- [ ] Parallel computing comparisons
- [ ] GPU acceleration examples
- [ ] More advanced optimization problems
- [ ] Uncertainty quantification
