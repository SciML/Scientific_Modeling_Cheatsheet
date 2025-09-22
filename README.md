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
  - System definition
  - Automatic simplification and index reduction
  - Acausal component modeling

- **Numerical Integration**
  - Quadrature methods
  - Adaptive integration

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

## 🧪 Testing

Test scripts are available in the `test/` directory:
- `test_julia_examples.jl` - Tests all Julia code examples
- `test_python_examples.py` - Tests all Python code examples

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with:
- Additional examples
- Corrections or improvements
- New topics relevant to scientific computing

## 🔗 Resources

- [Julia Documentation](https://docs.julialang.org)
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
- [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MATLAB Documentation](https://www.mathworks.com/help/matlab/)