# NR1 JAX - Numerical Relativity in Python with JAX

This is a Python/JAX implementation of the NR1 numerical relativity code forked from 20K on Github.

## Features

- BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation for 3+1 numerical relativity
- JIT-compiled finite difference operators for performance
- Clean, readable Python implementation
- Modular design with separate components for:
  - BSSN evolution equations
  - Finite difference derivatives
  - Initial data setup
  - Kreiss-Oliger dissipation
  - Error analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main.py
```

## Structure

- `main.py` - Main simulation loop and setup
- `bssn.py` - BSSN evolution equations and field definitions
- `derivatives.py` - Finite difference operators
- `init.py` - Initial data setup (gravitational waves, etc.)
- `kreiss_oliger.py` - Kreiss-Oliger dissipation
- `tensor_algebra.py` - Tensor operations (Christoffel symbols, etc.)
- `errors.py` - Constraint violation analysis

## Usage

The code simulates gravitational wave evolution using the BSSN formulation. Key parameters can be modified in `main.py`:

- Grid size and resolution
- Evolution time and timestep
- Initial data type (gravitational waves, black holes, etc.)
- Boundary conditions

## Performance

All computationally intensive operations are JIT-compiled with JAX for near-C++ performance while maintaining Python's readability and ease of use.
