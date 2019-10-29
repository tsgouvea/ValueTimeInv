import sympy as sy

p, t, b, r = sy.symbols('P(B), t, beta, r')

PR = r * (p * sy.exp(-t/b)/b)