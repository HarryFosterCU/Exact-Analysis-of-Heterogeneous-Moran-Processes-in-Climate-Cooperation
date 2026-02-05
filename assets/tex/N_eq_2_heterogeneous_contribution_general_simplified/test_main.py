from sympy.parsing.latex import parse_latex
import sympy as sym

with open("./main.tex", "r") as f:
    text = f.read()


def clean_latex_for_sympy(s):
    # Remove \left and \right
    s = s.replace(r"\left", "").replace(r"\right", "")
    # Remove any surrounding brackets if needed (optional)
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return s


cleaned = clean_latex_for_sympy(text)

sym_matrix = sym.Matrix(parse_latex(cleaned))

a1 = sym.Symbol("\alpha_1")
a2 = sym.Symbol("\alpha_2")
r = sym.Symbol("r")
omega = sym.Symbol("w")

expected_matrix = sym.Matrix(
    [
        [
            (a2 * r * omega + 2) / (a2 * r * omega + a2 * omega * (r - 2) + 4),
            (a2 * omega * (r - 2) + 2) / (a2 * r * omega + a2 * omega * (r - 2) + 4),
        ],
        [
            (a1 * r * omega + 2) / (a1 * r * omega + a1 * omega * (r - 2) + 4),
            (a1 * omega * (r - 2) + 2) / (a1 * r * omega + a1 * omega * (r - 2) + 4),
        ],
    ]
)


assert expected_matrix == sym_matrix
