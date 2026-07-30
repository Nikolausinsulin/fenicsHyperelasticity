import ufl
import dolfinx


def neoHookDokken(domain, u, youngsModulus, poissonRatio):
    # Define kinematic quantities used in the problem
    # Spatial dimension
    d = len(u)

    # Identity tensor
    I = ufl.variable(ufl.Identity(d))

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    # Define the elasticity model via a stored strain energy density function W, and create the expression for the first Piola-Kirchhoff stress:
    # Elasticity parameters

    E = dolfinx.default_scalar_type(youngsModulus)
    nu = dolfinx.default_scalar_type(poissonRatio)
    mu = dolfinx.fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda = dolfinx.fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

    # Stored strain energy density (compressible neo-Hookean model)
    W = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2

    # Hyper-elasticity
    P = ufl.diff(W, F)

    return P


def neoHookBleiler(domain, u, mu, Lambda):
    # same as dokken, but J instead of ln(J) in last term.
    d = len(u)

    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)

    I_1 = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    W = (mu / 2) * (I_1 - 3) - mu * ufl.ln(J) + (Lambda / 2) * (J - 1) ** 2

    P = ufl.diff(W, F)

    return P


def simplifiedMooneyRivlinTP1(domain, u, c_10, c_01, c_02):

    C_10 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_10))
    C_01 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_01))
    C_02 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_02))

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)

    # see https://en.wikipedia.org/wiki/Invariants_of_tensors
    I_1 = ufl.tr(C)
    I_2 = 0.5 * (ufl.tr(C) ** 2 - ufl.tr(C * C))

    W = C_10 * (I_1 - 3) + C_01 * (I_2 - 3) + C_02 * (I_2 - 3) ** 2

    P = ufl.diff(W, F)
    return P


def generalizedMooneyRivlinDegree2(domain, u, c_10, c_01, c_11, c_20, c_02, d_1, d_2):

    C_10 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_10))
    C_01 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_01))
    C_11 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_11))
    C_20 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_20))
    C_02 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_02))

    D_1 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(d_1))
    D_2 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(d_2))

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)

    # see https://en.wikipedia.org/wiki/Invariants_of_tensors
    I_1 = ufl.variable(ufl.tr(C))
    I_2 = ufl.variable(0.5 * (ufl.tr(C) ** 2 - ufl.tr(C * C)))
    J = ufl.variable(ufl.det(C))

    W = (
        C_10 * (I_1 - 3)
        + C_01 * (I_2 - 3)
        + C_11 * (I_1 - 3)(I_2 - 3)
        + C_20 * (I_1 - 3) ** 2
        + C_02 * (I_2 - 3) ** 2
        # + 1 / D_1 * (J - 1) ** 2
    )
    # if d_1 != 0.0:
    # if d_2 != 0.0:
    #     W += 1 / D_2 * (J - 1) ** 4

    # W += might lead to self reference problems.

    P = ufl.diff(W, F)
    return P


def generalizedMooneyRivlinDegree1(domain, u, c_10, c_01, d_1):

    C_10 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_10))
    C_01 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_01))
    D_1 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(d_1))

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)

    # see https://en.wikipedia.org/wiki/Invariants_of_tensors
    I_1 = ufl.variable(ufl.tr(C))
    I_2 = ufl.variable(0.5 * (ufl.tr(C) ** 2 - ufl.tr(C * C)))
    J = ufl.variable(ufl.det(C))

    W = C_10 * (I_1 - 3) + C_01 * (I_2 - 3) + 1 / D_1 * (J - 1) ** 2

    P = ufl.diff(W, F)
    return P


def generalizedMooneyRivlinDegree1Wikipedia(domain, u, c_10, c_01, d_1):
    # https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid

    C_10 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_10))
    C_01 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(c_01))
    D_1 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(d_1))

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    B = ufl.variable(F * F.T)

    J_C = ufl.variable(ufl.det(C))
    J_F = ufl.variable(ufl.det(F))

    I_1bar = ufl.variable(J_F ** (-2 / 3) * ufl.tr(C))
    I_2bar = ufl.variable(J_F ** (-4 / 3) * (0.5 * (ufl.tr(C) ** 2 - ufl.tr(C * C))))

    W = C_10 * (I_1bar - 3) + C_01 * (I_2bar - 3) + 1 / D_1 * (J_F - 1) ** 2

    P = ufl.diff(W, F)
    return P
