def SimplifiedLévyCoefficientsSimplySupportedSimplySupported(beta, lambda_1, lambda_2, a, b, w_p, n):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from numpy import cos, cosh, isclose, sin, sinh, tanh
    from math  import pi

    gamma_1 = n * pi * lambda_1 / b
    gamma_2 = n * pi * lambda_2 / b

    if beta >= 0:
        if not isclose(lambda_1, lambda_2):
            A_n = gamma_2**2 * w_p / (gamma_1**2 - gamma_2**2)

            B_n = - gamma_2**2 * w_p * (cosh(a * gamma_1) - 1) / ( (gamma_1**2 - gamma_2**2) * sinh(a * gamma_1) )

            C_n = - gamma_1**2 / gamma_2**2 * A_n

            D_n = gamma_1**2 * w_p * (cosh(a * gamma_2) - 1) / ( (gamma_1**2 - gamma_2**2) * sinh(a * gamma_2) )

        else:
            A_n = - w_p

            B_n = - gamma_1 * w_p * (cosh(a * gamma_1) - 1) / (2 * sinh(a * gamma_1) )

            C_n = - w_p * (a * gamma_1 * cosh(a * gamma_1) / sinh(a * gamma_1)**2 - a * gamma_1 / sinh(a * gamma_1)**2 - 2 / tanh(a * gamma_1) \
                + 2 / sinh(a * gamma_1) ) / 2

            D_n = gamma_1 * w_p / 2

    else:
        A_n = - w_p

        B_n = w_p * (2 * gamma_1**2 * cos(a * gamma_2) * sinh(a * gamma_1) - gamma_1**2 * sinh(2 * a * gamma_1) - 4 * gamma_1 * gamma_2 \
            * sin(a * gamma_2) * cosh(a * gamma_1) + 2 * gamma_1 * gamma_2 * sin(2 * a * gamma_2) - 2 * gamma_2**2 * cos(a * gamma_2) \
            * sinh(a * gamma_1) + gamma_2**2 * sinh(2 * a * gamma_1) ) / (2 * gamma_1 * gamma_2 * (sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) \
            + sin(a * gamma_2)**2 + cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) - cos(a * gamma_2)**2) )

        C_n = - w_p * (2 * gamma_1**2 * sin(a * gamma_2) * cosh(a * gamma_1) - gamma_1**2 * sin(2 * a * gamma_2) + 4 * gamma_1 * gamma_2 \
            * cos(a * gamma_2) * sinh(a * gamma_1) - 2 * gamma_1 * gamma_2 * sinh(2 * a * gamma_1) - 2 * gamma_2**2 * sin(a * gamma_2) \
            * cosh(a * gamma_1) + gamma_2**2 * sin(2 * a * gamma_2) ) / (2 * gamma_1 * gamma_2 * (sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) \
            + sin(a * gamma_2)**2 + cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) - cos(a * gamma_2)**2) )

        D_n = w_p * (gamma_1**2 - gamma_2**2) / (2 * gamma_1 * gamma_2)

    return(A_n, B_n, C_n, D_n)