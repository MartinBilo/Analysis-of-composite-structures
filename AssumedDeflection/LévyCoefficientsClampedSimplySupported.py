def LévyCoefficientsClampedSimplySupported(beta, lambda_1, lambda_2, a, b, dw_p_0, n):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from numpy import cos, cosh, isclose, sin, sinh, zeros
    from math  import pi

    gamma_1 = n * pi * lambda_1 / b
    gamma_2 = n * pi * lambda_2 / b

    if beta >= 0:
        if not isclose(lambda_1, lambda_2):
            A_n = dw_p_0 * sinh(a * gamma_1) * sinh(a * gamma_2) / (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) )

            B_n = - dw_p_0 * sinh(a * gamma_2) * cosh(a * gamma_1) / (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) )

            C_n = - A_n

            D_n = dw_p_0 * sinh(a * gamma_1) * cosh(a * gamma_2) / (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) )

        else:
            A_n = zeros(len(n) )

            B_n = dw_p_0 * sinh(2 * a * gamma_1) / (2 * a * gamma_1 - sinh(2 * a * gamma_1) )

            C_n = - a * dw_p_0 / (a * gamma_1 - sinh(2 * a * gamma_1) / 2)

            D_n = - dw_p_0 * (cosh(2 * a * gamma_1) - 1) / (2 * a * gamma_1 - sinh(2 * a * gamma_1) )

    else:
        A_n = zeros(len(n) )

        B_n = dw_p_0 * sinh(2 * a * gamma_1) / (gamma_1 * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1) )

        C_n = - dw_p_0 * sin(2 * a * gamma_2) / (gamma_1 * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1) )

        D_n = - dw_p_0 * (sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) + sin(a * gamma_2)**2 + cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) \
            - cos(a * gamma_2)**2) / (gamma_1 * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1) )


    return(A_n, B_n, C_n, D_n)