def LévyCoefficientsFreeSimplySupported(beta, lambda_1, lambda_2, a, b, dddw_p_0, n):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from numpy import cos, cosh, isclose, sin, sinh
    from math  import pi

    gamma_1 = n * pi * lambda_1 / b
    gamma_2 = n * pi * lambda_2 / b

    if beta >= 0:
        if not isclose(lambda_1, lambda_2):
            A_n = dddw_p_0 * sinh(a * gamma_1) * sinh(a * gamma_2) / (gamma_1**2 * (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) ) )

            B_n = - dddw_p_0 * sinh(a * gamma_2) * cosh(a * gamma_1) / (gamma_1**2 * (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) ) )

            C_n = - gamma_1**2 / gamma_2**2 * A_n

            D_n = dddw_p_0 * sinh(a * gamma_1) * cosh(a * gamma_2) / (gamma_2**2 * (gamma_1 * sinh(a * gamma_2) * cosh(a * gamma_1) \
                - gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) ) )

        else:
            A_n = 2 * dddw_p_0 * (cosh(2 * a * gamma_1) - 1) / (gamma_1**3 * (2 * a * gamma_1 - sinh(2 * a * gamma_1) ) )

            B_n = dddw_p_0 * sinh(2 * a * gamma_1) / (gamma_1**2 * (2 * a * gamma_1 - sinh(2 * a * gamma_1) ) )

            C_n = - 2 * dddw_p_0 * (a * gamma_1 + sinh(2 * a * gamma_1) ) / (gamma_1**3 * (2 * a * gamma_1 - sinh(2 * a * gamma_1) ) )

            D_n = - gamma_1 * A_n / 2

    else:
        A_n = 2 * dddw_p_0 * gamma_1 * gamma_2 * (sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) + sin(a * gamma_2)**2 \
            + cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) - cos(a * gamma_2)**2) / ( (gamma_1**2 + gamma_2**2)**2 * (gamma_1 \
            * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1) ) )

        B_n = - dddw_p_0 * (- gamma_1**2 * sinh(2 * a * gamma_1) + 2 * gamma_1 * gamma_2 * sin(2 * a * gamma_2) \
            + gamma_2**2 * sinh(2 * a * gamma_1) ) / ( (gamma_1**2 + gamma_2**2)**2 * (gamma_1 * sin(2 * a * gamma_2) \
            - gamma_2 * sinh(2 * a * gamma_1) ) )

        C_n = - dddw_p_0 * (gamma_1**2 * sin(2 * a * gamma_2) + 2 * gamma_1 * gamma_2 * sinh(2 * a * gamma_1) \
            - gamma_2**2 * sin(2 * a * gamma_2) ) / ( (gamma_1**2 + gamma_2**2)**2 * (gamma_1 * sin(2 * a * gamma_2) \
            - gamma_2 * sinh(2 * a * gamma_1) ) )

        D_n = - dddw_p_0 * (gamma_1**2 * sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) + gamma_1**2 * sin(a * gamma_2)**2 \
            + gamma_1**2 * cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) - gamma_1**2 * cos(a * gamma_2)**2 \
            - gamma_2**2 * sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) - gamma_2**2 * sin(a * gamma_2)**2 \
            - gamma_2**2 * cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) + gamma_2**2 * cos(a * gamma_2)**2) \
            / ( (gamma_1**2 + gamma_2**2)**2 * (gamma_1 * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1) ) )


    return(A_n, B_n, C_n, D_n)