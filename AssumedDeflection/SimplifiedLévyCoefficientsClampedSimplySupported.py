def SimplifiedLévyCoefficientsClampedSimplySupported(beta, lambda_1, lambda_2, a, b, w_p, n):
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
            H_n = gamma_1**3 * sinh(a * gamma_2) * cosh(a * gamma_1) - gamma_1**2 * gamma_2 * sinh(a * gamma_1) * cosh(a * gamma_2) - gamma_1 \
                * gamma_2**2 * sinh(a * gamma_2) * cosh(a * gamma_1) + gamma_2**3 * sinh(a * gamma_1) * cosh(a * gamma_2)

            A_n = gamma_2 * w_p * (gamma_1**2 * sinh(a * gamma_1) * cosh(a * gamma_2) - gamma_1**2 * sinh(a * gamma_1) + gamma_1 * gamma_2 \
                * sinh(a * gamma_2) - gamma_2**2 * sinh(a * gamma_1) * cosh(a * gamma_2) ) / H_n

            B_n = - gamma_2 * w_p * (gamma_1**2 * cosh(a * gamma_1) * cosh(a * gamma_2) - gamma_1**2 * cosh(a * gamma_1) - gamma_2**2 \
                * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_2**2 * cosh(a * gamma_2) ) / H_n

            C_n = - gamma_1 * w_p * (gamma_1**2 * sinh(a * gamma_2) * cosh(a * gamma_1) - gamma_1 * gamma_2 * sinh(a * gamma_1) - gamma_2**2 \
                * sinh(a * gamma_2) * cosh(a * gamma_1) + gamma_2**2 * sinh(a * gamma_2) ) / H_n

            D_n = - gamma_1 / gamma_2 * B_n

        else:
            H_n = 2 * a * gamma_1 - sinh(2 * a * gamma_1)

            A_n = - w_p

            B_n = - gamma_1 * w_p * (a * gamma_1 * sinh(a * gamma_1) + 2 * cosh(a * gamma_1) - cosh(2 * a * gamma_1) - 1) / H_n

            C_n = - B_n / gamma_1

            D_n = gamma_1 * w_p * (a * gamma_1 * cosh(a * gamma_1) + sinh(a * gamma_1) - sinh(2 * a * gamma_1) ) / H_n

    else:
        H_n = gamma_1 * sin(2 * a * gamma_2) - gamma_2 * sinh(2 * a * gamma_1)

        A_n = - w_p

        B_n = w_p * (- gamma_1**2 * sin(a * gamma_2) * sinh(a * gamma_1) + gamma_1 * gamma_2 * sin(a * gamma_2)**2 * cosh(2 * a * gamma_1) \
            - gamma_1 * gamma_2 * sin(a * gamma_2)**2 + gamma_1 * gamma_2 * cos(a * gamma_2)**2 * cosh(2 * a * gamma_1) + gamma_1 * gamma_2 \
            * cos(a * gamma_2)**2 - 2 * gamma_1 * gamma_2 * cos(a * gamma_2) * cosh(a * gamma_1) + gamma_2**2 * sin(a * gamma_2) \
            * sinh(a * gamma_1) ) / (gamma_2 * H_n)

        C_n = - gamma_2 / gamma_1 * B_n

        D_n = w_p * (gamma_1**3 * sin(a * gamma_2) * cosh(a * gamma_1) + gamma_1**2 * gamma_2 * cos(a * gamma_2) * sinh(a * gamma_1) \
            - gamma_1**2 * gamma_2 * sinh(2 * a * gamma_1) + gamma_1 * gamma_2**2 * sin(a * gamma_2) * cosh(a * gamma_1) - gamma_1 * gamma_2**2 \
            * sin(2 * a * gamma_2) + gamma_2**3 * cos(a * gamma_2) * sinh(a * gamma_1) ) / (gamma_1 * gamma_2 * H_n)


    return(A_n, B_n, C_n, D_n)