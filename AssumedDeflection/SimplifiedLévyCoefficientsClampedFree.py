def SimplifiedLévyCoefficientsClampedFree(beta, lambda_1, lambda_2, a, b, w_p, n):
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
            A_n = - gamma_2**2 * w_p * (-gamma_1**2 * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_1 * gamma_2 * sinh(a * gamma_1) \
                * sinh(a * gamma_2) + gamma_2**2) / (gamma_1**4 + gamma_1**3 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) \
                - 2 * gamma_1**2 * gamma_2**2 * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_1 * gamma_2**3 * sinh(a * gamma_1) \
                * sinh(a * gamma_2) + gamma_2**4)

            B_n = - gamma_1 * gamma_2**2 * w_p * (gamma_1 * sinh(a * gamma_1) * cosh(a * gamma_2) - gamma_2 * sinh(a * gamma_2) \
                * cosh(a * gamma_1) ) / (gamma_1**4 + gamma_1**3 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) - 2 * gamma_1**2 \
                * gamma_2**2 * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_1 * gamma_2**3 * sinh(a * gamma_1) * sinh(a * gamma_2) \
                + gamma_2**4)

            C_n = - gamma_1**2 * w_p * (gamma_1**2 + gamma_1 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) - gamma_2**2 * cosh(a * gamma_1) \
                * cosh(a * gamma_2) ) / (gamma_1**4 + gamma_1**3 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) - 2 * gamma_1**2 \
                * gamma_2**2 * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_1 * gamma_2**3 * sinh(a * gamma_1) * sinh(a * gamma_2) \
                + gamma_2**4)

            D_n = - gamma_1 / gamma_2 * B_n

        else:
            A_n = - w_p

            B_n = - gamma_1 * w_p * (2 * a * gamma_1 + sinh(2 * a * gamma_1) ) / (- 2 * a**2 * gamma_1**2 + cosh(2 * a * gamma_1) + 7)

            C_n = w_p * (2 * a * gamma_1 + sinh(2 * a * gamma_1) ) / (- 2 * a**2 * gamma_1**2 + cosh(2 * a * gamma_1) + 7)

            D_n = gamma_1 * w_p * (sinh(a * gamma_1)**2 + 2) / (- a**2 * gamma_1**2 + sinh(a * gamma_1)**2 + 4)

    else:
        A_n = - w_p

        B_n = gamma_1 * w_p * (gamma_1**3 * sin(2 * a * gamma_2) + gamma_1**2 * gamma_2 * sinh(2 * a * gamma_1) + gamma_1 * gamma_2**2 \
            * sin(2 * a * gamma_2) + gamma_2**3 * sinh(2 * a * gamma_1) ) / (2 * gamma_1**4 * sin(a * gamma_2)**2 + 2 * gamma_1**2 \
            * gamma_2**2 * sin(a * gamma_2)**2 - gamma_1**2 * gamma_2**2 * cosh(2 * a * gamma_1) - 7 * gamma_1**2 * gamma_2**2 \
            - gamma_2**4 * cosh(2 * a * gamma_1) + gamma_2**4)

        C_n = - gamma_2 / gamma_1 * B_n

        D_n = gamma_1 * gamma_2 * w_p * (gamma_1**2 * sin(a * gamma_2)**2 + gamma_1**2 * sinh(a * gamma_1)**2 + 2 * gamma_1**2 + gamma_2**2 \
            * sin(a * gamma_2)**2 + gamma_2**2 * sinh(a * gamma_1)**2 - 2 * gamma_2**2) / (-gamma_1**4 * sin(a * gamma_2)**2 - gamma_1**2 \
            * gamma_2**2 * sin(a * gamma_2)**2 + gamma_1**2 * gamma_2**2 * sinh(a * gamma_1)**2 + 4 * gamma_1**2 * gamma_2**2 + gamma_2**4 \
            * sinh(a * gamma_1)**2)


    return(A_n, B_n, C_n, D_n)