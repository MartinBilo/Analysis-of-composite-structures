def LévyCoefficientsClampedFree(beta, lambda_1, lambda_2, a, b, dw_p_0, dddw_p_a, n):
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
            H_n = gamma_1**4 + gamma_1**3 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) - 2 * gamma_1**2 * gamma_2**2 * cosh(a * gamma_1) \
                * cosh(a * gamma_2) + gamma_1 * gamma_2**3 * sinh(a * gamma_1) * sinh(a * gamma_2) + gamma_2**4

            A_n = (dddw_p_a * gamma_1 * sinh(a * gamma_1) - dddw_p_a * gamma_2 * sinh(a * gamma_2) + dw_p_0 * gamma_1**2 * gamma_2 \
                * sinh(a * gamma_2) * cosh(a * gamma_1) - dw_p_0 * gamma_1 * gamma_2**2 * sinh(a * gamma_1) * cosh(a * gamma_2) ) \
                / (gamma_1**4 + gamma_1**3 * gamma_2 * sinh(a * gamma_1) * sinh(a * gamma_2) - 2 * gamma_1**2 * gamma_2**2 \
                * cosh(a * gamma_1) * cosh(a * gamma_2) + gamma_1 * gamma_2**3 * sinh(a * gamma_1) * sinh(a * gamma_2) + gamma_2**4)

            B_n = - (dddw_p_a * gamma_1**2 * cosh(a * gamma_1) - dddw_p_a * gamma_2**2 * cosh(a * gamma_2) + dw_p_0 * gamma_1**3 * gamma_2 \
                * sinh(a * gamma_1) * sinh(a * gamma_2) - dw_p_0 * gamma_1**2 * gamma_2**2 * cosh(a * gamma_1) * cosh(a * gamma_2) \
                + dw_p_0 * gamma_2**4) / (gamma_1 * H_n)

            C_n = - A_n

            D_n = (dddw_p_a * gamma_1**2 * cosh(a * gamma_1) - dddw_p_a * gamma_2**2 * cosh(a * gamma_2) - dw_p_0 * gamma_1**4 \
                + dw_p_0 * gamma_1**2 * gamma_2**2 * cosh(a * gamma_1) * cosh(a * gamma_2) - dw_p_0 * gamma_1 * gamma_2**3 * sinh(a * gamma_1) \
                * sinh(a * gamma_2) ) / (gamma_2 * H_n)

        else:
            A_n = zeros(len(n) )

            B_n =  - (a * dddw_p_a * gamma_1 * sinh(a * gamma_1) + 2 * dddw_p_a * cosh(a * gamma_1) + dw_p_0 * gamma_1**2 \
                * sinh(a * gamma_1)**2 - 2 * dw_p_0 * gamma_1**2) / (gamma_1**2 * (- a**2 * gamma_1**2 + sinh(a * gamma_1)**2 + 4) )

            C_n = (a**2 * dw_p_0 * gamma_1**4 + a * dddw_p_a * gamma_1 * sinh(a * gamma_1) + 2 * dddw_p_a * cosh(a * gamma_1) \
                - 6 * dw_p_0 * gamma_1**2) / (gamma_1**3 * (- a**2 * gamma_1**2 + sinh(a * gamma_1)**2 + 4) )

            D_n = (2 * a * dddw_p_a * gamma_1 * cosh(a * gamma_1) - 2 * a * dw_p_0 * gamma_1**3 + 2 * dddw_p_a * sinh(a * gamma_1) \
                + dw_p_0 * gamma_1**2 * sinh(2 * a * gamma_1) ) / (gamma_1**2 * (- 2 * a**2 * gamma_1**2 + cosh(2 * a * gamma_1) + 7) )

    else:
        A_n = zeros(len(n) )

        B_n = - (dddw_p_a * gamma_1**3 * sin(a * gamma_2) * sinh(a * gamma_1) + 2 * dddw_p_a * gamma_1**2 * gamma_2 * cos(a * gamma_2) \
            * cosh(a * gamma_1) - dddw_p_a * gamma_1 * gamma_2**2 * sin(a * gamma_2) * sinh(a * gamma_1) + dw_p_0 * gamma_1**4 * gamma_2 \
            * sinh(a * gamma_1)**2 - 2 * dw_p_0 * gamma_1**4 * gamma_2 + 2 * dw_p_0 * gamma_1**2 * gamma_2**3 * sinh(a * gamma_1)**2 \
            + 6 * dw_p_0 * gamma_1**2 * gamma_2**3 + dw_p_0 * gamma_2**5 * sinh(a * gamma_1)**2) / (-gamma_1**6 * sin(a * gamma_2)**2 \
            - 2 * gamma_1**4 * gamma_2**2 * sin(a * gamma_2)**2 + gamma_1**4 * gamma_2**2 * sinh(a * gamma_1)**2 + 4 * gamma_1**4 \
            * gamma_2**2 - gamma_1**2 * gamma_2**4 * sin(a * gamma_2)**2 + 2 * gamma_1**2 * gamma_2**4 * sinh(a * gamma_1)**2 + 4 * gamma_1**2 \
            * gamma_2**4 + gamma_2**6 * sinh(a * gamma_1)**2)

        C_n = (dddw_p_a * gamma_1**2 * gamma_2 * sin(a * gamma_2) * sinh(a * gamma_1) + 2 * dddw_p_a * gamma_1 * gamma_2**2 * cos(a * gamma_2) \
            * cosh(a * gamma_1) - dddw_p_a * gamma_2**3 * sin(a * gamma_2) * sinh(a * gamma_1) + dw_p_0 * gamma_1**5 * sin(a * gamma_2)**2 \
            + 2 * dw_p_0 * gamma_1**3 * gamma_2**2 * sin(a * gamma_2)**2 - 6 * dw_p_0 * gamma_1**3 * gamma_2**2 + dw_p_0 * gamma_1 * gamma_2**4 \
            * sin(a * gamma_2)**2 + 2 * dw_p_0 * gamma_1 * gamma_2**4) / (- gamma_1**6 * sin(a * gamma_2)**2 - 2 * gamma_1**4 * gamma_2**2 \
            * sin(a * gamma_2)**2 + gamma_1**4 * gamma_2**2 * sinh(a * gamma_1)**2 + 4 * gamma_1**4 * gamma_2**2 - gamma_1**2 * gamma_2**4 \
            * sin(a * gamma_2)**2 + 2 * gamma_1**2 * gamma_2**4 * sinh(a * gamma_1)**2 + 4 * gamma_1**2 * gamma_2**4 + gamma_2**6 \
            * sinh(a * gamma_1)**2)

        D_n = - (2 * dddw_p_a * gamma_1 * sin(a * gamma_2) * cosh(a * gamma_1) + 2 * dddw_p_a * gamma_2 * cos(a * gamma_2) * sinh(a * gamma_1) \
            - dw_p_0 * gamma_1**3 * sin(2 * a * gamma_2) + dw_p_0 * gamma_1**2 * gamma_2 * sinh(2 * a * gamma_1) - dw_p_0 * gamma_1 \
            * gamma_2**2 * sin(2 * a * gamma_2) + dw_p_0 * gamma_2**3 * sinh(2 * a * gamma_1) ) / (2 * gamma_1**4 * sin(a * gamma_2)**2 \
            + 2 * gamma_1**2 * gamma_2**2 * sin(a * gamma_2)**2 - gamma_1**2 * gamma_2**2 * cosh(2 * a * gamma_1) - 7 * gamma_1**2 * gamma_2**2 \
            - gamma_2**4 * cosh(2 * a * gamma_1) + gamma_2**4)


    return(A_n, B_n, C_n, D_n)