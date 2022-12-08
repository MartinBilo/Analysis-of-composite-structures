def BeamApproximationClampedClamped(a, m, nodes_x):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from math  import pi
    from numpy import cos, cosh, einsum, sin, sinh
    from sys   import float_info

    # Table 5.3 and equation 5.54
    lambda_m = (m == 1) * 4.712 \
             + (m == 2) * 7.854 \
             + (m > 2)  * (2 * m + 1) * pi / 2

    # Equation 5.51
    gamma_m = (cos(lambda_m) - cosh(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )

    # Equation 5.49
    xm       = einsum('k,ijk->ijk', gamma_m[0, :],  cos(einsum('k,ij->ijk', lambda_m[0, :] / a, nodes_x) ) ) \
             - einsum('k,ijk->ijk', gamma_m[0, :], cosh(einsum('k,ij->ijk', lambda_m[0, :] / a, nodes_x) ) ) \
             +                                      sin(einsum('k,ij->ijk', lambda_m[0, :] / a, nodes_x) ) \
             -                                     sinh(einsum('k,ij->ijk', lambda_m[0, :] / a, nodes_x) )

    Xm       = - a * (- gamma_m[0, :] *  sin(lambda_m[0, :] ) \
                      + gamma_m[0, :] * sinh(lambda_m[0, :] ) \
                      +                  cos(lambda_m[0, :] ) \
                      +                 cosh(lambda_m[0, :] ) \
                      - 2) / lambda_m[0, :]

    XkXm     = (m == m.T) * (a * (2 * gamma_m**2 * lambda_m - 2 * gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                          + gamma_m**2 * sin(2 * lambda_m) / 2 - 2 * gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                          + gamma_m**2 * sinh(2 * lambda_m) / 2 - 4 * gamma_m * sin(lambda_m) * sinh(lambda_m) \
                          - 2 * gamma_m * cos(lambda_m)**2 + gamma_m * cosh(2 * lambda_m) + gamma_m \
                          - 2 * sin(lambda_m) * cosh(lambda_m) - sin(2 * lambda_m) / 2 \
                          + 2 * cos(lambda_m) * sinh(lambda_m) + sinh(2 * lambda_m) / 2) / (2 * lambda_m) ) \
             + (m != m.T) * (a * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                + gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                + gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                - gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                - gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                - gamma_m.T * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                + gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                - gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                - gamma_m.T * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                + gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m.T * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                - gamma_m.T * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                + gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                + gamma_m * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                - gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                + gamma_m * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                - gamma_m * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                + gamma_m * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                + gamma_m * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                - gamma_m * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                - lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                - lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                + lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                + lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                - lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                + lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                - lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                - lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                + lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                - lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                + lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                + lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                + lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                - lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                - lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) ) \
                                / (lambda_m.T**4 - lambda_m**4 + (m == m.T) * float_info.max) )

    XkXm_m   = (m == m.T) * (6 * (2 * gamma_m**2 * lambda_m * sin(lambda_m) * cosh(lambda_m) \
                                    - gamma_m**2 * lambda_m * sin(2 * lambda_m) / 2 \
                                + 2 * gamma_m**2 * lambda_m * cos(lambda_m) * sinh(lambda_m) \
                                    - gamma_m**2 * lambda_m * sinh(2 * lambda_m) / 2 \
                                - 4 * gamma_m**2 * sin(lambda_m) * sinh(lambda_m) \
                                    - gamma_m**2 * cos(lambda_m)**2 \
                                    + gamma_m**2 * cosh(2 * lambda_m) / 2 \
                                    + gamma_m**2 / 2 \
                                - 2 * gamma_m * lambda_m * sin(lambda_m)**2 \
                                + 4 * gamma_m * lambda_m * sin(lambda_m) * sinh(lambda_m) \
                                    - gamma_m * lambda_m * cosh(2 * lambda_m) \
                                    + gamma_m * lambda_m \
                                - 4 * gamma_m * sin(lambda_m) * cosh(lambda_m) \
                                    - gamma_m * sin(2 * lambda_m) \
                                + 4 * gamma_m * cos(lambda_m) * sinh(lambda_m) \
                                    + gamma_m * sinh(2 * lambda_m) \
                                + 2 * lambda_m * sin(lambda_m) * cosh(lambda_m) \
                                    + lambda_m * sin(2 * lambda_m) / 2 \
                                - 2 * lambda_m * cos(lambda_m) * sinh(lambda_m) \
                                    - lambda_m * sinh(2 * lambda_m) / 2 \
                                               + cos(lambda_m)**2 \
                                           + 4 * cos(lambda_m) * cosh(lambda_m) \
                                               + cosh(2 * lambda_m) / 2 - 11 / 2) / (a * lambda_m**2) ) \
             + (m != m.T) * (- (12 * gamma_m.T * gamma_m * lambda_m.T**7 * sin(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**7 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**7 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**7 * sinh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 24 * gamma_m.T * gamma_m * lambda_m.T**6 * cos(lambda_m.T) * cos(lambda_m) \
                              - 24 * gamma_m.T * gamma_m * lambda_m.T**6 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 24 * gamma_m.T * gamma_m * lambda_m.T**6 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 24 * gamma_m.T * gamma_m * lambda_m.T**6 * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 72 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                              + 72 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 72 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 72 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 96 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                              - 96 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 96 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 96 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * cos(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 72 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * cos(lambda_m) \
                              - 72 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 72 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 72 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * sin(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 48 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m**7 * sin(lambda_m) * cos(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m**7 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * gamma_m * lambda_m**7 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * gamma_m * lambda_m**7 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 24 * gamma_m.T * gamma_m * lambda_m**6 * cos(lambda_m.T) * cos(lambda_m) \
                              + 24 * gamma_m.T * gamma_m * lambda_m**6 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 24 * gamma_m.T * gamma_m * lambda_m**6 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 24 * gamma_m.T * gamma_m * lambda_m**6 * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**7 * sin(lambda_m.T) * sin(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**7 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**7 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m.T * lambda_m.T**7 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**6 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**6 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**6 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T**6 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 24 * gamma_m.T * lambda_m.T**6 * sin(lambda_m) * cos(lambda_m.T) \
                              + 24 * gamma_m.T * lambda_m.T**6 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 24 * gamma_m.T * lambda_m.T**6 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 24 * gamma_m.T * lambda_m.T**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m.T * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m.T * lambda_m.T**5 * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                              - 48 * gamma_m.T * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                              + 48 * gamma_m.T * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                              - 48 * gamma_m.T * lambda_m.T**5 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                              + 48 * gamma_m.T * lambda_m.T**5 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**4 * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T**4 * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 72 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                              - 72 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 72 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 72 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * sin(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T**3 * lambda_m**4 * sinh(lambda_m.T) * sinh(lambda_m) \
                              - 96 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                              - 96 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 96 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 96 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m.T * lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T**2 * lambda_m**5 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m.T * lambda_m.T**2 * lambda_m**5 * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 72 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * cos(lambda_m.T) \
                              + 72 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 72 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 72 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * sin(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m.T * lambda_m**6 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m.T * lambda_m.T * lambda_m**6 * sinh(lambda_m.T) * sinh(lambda_m) \
                              - 48 * gamma_m.T * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * cos(lambda_m) \
                              + 48 * gamma_m.T * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 48 * gamma_m.T * lambda_m.T * lambda_m**5 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 48 * gamma_m.T * lambda_m.T * lambda_m**5 * sinh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m.T * lambda_m**7 * cos(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m.T * lambda_m**7 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m.T * lambda_m**7 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m.T * lambda_m**7 * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 24 * gamma_m.T * lambda_m**6 * sin(lambda_m) * cos(lambda_m.T) \
                              - 24 * gamma_m.T * lambda_m**6 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 24 * gamma_m.T * lambda_m**6 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 24 * gamma_m.T * lambda_m**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T**7 * cos(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**7 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**7 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m.T**7 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T**6 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 24 * gamma_m * lambda_m.T**6 * sin(lambda_m.T) * cos(lambda_m) \
                              - 24 * gamma_m * lambda_m.T**6 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 24 * gamma_m * lambda_m.T**6 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 24 * gamma_m * lambda_m.T**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**5 * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m.T**5 * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 48 * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                              + 48 * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                              - 48 * gamma_m * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                              + 48 * gamma_m * lambda_m.T**5 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T**4 * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 72 * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                              + 72 * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 72 * gamma_m * lambda_m.T**4 * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 72 * gamma_m * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * cos(lambda_m) \
                              - 12 * gamma_m * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**3 * lambda_m**4 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T**3 * lambda_m**4 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 96 * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                              - 96 * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 96 * gamma_m * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 96 * gamma_m * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * sin(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m.T**2 * lambda_m**5 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 72 * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * cos(lambda_m) \
                              - 72 * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 72 * gamma_m * lambda_m.T**2 * lambda_m**4 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 72 * gamma_m * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * gamma_m * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * cos(lambda_m) \
                              + 12 * gamma_m * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 12 * gamma_m * lambda_m.T * lambda_m**6 * cos(lambda_m) * cosh(lambda_m.T) \
                              - 12 * gamma_m * lambda_m.T * lambda_m**6 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 48 * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m) * cos(lambda_m.T) \
                              + 48 * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 48 * gamma_m * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 48 * gamma_m * lambda_m.T * lambda_m**5 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m**7 * sin(lambda_m.T) * sin(lambda_m) \
                              - 12 * gamma_m * lambda_m**7 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 12 * gamma_m * lambda_m**7 * sin(lambda_m) * sinh(lambda_m.T) \
                              + 12 * gamma_m * lambda_m**7 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 24 * gamma_m * lambda_m**6 * sin(lambda_m.T) * cos(lambda_m) \
                              + 24 * gamma_m * lambda_m**6 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 24 * gamma_m * lambda_m**6 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 24 * gamma_m * lambda_m**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * lambda_m.T**7 * sin(lambda_m) * cos(lambda_m.T) \
                              - 12 * lambda_m.T**7 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 12 * lambda_m.T**7 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 12 * lambda_m.T**7 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 12 * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                              - 12 * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                              + 12 * lambda_m.T**6 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                              - 12 * lambda_m.T**6 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 24 * lambda_m.T**6 * sin(lambda_m.T) * sin(lambda_m) \
                              - 24 * lambda_m.T**6 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 24 * lambda_m.T**6 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 24 * lambda_m.T**6 * sinh(lambda_m.T) * sinh(lambda_m) \
                              - 12 * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                              + 12 * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 12 * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                              + 12 * lambda_m.T**5 * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 48 * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                              - 48 * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                              - 48 * lambda_m.T**5 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                              + 48 * lambda_m.T**5 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                              + 12 * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                              + 12 * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 12 * lambda_m.T**4 * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                              - 12 * lambda_m.T**4 * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 72 * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                              + 72 * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 72 * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 72 * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 12 * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * cos(lambda_m.T) \
                              + 12 * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * cosh(lambda_m.T) \
                              - 12 * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 12 * lambda_m.T**3 * lambda_m**4 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 96 * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                              + 96 * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                              + 96 * lambda_m.T**3 * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 96 * lambda_m.T**3 * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 384 * lambda_m.T**3 * lambda_m**3 \
                              - 12 * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * cos(lambda_m) \
                              + 12 * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * cosh(lambda_m) \
                              - 12 * lambda_m.T**2 * lambda_m**5 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 12 * lambda_m.T**2 * lambda_m**5 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 72 * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * sin(lambda_m) \
                              - 72 * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * sinh(lambda_m) \
                              + 72 * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 72 * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m.T) * sinh(lambda_m) \
                              + 12 * lambda_m.T * lambda_m**6 * sin(lambda_m) * cos(lambda_m.T) \
                              - 12 * lambda_m.T * lambda_m**6 * sin(lambda_m) * cosh(lambda_m.T) \
                              + 12 * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * sinh(lambda_m) \
                              - 12 * lambda_m.T * lambda_m**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                              + 48 * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * cos(lambda_m) \
                              - 48 * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * cosh(lambda_m) \
                              - 48 * lambda_m.T * lambda_m**5 * cos(lambda_m) * cosh(lambda_m.T) \
                              + 48 * lambda_m.T * lambda_m**5 * cosh(lambda_m.T) * cosh(lambda_m) \
                              - 12 * lambda_m**7 * sin(lambda_m.T) * cos(lambda_m) \
                              - 12 * lambda_m**7 * sin(lambda_m.T) * cosh(lambda_m) \
                              + 12 * lambda_m**7 * cos(lambda_m) * sinh(lambda_m.T) \
                              + 12 * lambda_m**7 * sinh(lambda_m.T) * cosh(lambda_m) \
                              + 24 * lambda_m**6 * sin(lambda_m.T) * sin(lambda_m) \
                              + 24 * lambda_m**6 * sin(lambda_m.T) * sinh(lambda_m) \
                              - 24 * lambda_m**6 * sin(lambda_m) * sinh(lambda_m.T) \
                              - 24 * lambda_m**6 * sinh(lambda_m.T) * sinh(lambda_m) ) \
                              / (a * (lambda_m.T**8 - 2 * lambda_m.T**4 * lambda_m**4 + lambda_m**8) + (m == m.T) * float_info.max) )

    XkddXm   = (m == m.T) * (lambda_m * (- gamma_m**2 * sin(2 * lambda_m) / 2 \
                                         + gamma_m**2 * sinh(2 * lambda_m) / 2 \
                                         + gamma_m * cos(2 * lambda_m) \
                                         + gamma_m * cosh(2 * lambda_m) \
                                         - 2 * gamma_m - 2 * lambda_m + sin(2 * lambda_m) / 2 + sinh(2 * lambda_m) / 2) / (2 * a) ) \
             + (m != m.T) * (- lambda_m**2 * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m)
                                            + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                            - gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                            - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                            + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                            + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                            - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                            + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                            - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                            + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                            - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                            + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                            + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                            - gamma_m.T * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                            - gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                            + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                            + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                            + gamma_m.T * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                            - 4 * gamma_m.T * lambda_m.T**2 * lambda_m \
                                            + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                            - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                            - gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                            - gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m.T * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                            + gamma_m.T * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                            - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                            - gamma_m * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                            + 4 * gamma_m * lambda_m.T**3 \
                                            - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                            + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                            - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                            + gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                            - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                            + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                            + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                            - gamma_m * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                            - gamma_m * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                            - gamma_m * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                            + gamma_m * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                            + gamma_m * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                            - lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                            - lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                            - lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                            - lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                            + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                            + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                            + lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                            + lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                            - lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                            + lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                            + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                            - lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                            + lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                            - lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                            - lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                            + lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) ) \
                                            / (a * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) )

    dXkdXm   = (m == m.T) * (lambda_m * (2 * gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                                           - gamma_m**2 * sin(2 * lambda_m) / 2 \
                                           - 2 * gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                                           + gamma_m**2 * sinh(2 * lambda_m) / 2 \
                                           - 4 * gamma_m * cos(lambda_m) * cosh(lambda_m) \
                                           + gamma_m * cos(2 * lambda_m) \
                                           + gamma_m * cosh(2 * lambda_m) \
                                           + 2 * gamma_m + 2 * lambda_m \
                                           - 2 * sin(lambda_m) * cosh(lambda_m) \
                                           + sin(2 * lambda_m) / 2 \
                                           - 2 * cos(lambda_m) * sinh(lambda_m) \
                                           + sinh(2 * lambda_m) / 2) / (2 * a) ) \
             + (m != m.T) * (- lambda_m.T * lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                      - gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                      + gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                      - gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                      - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                      - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                      + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                      + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                      - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                      - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                      - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                      + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                      - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                      + gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                      - gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                      + gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                      + gamma_m.T * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                      - gamma_m.T * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                      - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                      - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                      + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                      + gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                      - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                      - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                      - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                      - gamma_m.T * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                      + 4 * gamma_m.T * lambda_m.T * lambda_m**2 \
                                                      - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                      + gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                      - gamma_m.T * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                      + gamma_m.T * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                      + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                      + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                      - gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                      - gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                      + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                      + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                      + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                      + gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                      - 4 * gamma_m * lambda_m.T**2 * lambda_m \
                                                      + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                      - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                      + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                      - gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                      + gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                      - gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                      - gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                      + gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                      - lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                      + lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                      + lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                      - lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                      + lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                      + lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                      + lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                      + lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                      - lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                      - lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                      - lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                      - lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                      + lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                      - lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                      - lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                      + lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) ) \
                                                      / (a * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) )

    ddXkddXm = (m == m.T) * (lambda_m**3 * (gamma_m**2 * lambda_m \
                                          + gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                                          + gamma_m**2 * sin(2 * lambda_m) / 4 \
                                          + gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                                          + gamma_m**2 * sinh(2 * lambda_m) / 4 \
                                          + 2 * gamma_m * sin(lambda_m) * sinh(lambda_m) \
                                          - gamma_m * cos(2 * lambda_m) / 2 \
                                          + gamma_m * cosh(2 * lambda_m) / 2 \
                                          + sin(lambda_m) * cosh(lambda_m) \
                                          - sin(2 * lambda_m) / 4 \
                                          - cos(lambda_m) * sinh(lambda_m) \
                                          + sinh(2 * lambda_m) / 4) / a**3) \
             + (m != m.T) * (lambda_m.T**2 * lambda_m**2 * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                          + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          + gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          + gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                          + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                          + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                          - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          - gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                          + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                          + gamma_m.T * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                          + gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                          + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                          - gamma_m.T * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                          - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                          - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                          + gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                          - gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                          + gamma_m.T * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                          - gamma_m.T * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                          - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                          + gamma_m * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                          + gamma_m * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                          + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                          + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                          - gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                          + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                          + gamma_m * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - gamma_m * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                          - gamma_m * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                          - gamma_m * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                          - gamma_m * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          - lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                          + lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          - lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          + lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                          + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                          - lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                          - lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                          - lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                          - lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          + lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                          - lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          + lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          - lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) ) \
                                                          / (a**3 * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) )

    XkdXm    = (m == m.T) * (gamma_m**2 * cos(lambda_m)**2 / 2 \
                           - gamma_m**2 * cos(lambda_m) * cosh(lambda_m) \
                           + gamma_m**2 * cosh(2 * lambda_m) / 4 \
                           + gamma_m**2 / 4 \
                           - gamma_m * sin(lambda_m) * cosh(lambda_m) \
                           + gamma_m * sin(2 * lambda_m) / 2 \
                           - gamma_m * cos(lambda_m) * sinh(lambda_m) \
                           + gamma_m * sinh(2 * lambda_m) / 2 \
                           - sin(lambda_m) * sinh(lambda_m) \
                           - cos(lambda_m)**2 / 2 \
                           + cosh(2 * lambda_m) / 4 \
                           + 1 / 4) \
             + (m != m.T) * (- lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                         - gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                         - gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                         - 4 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m \
                                         + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                         - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                         - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                         + gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                         - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                         - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m.T * gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                         - gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                         + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                         + gamma_m.T * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                         - gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                         + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                         + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                         + gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                         - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                         - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                         - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                         - gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                         + gamma_m.T * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                         - gamma_m.T * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                         - gamma_m.T * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                         + gamma_m.T * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                         - gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                         - gamma_m * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                         - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                         - gamma_m * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                         + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                         + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                         + gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                         - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                         + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                         - gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                         + gamma_m * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                         - gamma_m * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                         - gamma_m * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                         + gamma_m * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                         + lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                         - lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                         + lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                         - lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                         + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                         + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                         + lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                         + lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                         + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                         + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                         - lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                         - lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                         + lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                         - lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                         - lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                         + lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) ) \
                                         / (lambda_m.T**4 - lambda_m**4 + (m == m.T) * float_info.max) )

    ddXkdXm  = (m == m.T) * (lambda_m**2 * (gamma_m**2 * sin(lambda_m) * sinh(lambda_m) \
                                          - gamma_m**2 * cos(lambda_m)**2 / 2 \
                                          + gamma_m**2 * cosh(2 * lambda_m) / 4 \
                                          + gamma_m**2 / 4 \
                                          + gamma_m * sin(lambda_m) * cosh(lambda_m) \
                                          - gamma_m * sin(2 * lambda_m) / 2 \
                                          - gamma_m * cos(lambda_m) * sinh(lambda_m) \
                                          + gamma_m * sinh(2 * lambda_m) / 2 \
                                          + cos(lambda_m)**2 / 2 \
                                          - cos(lambda_m) * cosh(lambda_m) \
                                          + cosh(2 * lambda_m) / 4 \
                                          + 1 / 4) \
                                          / a**2) \
             + (m != m.T) * (lambda_m.T**2 * lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                       + gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                       - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                       - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                       - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                       + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                       - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m.T * gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                       - gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                       + gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                       - gamma_m.T * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                       + gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                       - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                       + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                       - gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                       - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                       + gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m.T * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                       + gamma_m.T * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m.T * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                       - gamma_m.T * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                       + gamma_m * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                       - gamma_m * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                       + gamma_m * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                       + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                       + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                       - gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                       - gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                       - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                       - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                       + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                       + gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                       + gamma_m * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                       - gamma_m * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                       + gamma_m * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                       - gamma_m * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                       + lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                       - lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                       - lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                       + lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                       + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                       + lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                       - lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                       - lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                       + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                       + lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                       + lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                       + lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                       - 4 * lambda_m.T * lambda_m**2 \
                                                       + lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                       - lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                       + lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                       - lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) ) \
                                                       / (a**2 * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) )

    return(xm, Xm, XkXm, XkXm_m, XkddXm, dXkdXm, ddXkddXm, XkdXm, ddXkdXm)