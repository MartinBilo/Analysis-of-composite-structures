def BeamApproximationFreeFree(a, m, nodes_x):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]

    Sources:
        [1]: Whitney, J.M., Structural Analysis of Laminated Anisotropic Plates, Technomic Publishing Co., Lancaster PA, 1987, Sections 5.4 and
             6.8
    """

    # Import packages into library
    from math  import pi
    from numpy import cos, cosh, einsum, ones, sin, sinh, sqrt, zeros
    from sys   import float_info

    # Table 5.2/5.3 and equation 6.56
    lambda_m = (m == 1) * 4.712 \
             + (m == 2) * 7.854 \
             + (m > 2)  * (2 * m + 1) * pi / 2

    # Equation 6.55
    gamma_m = (cosh(lambda_m) - cos(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )

    # Equation 6.52 and 6.53
    xm       = zeros( [nodes_x.shape[0], nodes_x.shape[1], len(m)] )
    xm[:,:, m[0, :] == 1] = 1
    xm[:,:, m[0, :] == 2] = einsum('k,ij->ijk', ones( int(sqrt(len(m[0, :] ) ) ) ), sqrt(3) * (1 - 2 * nodes_x / a) )
    xm[:,:, m[0, :] > 2]  =                                               cosh(einsum('k,ij->ijk', lambda_m[0, m[0, :] > 2] / a, nodes_x) ) \
                          +                                                cos(einsum('k,ij->ijk', lambda_m[0, m[0, :] > 2] / a, nodes_x) ) \
                          - einsum('k,ijk->ijk', gamma_m[0, m[0, :] > 2], sinh(einsum('k,ij->ijk', lambda_m[0, m[0, :] > 2] / a, nodes_x) ) ) \
                          - einsum('k,ijk->ijk', gamma_m[0, m[0, :] > 2],  sin(einsum('k,ij->ijk', lambda_m[0, m[0, :] > 2] / a, nodes_x) ) )

    Xm       = (m[0, :] == 1) * (a) \
             + (m[0, :] == 2) * (0) \
             + (m[0, :] > 2)  * (a * (gamma_m[0, :] *  cos(lambda_m[0, :] ) \
                                    - gamma_m[0, :] * cosh(lambda_m[0, :] ) \
                                    +                  sin(lambda_m[0, :] ) \
                                    +                 sinh(lambda_m[0, :] ) ) \
                                    / lambda_m[0, :] )

    XkXm     = (m == 1) * ( (m.T == 1) * (a) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (a * (gamma_m.T * cos(lambda_m.T) - gamma_m.T * cosh(lambda_m.T) \
                                              +            sin(lambda_m.T) +             sinh(lambda_m.T) ) \
                                              / lambda_m.T) ) \
             + (m == 2) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (a) \
                          + (m.T > 2)  * (- sqrt(3) * a * (- 2 * gamma_m.T * sin(lambda_m.T) \
                                                           + 2 * gamma_m.T * sinh(lambda_m.T) \
                                                           + lambda_m.T * (gamma_m.T * cos(lambda_m.T) \
                                                           - gamma_m.T * cosh(lambda_m.T) \
                                                           + sin(lambda_m.T) \
                                                           + sinh(lambda_m.T) ) \
                                                           + 2 * cos(lambda_m.T) \
                                                           - 2 * cosh(lambda_m.T) ) /lambda_m.T**2) ) \
             + (m > 2)  * ( (m.T == 1) * (a * (gamma_m * cos(lambda_m) - gamma_m * cosh(lambda_m) \
                                              +          sin(lambda_m) +           sinh(lambda_m) ) \
                                              / lambda_m) \
                          + (m.T == 2) * (- sqrt(3) * a * (- 2 * gamma_m * sin(lambda_m) \
                                                           + 2 * gamma_m * sinh(lambda_m) \
                                                           + lambda_m * (gamma_m * cos(lambda_m) \
                                                           - gamma_m * cosh(lambda_m) \
                                                           + sin(lambda_m) \
                                                           + sinh(lambda_m) ) \
                                                           + 2 * cos(lambda_m) \
                                                           - 2 * cosh(lambda_m) ) /lambda_m**2) \
                          + (m.T > 2)  * ( (m == m.T) * (a * (2 * gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                                                            -     gamma_m**2 * sin(2 * lambda_m) / 2 \
                                                            - 2 * gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                                                            +     gamma_m**2 * sinh(2 * lambda_m) / 2 \
                                                            - 4 * gamma_m * sin(lambda_m) * sinh(lambda_m) \
                                                            + 2 * gamma_m * cos(lambda_m)**2 \
                                                            -     gamma_m * cosh(2 * lambda_m) \
                                                            -     gamma_m \
                                                            + 2 * lambda_m \
                                                            + 2 * sin(lambda_m) * cosh(lambda_m) \
                                                            + sin(2 * lambda_m) / 2 \
                                                            + 2 * cos(lambda_m) * sinh(lambda_m) \
                                                            + sinh(2 * lambda_m) / 2) \
                                                            / (2 * lambda_m) ) \
                                         + (m != m.T) * (- a * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
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
                                                              - gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                              - lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              - lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              - lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                              - lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                              - lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              - lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                              + lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              + lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              - lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                              + lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              + lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) ) \
                                                              / (lambda_m.T**4 - lambda_m**4 + (m == m.T) * float_info.max) ) ) )

    XkXm_m   = - 2 * XkXm / a \
             + (m == 1) * ( (m.T == 1) * (2) \
                          + (m.T == 2) * (- 2 / sqrt(3) ) \
                          + (m.T > 2)  * (4 * (- gamma_m.T * sin(lambda_m.T) \
                                               + gamma_m.T * sinh(lambda_m.T) \
                                 + lambda_m.T * (gamma_m.T * cos(lambda_m.T) \
                                               - gamma_m.T * cosh(lambda_m.T) \
                                                           + sin(lambda_m.T) \
                                                           + sinh(lambda_m.T) ) \
                                                           + cos(lambda_m.T) \
                                                           - cosh(lambda_m.T) ) / lambda_m.T**2) ) \
             + (m == 2) * ( (m.T == 1) * (- 2 / sqrt(3) ) \
                          + (m.T == 2) * (2) \
                          + (m.T > 2)  * (4 * sqrt(3) * (4 * gamma_m.T * cos(lambda_m.T) \
                                                       + 4 * gamma_m.T * cosh(lambda_m.T) \
                                                       - 8 * gamma_m.T \
                                        - lambda_m.T**2 * (gamma_m.T * cos(lambda_m.T) \
                                                         - gamma_m.T * cosh(lambda_m.T) \
                                                                     + sin(lambda_m.T) \
                                                                     + sinh(lambda_m.T) ) \
                                        + 3 * lambda_m.T * (gamma_m.T * sin(lambda_m.T) \
                                                          - gamma_m.T * sinh(lambda_m.T) \
                                                                      - cos(lambda_m.T) \
                                                                      + cosh(lambda_m.T) ) \
                                                                  + 4 * sin(lambda_m.T) \
                                                                  - 4 * sinh(lambda_m.T) ) / lambda_m.T**3) ) \
             + (m > 2)  * ( (m.T == 1) * (4 * (- gamma_m * sin(lambda_m) \
                                               + gamma_m * sinh(lambda_m) \
                                   + lambda_m * (gamma_m * cos(lambda_m) \
                                               - gamma_m * cosh(lambda_m) \
                                                         + sin(lambda_m) \
                                                         + sinh(lambda_m) ) \
                                                         + cos(lambda_m) \
                                                         - cosh(lambda_m) ) / lambda_m**2) \
                          + (m.T == 2) * (4 * sqrt(3) * (4 * gamma_m * cos(lambda_m) \
                                                       + 4 * gamma_m * cosh(lambda_m) \
                                                       - 8 * gamma_m \
                                        - lambda_m**2 * (gamma_m * cos(lambda_m) \
                                                       - gamma_m * cosh(lambda_m) \
                                                                 + sin(lambda_m) \
                                                                 + sinh(lambda_m) ) \
                                        + 3 * lambda_m * (gamma_m * sin(lambda_m) \
                                                        - gamma_m * sinh(lambda_m) \
                                                                  - cos(lambda_m) \
                                                                  + cosh(lambda_m) ) \
                                                              + 4 * sin(lambda_m) \
                                                              - 4 * sinh(lambda_m) ) / lambda_m**3) \
                          + (m.T > 2)  * ( (m == m.T) * ( (4 * gamma_m**2 * lambda_m * sin(lambda_m) * cosh(lambda_m) \
                                                             - gamma_m**2 * lambda_m * sin(2 * lambda_m) \
                                                         - 4 * gamma_m**2 * lambda_m * cos(lambda_m) * sinh(lambda_m) \
                                                             + gamma_m**2 * lambda_m * sinh(2 * lambda_m) \
                                                             - gamma_m**2 * cos(lambda_m)**2 \
                                                         + 4 * gamma_m**2 * cos(lambda_m) * cosh(lambda_m) \
                                                             - gamma_m**2 * cosh(2 * lambda_m) / 2 \
                                                         - 5 * gamma_m**2 / 2 \
                                                         - 8 * gamma_m * lambda_m * sin(lambda_m) * sinh(lambda_m) \
                                                         + 2 * gamma_m * lambda_m * cos(2 * lambda_m) \
                                                         - 2 * gamma_m * lambda_m * cosh(2 * lambda_m) \
                                                         + 4 * gamma_m * sin(lambda_m) * cosh(lambda_m) \
                                                             - gamma_m * sin(2 * lambda_m) \
                                                         - 4 * gamma_m * cos(lambda_m) * sinh(lambda_m) \
                                                             + gamma_m * sinh(2 * lambda_m) \
                                                        + 2 * lambda_m**2 \
                                                        + 4 * lambda_m * sin(lambda_m) * cosh(lambda_m) \
                                                            + lambda_m * sin(2 * lambda_m) \
                                                        + 4 * lambda_m * cos(lambda_m) * sinh(lambda_m) \
                                                            + lambda_m * sinh(2 * lambda_m) \
                                                        - 4 * sin(lambda_m) * sinh(lambda_m) \
                                                        + cos(lambda_m)**2 \
                                                        - cosh(2 * lambda_m) / 2 \
                                                        - 1 / 2) / lambda_m**2) \
                                         + (m != m.T) * (4 * (- gamma_m.T * gamma_m * lambda_m.T**7 * sin(lambda_m) * cos(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**7 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**7 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**7 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**6 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**6 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**6 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**6 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**6 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - 8 * gamma_m.T * gamma_m * lambda_m.T**5 * lambda_m \
                                                              + gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                          + 3 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                          - 3 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                          + 3 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                          - 3 * gamma_m.T * gamma_m * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * cos(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**4 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + 4 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                          - 4 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                          - 4 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                          + 4 * gamma_m.T * gamma_m * lambda_m.T**3 * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**5 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                          + 3 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * sin(lambda_m) \
                                                          + 3 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * sinh(lambda_m) \
                                                          - 3 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * sinh(lambda_m.T) \
                                                          - 3 * gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m) * cos(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m.T * lambda_m**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * cos(lambda_m) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * cosh(lambda_m) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * cos(lambda_m) * cosh(lambda_m.T) \
                                                          + 2 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - 8 * gamma_m.T * gamma_m * lambda_m.T * lambda_m**5 \
                                                              - gamma_m.T * gamma_m * lambda_m**7 * sin(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m**7 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m**7 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * gamma_m * lambda_m**7 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m**6 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m.T * gamma_m * lambda_m**6 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * gamma_m * lambda_m**6 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * gamma_m * lambda_m**6 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**7 * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**7 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**7 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m.T**7 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**6 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**6 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**6 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**6 * sin(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**6 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**6 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**5 * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**5 * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m.T**5 * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          + 2 * gamma_m.T * lambda_m.T**5 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                          + 2 * gamma_m.T * lambda_m.T**5 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                          - 2 * gamma_m.T * lambda_m.T**5 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                          - 2 * gamma_m.T * lambda_m.T**5 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**4 * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**4 * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          - 3 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                          + 3 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          - 3 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          + 3 * gamma_m.T * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m.T * lambda_m.T**3 * lambda_m**4 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**3 * lambda_m**4 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T**3 * lambda_m**4 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          + 4 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                          - 4 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          + 4 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          - 4 * gamma_m.T * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**2 * lambda_m**5 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m.T * lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m.T**2 * lambda_m**5 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          - 3 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * cos(lambda_m) \
                                                          - 3 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          + 3 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          + 3 * gamma_m.T * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * lambda_m.T * lambda_m**6 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m.T * lambda_m**6 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m.T * lambda_m**6 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          + 2 * gamma_m.T * lambda_m.T * lambda_m**5 * sin(lambda_m) * cos(lambda_m.T) \
                                                          + 2 * gamma_m.T * lambda_m.T * lambda_m**5 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          - 2 * gamma_m.T * lambda_m.T * lambda_m**5 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          - 2 * gamma_m.T * lambda_m.T * lambda_m**5 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m**7 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m.T * lambda_m**7 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m**7 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m.T * lambda_m**7 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m.T * lambda_m**6 * sin(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m.T * lambda_m**6 * sin(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m.T * lambda_m**6 * cos(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m.T * lambda_m**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**7 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m * lambda_m.T**7 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m * lambda_m.T**7 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**7 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m * lambda_m.T**6 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m * lambda_m.T**6 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m.T**6 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**6 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**6 * sin(lambda_m) * cos(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**6 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**6 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**5 * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**5 * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m)\
                                                          + 2 * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                          + 2 * gamma_m * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                          - 2 * gamma_m * lambda_m.T**5 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                          - 2 * gamma_m * lambda_m.T**5 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m * lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**4 * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**4 * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - 3 * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                          - 3 * gamma_m * lambda_m.T**4 * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          + 3 * gamma_m * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          + 3 * gamma_m * lambda_m.T**4 * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * sin(lambda_m) \
                                                              + gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m.T**3 * lambda_m**4 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T**3 * lambda_m**4 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          + 4 * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                          - 4 * gamma_m * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          + 4 * gamma_m * lambda_m.T**3 * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          - 4 * gamma_m * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * cos(lambda_m) \
                                                              + gamma_m * lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m.T**2 * lambda_m**5 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m.T**2 * lambda_m**5 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                          - 3 * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * cos(lambda_m.T) \
                                                          + 3 * gamma_m * lambda_m.T**2 * lambda_m**4 * sin(lambda_m) * cosh(lambda_m.T) \
                                                          - 3 * gamma_m * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * sinh(lambda_m) \
                                                          + 3 * gamma_m * lambda_m.T**2 * lambda_m**4 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * sin(lambda_m) \
                                                              - gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m.T) * sinh(lambda_m) \
                                                              - gamma_m * lambda_m.T * lambda_m**6 * sin(lambda_m) * sinh(lambda_m.T) \
                                                              + gamma_m * lambda_m.T * lambda_m**6 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                          + 2 * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * cos(lambda_m) \
                                                          + 2 * gamma_m * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * cosh(lambda_m) \
                                                          - 2 * gamma_m * lambda_m.T * lambda_m**5 * cos(lambda_m) * sinh(lambda_m.T) \
                                                          - 2 * gamma_m * lambda_m.T * lambda_m**5 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m**7 * cos(lambda_m.T) * cos(lambda_m) \
                                                              - gamma_m * lambda_m**7 * cos(lambda_m.T) * cosh(lambda_m) \
                                                              + gamma_m * lambda_m**7 * cos(lambda_m) * cosh(lambda_m.T) \
                                                              - gamma_m * lambda_m**7 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                              - gamma_m * lambda_m**6 * sin(lambda_m) * cos(lambda_m.T) \
                                                              - gamma_m * lambda_m**6 * sin(lambda_m) * cosh(lambda_m.T) \
                                                              + gamma_m * lambda_m**6 * cos(lambda_m.T) * sinh(lambda_m) \
                                                              + gamma_m * lambda_m**6 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                        + lambda_m.T**7 * sin(lambda_m.T) * cos(lambda_m) \
                                                                        + lambda_m.T**7 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                        + lambda_m.T**7 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                        + lambda_m.T**7 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T**6 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                                        + lambda_m.T**6 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                                        + lambda_m.T**6 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                                        - lambda_m.T**6 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                        + lambda_m.T**6 * cos(lambda_m.T) * cos(lambda_m) \
                                                                        + lambda_m.T**6 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T**6 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                        - lambda_m.T**6 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                        + lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                                        - lambda_m.T**5 * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T**5 * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                        + lambda_m.T**5 * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                    + 2 * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                                    - 2 * lambda_m.T**5 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                                    - 2 * lambda_m.T**5 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                                    + 2 * lambda_m.T**5 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                        - lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                        - lambda_m.T**4 * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                        - lambda_m.T**4 * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                        - lambda_m.T**4 * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                    + 3 * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                                    - 3 * lambda_m.T**4 * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                    + 3 * lambda_m.T**4 * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                    - 3 * lambda_m.T**4 * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * cos(lambda_m) \
                                                                        - lambda_m.T**3 * lambda_m**4 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T**3 * lambda_m**4 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                        - lambda_m.T**3 * lambda_m**4 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                    + 4 * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                    + 4 * lambda_m.T**3 * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                    + 4 * lambda_m.T**3 * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                    + 4 * lambda_m.T**3 * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                        + lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * cos(lambda_m.T) \
                                                                        - lambda_m.T**2 * lambda_m**5 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                        - lambda_m.T**2 * lambda_m**5 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                        + lambda_m.T**2 * lambda_m**5 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                    + 3 * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * cos(lambda_m) \
                                                                    + 3 * lambda_m.T**2 * lambda_m**4 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                    - 3 * lambda_m.T**2 * lambda_m**4 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                    - 3 * lambda_m.T**2 * lambda_m**4 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                        - lambda_m.T * lambda_m**6 * sin(lambda_m.T) * cos(lambda_m) \
                                                                        + lambda_m.T * lambda_m**6 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                        + lambda_m.T * lambda_m**6 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                        - lambda_m.T * lambda_m**6 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                    + 2 * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * sin(lambda_m) \
                                                                    - 2 * lambda_m.T * lambda_m**5 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                    - 2 * lambda_m.T * lambda_m**5 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                    + 2 * lambda_m.T * lambda_m**5 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                        + lambda_m**7 * sin(lambda_m) * cos(lambda_m.T) \
                                                                        + lambda_m**7 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                        + lambda_m**7 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                        + lambda_m**7 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                        + lambda_m**6 * cos(lambda_m.T) * cos(lambda_m) \
                                                                        - lambda_m**6 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                        + lambda_m**6 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                        - lambda_m**6 * cosh(lambda_m.T) * cosh(lambda_m) ) \
                                                          / (lambda_m.T**8 - 2 * lambda_m.T**4 * lambda_m**4 + lambda_m**8 \
                                                          + (m == m.T) * float_info.max) ) ) )

    XkddXm   = (m == 1) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m == 2) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m > 2)  * ( (m.T == 1) * (- lambda_m * (gamma_m * cos(lambda_m) \
                                                      + gamma_m * cosh(lambda_m) \
                                                      - 2 * gamma_m \
                                                      + sin(lambda_m) \
                                                      - sinh(lambda_m) ) / a) \
                          + (m.T == 2) * (sqrt(3) * (gamma_m * lambda_m * cos(lambda_m) \
                                                   + gamma_m * lambda_m * cosh(lambda_m) \
                                                   + 2 * gamma_m * lambda_m \
                                                   - 2 * gamma_m * sin(lambda_m) \
                                                   - 2 * gamma_m * sinh(lambda_m) \
                                                   + lambda_m * sin(lambda_m) \
                                                   - lambda_m * sinh(lambda_m) \
                                                   + 2 * cos(lambda_m) \
                                                   + 2 * cosh(lambda_m) \
                                                   - 4) / a) \
                          + (m.T > 2)  * ( (m == m.T) * (- lambda_m * (2 * gamma_m**2 * lambda_m \
                                                                    -      gamma_m**2 * sin(2 * lambda_m) / 2 \
                                                                    -      gamma_m**2 * sinh(2 * lambda_m) / 2 \
                                                                    +      gamma_m * cos(2 * lambda_m) \
                                                                    +      gamma_m * cosh(2 * lambda_m) \
                                                                    -  2 * gamma_m + sin(2 * lambda_m) / 2 \
                                                                    - sinh(2 * lambda_m) / 2) / (2 * a) ) \
                                         + (m != m.T) * (lambda_m**2 * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                      + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                      - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                      - gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                      - gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
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
                                                                      - gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                      + gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                      - gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                      + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                                      - gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                                      - gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                                      + gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                      + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                                      + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                      - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                      - gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                      + gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                      + gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                      + gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                      + gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                      - 4 * gamma_m * lambda_m**3 - lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                      + lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                      - lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                      + lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                      + lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                                      - lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                                      + lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                                      - lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                      - lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                                      - lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                      + lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                      + lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                      + lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                      + lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                      - lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                      - lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) ) \
                                                                      / (a * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) ) ) )

    dXkdXm   = (m == 1) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m == 2) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (12) \
                          + (m.T > 2)  * (2 * sqrt(3) * (gamma_m.T * sin(lambda_m.T) \
                                                       + gamma_m.T * sinh(lambda_m.T) \
                                                       - cos(lambda_m.T) \
                                                       - cosh(lambda_m.T) + 2) \
                                                       / a) ) \
             + (m > 2)  * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (2 * sqrt(3) * (gamma_m * sin(lambda_m) \
                                                       + gamma_m * sinh(lambda_m) \
                                                       - cos(lambda_m) \
                                                       - cosh(lambda_m) + 2) \
                                                       / a) \
                          + (m.T > 2)  * ( (m == m.T) * (lambda_m * (2 * gamma_m**2 * lambda_m \
                                                                   + 2 * gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                                                                   +     gamma_m**2 * sin(2 * lambda_m) / 2 \
                                                                   + 2 * gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                                                                   +     gamma_m**2 * sinh(2 * lambda_m) / 2 \
                                                                   - 4 * gamma_m * cos(lambda_m) * cosh(lambda_m) \
                                                                   -     gamma_m * cos(2 * lambda_m) \
                                                                   -     gamma_m * cosh(2 * lambda_m) \
                                                                   + 6 * gamma_m \
                                                                   - 2 * sin(lambda_m) * cosh(lambda_m) \
                                                                   - sin(2 * lambda_m) / 2 \
                                                                   + 2 * cos(lambda_m) * sinh(lambda_m) \
                                                                   + sinh(2 * lambda_m) / 2) / (2 * a) ) \
                                         + (m != m.T) * (lambda_m.T * lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
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
                                                                                - gamma_m.T * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                + gamma_m.T * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                - gamma_m.T * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                                                - gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                - gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                + gamma_m.T * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                - gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                + gamma_m.T * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                + gamma_m.T * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                + gamma_m.T * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                - 4 * gamma_m.T * lambda_m**3 \
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
                                                                                / (a * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) ) ) )

    ddXkddXm = (m == 1) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m == 2) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m > 2)  * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * ( (m == m.T) * (lambda_m**3 * (- gamma_m**2 * sin(lambda_m) * cosh(lambda_m) \
                                                                        - gamma_m**2 * sin(2 * lambda_m) / 4 \
                                                                        + gamma_m**2 * cos(lambda_m) * sinh(lambda_m) \
                                                                        + gamma_m**2 * sinh(2 * lambda_m) / 4 \
                                                                        + 2 * gamma_m * sin(lambda_m) * sinh(lambda_m) \
                                                                        + gamma_m * cos(2 * lambda_m) / 2 \
                                                                        - gamma_m * cosh(2 * lambda_m) / 2 \
                                                                        + lambda_m \
                                                                        - sin(lambda_m) * cosh(lambda_m) \
                                                                        + sin(2 * lambda_m) / 4 \
                                                                        - cos(lambda_m) * sinh(lambda_m) \
                                                                        + sinh(2 * lambda_m) / 4) / a**3) \
                                         + (m != m.T) * (- lambda_m.T**2 * lambda_m**2 * (gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                                        + gamma_m.T * gamma_m * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                                                        + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                        + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                        + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                        + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                        - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                        - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                                        - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                        + gamma_m.T * gamma_m * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                        + gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                        + gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                        + gamma_m.T * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                        - gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                        - gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                        + gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                        + gamma_m.T * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                        + gamma_m.T * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                        + gamma_m.T * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                        - gamma_m.T * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                        + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                        - gamma_m * lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                        - gamma_m * lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                        + gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                        + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                                                        - gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                        + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                        - gamma_m * lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                        + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                        + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                        + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                        + gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                        + gamma_m * lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                        + gamma_m * lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                        - gamma_m * lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                        - gamma_m * lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
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
                                                                                        / (a**3 * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) ) ) )

    XkdXm    = (m == 1) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m == 2) * ( (m.T == 1) * (2 * sqrt(3) ) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (- sqrt(3) * a * (- 2 * gamma_m.T * sin(lambda_m.T) \
                                                           + 2 * gamma_m.T * sinh(lambda_m.T) \
                                                           + lambda_m.T * (gamma_m.T * cos(lambda_m.T) \
                                                           - gamma_m.T * cosh(lambda_m.T) \
                                                           + sin(lambda_m.T) \
                                                           + sinh(lambda_m.T) ) \
                                                           + 2 * cos(lambda_m.T) \
                                                           - 2 * cosh(lambda_m.T) ) \
                                                           / lambda_m.T**2) ) \
             + (m > 2)  * ( (m.T == 1) * (- gamma_m * sin(lambda_m) \
                                          - gamma_m * sinh(lambda_m) \
                                          + cos(lambda_m) \
                                          + cosh(lambda_m) \
                                          - 2) \
                          + (m.T == 2) * (sqrt(3) * (2 * gamma_m * cos(lambda_m) \
                                                   - 2 * gamma_m * cosh(lambda_m) \
                                                   + lambda_m * (gamma_m * sin(lambda_m) \
                                                   + gamma_m * sinh(lambda_m) \
                                                   - cos(lambda_m) \
                                                   - cosh(lambda_m) ) \
                                                   - 2 * lambda_m \
                                                   + 2 * sin(lambda_m) \
                                                   + 2 * sinh(lambda_m) ) \
                                                   / lambda_m) \
                          + (m.T > 2)  * ( (m == m.T) * (gamma_m**2 * sin(lambda_m)**2 / 2 \
                                                       + gamma_m**2 * sin(lambda_m) * sinh(lambda_m) \
                                                       + gamma_m**2 * cosh(2 * lambda_m) / 4 \
                                                       - gamma_m**2 / 4 \
                                                       - gamma_m * sin(lambda_m) * cosh(lambda_m) \
                                                       - gamma_m * sin(2 * lambda_m) / 2 \
                                                       - gamma_m * cos(lambda_m) * sinh(lambda_m) \
                                                       - gamma_m * sinh(2 * lambda_m) / 2 \
                                                       - sin(lambda_m)**2 / 2 \
                                                       + cos(lambda_m) * cosh(lambda_m) \
                                                       + cosh(2 * lambda_m) / 4 \
                                                       - 5 / 4) \
                                         + (m != m.T) * (- lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                                     + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                     + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                     + gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                                     + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                     + gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                     + gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                     - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                     - gamma_m.T * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                     - gamma_m.T * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                     + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                     + gamma_m * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                     + gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                     - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                                     + gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                                     + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                                     - gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                     + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                                     - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                     - gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                     + gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                     - gamma_m * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                     - gamma_m * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                     - gamma_m * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                     - gamma_m * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                     + lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                     - lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                     + lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                     - lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                     + lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                                     - lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                                     - lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                                     + lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                     + lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                                     + lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                     - lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                     - lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                     + lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                     + lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                     + lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                     + lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                     - 4 * lambda_m**3) / (lambda_m.T**4 - lambda_m**4 + (m == m.T) * float_info.max) ) ) )

    ddXkdXm  = (m == 1) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (0) ) \
             + (m == 2) * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * (2 * sqrt(3) * lambda_m.T * (gamma_m.T * cos(lambda_m.T) \
                                                                    + gamma_m.T * cosh(lambda_m.T) \
                                                                    - 2 * gamma_m.T \
                                                                    + sin(lambda_m.T) \
                                                                    - sinh(lambda_m.T) ) / a**2) ) \
             + (m > 2)  * ( (m.T == 1) * (0) \
                          + (m.T == 2) * (0) \
                          + (m.T > 2)  * ( (m == m.T) * (- lambda_m**2 * (gamma_m**2 * sin(lambda_m)**2 / 2 \
                                                         - gamma_m**2 * cos(lambda_m) * cosh(lambda_m) \
                                                         - gamma_m**2 * cosh(2 * lambda_m) / 4 \
                                                         + 5 * gamma_m**2 / 4 \
                                                         - gamma_m * sin(lambda_m) * cosh(lambda_m) \
                                                         - gamma_m * sin(2 * lambda_m) / 2 \
                                                         + gamma_m * cos(lambda_m) * sinh(lambda_m) \
                                                         + gamma_m * sinh(2 * lambda_m) / 2 \
                                                         - sin(lambda_m)**2 / 2 \
                                                         + sin(lambda_m) * sinh(lambda_m) \
                                                         - cosh(2 * lambda_m) / 4 \
                                                         + 1 / 4) / a**2) \
                                         + (m != m.T) * (lambda_m.T**2 * lambda_m * (gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T**3 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                   - 4 * gamma_m.T * gamma_m * lambda_m.T**3 \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sin(lambda_m) \
                                                                                   - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                   - gamma_m.T * gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                   - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                   - gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                   + gamma_m.T * gamma_m * lambda_m.T * lambda_m**2 * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                   + gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                   - gamma_m.T * gamma_m * lambda_m**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                   - gamma_m.T * gamma_m * lambda_m**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                   + gamma_m.T * lambda_m.T**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                                   + gamma_m.T * lambda_m.T**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                   - gamma_m.T * lambda_m.T**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                   - gamma_m.T * lambda_m.T**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                   - gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cos(lambda_m) \
                                                                                   + gamma_m.T * lambda_m.T**2 * lambda_m * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                   - gamma_m.T * lambda_m.T**2 * lambda_m * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                   + gamma_m.T * lambda_m.T**2 * lambda_m * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * cos(lambda_m.T) \
                                                                                   - gamma_m.T * lambda_m.T * lambda_m**2 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                   + gamma_m.T * lambda_m.T * lambda_m**2 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                   - gamma_m.T * lambda_m.T * lambda_m**2 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                   - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                                   - gamma_m.T * lambda_m**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m.T * lambda_m**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                   + gamma_m.T * lambda_m**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cos(lambda_m) \
                                                                                   + gamma_m * lambda_m.T**3 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                   - gamma_m * lambda_m.T**3 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                   - gamma_m * lambda_m.T**3 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                   - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cos(lambda_m.T) \
                                                                                   - gamma_m * lambda_m.T**2 * lambda_m * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                   + gamma_m * lambda_m.T**2 * lambda_m * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                   + gamma_m * lambda_m.T**2 * lambda_m * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                   + gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cos(lambda_m) \
                                                                                   - gamma_m * lambda_m.T * lambda_m**2 * sin(lambda_m.T) * cosh(lambda_m) \
                                                                                   + gamma_m * lambda_m.T * lambda_m**2 * cos(lambda_m) * sinh(lambda_m.T) \
                                                                                   - gamma_m * lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * cosh(lambda_m) \
                                                                                   - gamma_m * lambda_m**3 * sin(lambda_m) * cos(lambda_m.T) \
                                                                                   + gamma_m * lambda_m**3 * sin(lambda_m) * cosh(lambda_m.T) \
                                                                                   - gamma_m * lambda_m**3 * cos(lambda_m.T) * sinh(lambda_m) \
                                                                                   + gamma_m * lambda_m**3 * sinh(lambda_m) * cosh(lambda_m.T) \
                                                                                   + lambda_m.T**3 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                   - lambda_m.T**3 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                   - lambda_m.T**3 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                   + lambda_m.T**3 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                   + lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cos(lambda_m) \
                                                                                   - lambda_m.T**2 * lambda_m * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                   + lambda_m.T**2 * lambda_m * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                   - lambda_m.T**2 * lambda_m * cosh(lambda_m.T) * cosh(lambda_m) \
                                                                                   + lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sin(lambda_m) \
                                                                                   + lambda_m.T * lambda_m**2 * sin(lambda_m.T) * sinh(lambda_m) \
                                                                                   + lambda_m.T * lambda_m**2 * sin(lambda_m) * sinh(lambda_m.T) \
                                                                                   + lambda_m.T * lambda_m**2 * sinh(lambda_m.T) * sinh(lambda_m) \
                                                                                   + lambda_m**3 * cos(lambda_m.T) * cos(lambda_m) \
                                                                                   + lambda_m**3 * cos(lambda_m.T) * cosh(lambda_m) \
                                                                                   - lambda_m**3 * cos(lambda_m) * cosh(lambda_m.T) \
                                                                                   - lambda_m**3 * cosh(lambda_m.T) * cosh(lambda_m) ) \
                                                                                    / (a**2 * (lambda_m.T**4 - lambda_m**4) + (m == m.T) * float_info.max) ) ) )

    return(xm, Xm, XkXm, XkXm_m, XkddXm, dXkdXm, ddXkddXm, XkdXm, ddXkdXm)