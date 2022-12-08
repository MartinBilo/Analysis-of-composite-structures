def BeamApproximationSimplySupportedSimplySupported(a, m, nodes_x):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from math  import pi
    from numpy import einsum, sin
    from sys   import float_info

    xm       = sin(einsum('k,ij->ijk', m[0, :] * pi / a, nodes_x) )

    Xm       = a / (m[0, :] * pi) * (1 - (- 1)**m[0, :] )

    XkXm     = (m == m.T) * a / 2

    XkXm_m   = (m != m.T) * 48 * m * m.T * (1 - (- 1)**(m + m.T) ) / (pi**2 * a * (m**4 - 2*m**2 * m.T**2 + m.T**4) \
             + (m == m.T) * float_info.max)

    XkddXm   = (m == m.T) * - m**2 * pi**2 / (2 * a)

    dXkdXm   = - XkddXm

    ddXkddXm = (m == m.T) * m**4 * pi**4 / (2 * a**3)

    XkdXm    = (m != m.T) * m * m.T / (m**2 - m.T**2 + (m == m.T) * float_info.max) * (1 - (- 1)**m * (- 1)**m.T)

    ddXkdXm  = (m != m.T) * - m**2 * m.T * pi**2 / (a**2 * (m**2 - m.T**2) + (m == m.T) * float_info.max) * (1 - (- 1)**m * (- 1)**m.T)

    return(xm, Xm, XkXm, XkXm_m, XkddXm, dXkdXm, ddXkddXm, XkdXm, ddXkdXm)