def PolynomialApproximationClampedClamped(a, m, nodes_x):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from numpy import zeros

    length = len(m)

    xm = zeros( [nodes_x.shape[0], nodes_x.shape[1], length] )

    for i in range(length):
        xm[:, :, i] = (nodes_x**2 - a * nodes_x)**2 * nodes_x**(m[0, i] - 1)

    Xm       = a**(m[0, :] + 4) * 2 \
             / ( (m[0, :] + 2) * (m[0, :] + 3) * (m[0, :] + 4) )

    XkXm     = a**(m + m.T + 7) * 24 \
             / ( (m + m.T + 3) * (m + m.T + 4) * (m + m.T + 5) * (m + m.T + 6) * (m + m.T + 7) )

    XkXm_m   = a**(m + m.T + 6) * 96 \
             / ( (m + m.T + 4) * (m + m.T + 5) * (m + m.T + 6) * (m + m.T + 7) * (m + m.T + 8) )

    XkddXm   = a**(m + m.T + 5) * 4 * (m**2 + m.T**2 - 4 * m * m.T - 3 * m - 3 * m.T - 4) \
             / ( (m + m.T + 1) * (m + m.T + 2) * (m + m.T + 3) * (m + m.T + 4) * (m + m.T + 5) )

    dXkdXm   = - XkddXm

    ddXkddXm = a**(m + m.T + 3) * 24 * m * m.T * (m + 1) * (m.T + 1) \
             / ( (m + m.T - 1) * (m + m.T) * (m + m.T + 1) * (m + m.T + 2) * (m + m.T + 3) )

    XkdXm    = a**(m + m.T + 6) * 12 * (m - m.T) \
             / ( (m + m.T + 2) * (m + m.T + 3) * (m + m.T + 4) * (m + m.T + 5) * (m + m.T + 6) )

    ddXkdXm  = a**(m + m.T + 4) * 12 * (m - m.T) * (m + 1) * (m.T + 1) \
             / ( (m + m.T) * (m + m.T + 1) * (m + m.T + 2) * (m + m.T + 3) * (m + m.T + 4) )

    return(xm, Xm, XkXm, XkXm_m, XkddXm, dXkdXm, ddXkddXm, XkdXm, ddXkdXm)