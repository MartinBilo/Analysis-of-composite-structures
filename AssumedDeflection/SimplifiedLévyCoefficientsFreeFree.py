def SimplifiedLévyCoefficientsFreeFree(n):
    """[summary]

    Args:
        a ([type]): [description]
        m ([type]): [description]
        nodes_x ([type]): [description]
    """

    # Import packages into library
    from numpy import zeros

    length = len(n)

    A_n = zeros(length)

    B_n = zeros(length)

    C_n = zeros(length)

    D_n = zeros(length)

    return(A_n, B_n, C_n, D_n)