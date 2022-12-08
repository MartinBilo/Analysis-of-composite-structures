def PointLoad_SpeciallyOrthotropic(assumeddeflection, force, geometry, mesh, settings, stiffness, data=False, illustrations=False):

    ## Finicky methods with bounds on a, b and D and require special orthotopic laminate

    # Import packages into library
    from cmath           import sqrt
    from datetime import datetime
    from matplotlib import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.ticker import MaxNLocator
    from math            import inf, isclose, pi
    from numpy           import reshape, real, sum, cos, cosh, sin, sinh, imag, arange, array, copy, einsum, linspace, tile, ones, zeros
    from numpy.linalg    import solve
    from os              import remove
    from os.path         import exists
    from statistics      import median
    from tabulate        import tabulate
    from textwrap        import fill

    ##
    #
    from AssumedDeflection.LévyCoefficientsSimplySupportedSimplySupported import LévyCoefficientsSimplySupportedSimplySupported
    from AssumedDeflection.LévyCoefficientsClampedSimplySupported         import LévyCoefficientsClampedSimplySupported
    from AssumedDeflection.LévyCoefficientsFreeSimplySupported            import LévyCoefficientsFreeSimplySupported
    from AssumedDeflection.LévyCoefficientsClampedClamped                 import LévyCoefficientsClampedClamped
    from AssumedDeflection.LévyCoefficientsFreeFree                       import LévyCoefficientsFreeFree
    from AssumedDeflection.LévyCoefficientsClampedFree                    import LévyCoefficientsClampedFree

    ##
    #
    xm       = assumeddeflection['xm']
    yn       = assumeddeflection['yn']
    XkXm     = assumeddeflection['XkXm']
    YlYn     = assumeddeflection['YlYn']
    XkddXm   = assumeddeflection['XkddXm']
    YlddYn   = assumeddeflection['YlddYn']
    dXkdXm   = assumeddeflection['dXkdXm']
    dYldYn   = assumeddeflection['dYldYn']
    ddXkddXm = assumeddeflection['ddXkddXm']
    ddYlddYn = assumeddeflection['ddYlddYn']
    XkdXm    = assumeddeflection['XkdXm']
    YldYn    = assumeddeflection['YldYn']
    ddXkdXm  = assumeddeflection['ddXkdXm']
    ddYldYn  = assumeddeflection['ddYldYn']

    ##
    #
    xi = force['xi']
    #
    eta = force['eta']
    #
    F_z = force['F_z']

    ## Import geometry properties
    #
    a = geometry['a']
    #
    b = geometry['b']

    ##
    #
    nodes_x = mesh['nodes_x']
    #
    nodes_y = mesh['nodes_y']

    ##
    #
    fileformat           = settings['fileformat']
    number_of_elements_x = settings['number_of_elements_x']
    number_of_elements_y = settings['number_of_elements_y']
    m                    = settings['m']
    margin               = settings['margin']
    mn_max               = settings['mn_max']
    n                    = settings['n']
    simulation           = settings['simulation']

    ## Import stiffness properties
    #
    B = stiffness['B']
    #
    D = stiffness['D']

    ##
    #
    BoundaryConditions_Navier = [f'Simply supported']

    #
    BoundaryConditions_Ritz   = [f'Simply supported',
                                 f'Simply supported (x = 0,a; y = 0); clamped (y = b)',
                                 f'Simply supported (y = 0,b; x = 0); clamped (x = a)',
                                 f'Simply supported (x = 0,a); clamped (y = 0,b)',
                                 f'Simply supported (x = 0,a); clamped (y = 0,b) [Polynomial]',
                                 f'Simply supported (y = 0,b); clamped (x = 0,a)',
                                 f'Simply supported (y = 0,b); clamped (x = 0,a) [Polynomial]',
                                 f'Simply supported (x = a; y = b); clamped (x = 0; y = 0)',
                                 f'Simply supported (x = a); clamped (x = 0); clamped (y = 0,b)',
                                 f'Simply supported (x = a); clamped (x = 0); clamped (y = 0,b) [Polynomial]',
                                 f'Simply supported (y = b); clamped (y = 0); clamped (x = 0,a)',
                                 f'Simply supported (y = b); clamped (y = 0); clamped (x = 0,a) [Polynomial]',
                                 f'Clamped',
                                 f'Clamped (x = 0,a); clamped (y = 0,b) [Polynomial]',
                                 f'Clamped (x = 0,a) [Polynomial]; clamped (y = 0,b)',
                                 f'Clamped [Polynomial]',
                                 f'Simply supported (x = 0,a); free (y = 0,b)',
                                 f'Simply supported (y = 0,b); free (x = 0,a)',
                                 f'Simply supported (x = a); clamped (x = 0); free (y = 0,b)',
                                 f'Simply supported (y = b); clamped (y = 0); free (x = 0,a)',
                                 f'Clamped (x = 0,a); free (y = 0,b)',
                                 f'Clamped (x = 0,a) [Polynomial]; free (y = 0,b)',
                                 f'Clamped (y = 0,b); free (x = 0,a)',
                                 f'Clamped (y = 0,b) [Polynomial]; free (x = 0,a)' ]

    #
    BoundaryConditions_Lévy   = [f'Simply supported (x = 0,a); simply supported (y = 0,b)',
                                 f'Simply supported (y = 0,b); simply supported (x = 0,a)',
                                 f'Simply supported (x = 0,a; y = 0); clamped (y = b)',
                                 f'Simply supported (y = 0,b; x = 0); clamped (x = a)',
                                 f'Simply supported (x = 0,a); clamped (y = 0,b)',
                                 f'Simply supported (y = 0,b); clamped (x = 0,a)',
                                 f'Simply supported (x = 0,a); free (y = 0,b)',
                                 f'Simply supported (y = 0,b); free (x = 0,a)',
                                 f'Simply supported (x = 0,a; y = b); free (y = 0)',
                                 f'Simply supported (y = 0,b; x = a); free (x = 0)',
                                 f'Simply supported (x = 0,a); clamped (y = 0); free (y = b)',
                                 f'Simply supported (y = 0,b); clamped (x = 0); free (x = a)' ]

    #
    Situations_Navier = len(BoundaryConditions_Navier)

    #
    Situations_Ritz = len(BoundaryConditions_Ritz)

    #
    Situations_Lévy = len(BoundaryConditions_Lévy)

    #
    BC_Ritz = array( [ [1, 1],
                       [1, 2],
                       [2, 1],
                       [1, 3],
                       [1, 4],
                       [3, 1],
                       [4, 1],
                       [2, 2],
                       [2, 3],
                       [2, 4],
                       [3, 2],
                       [4, 2],
                       [3, 3],
                       [3, 4],
                       [4, 3],
                       [4, 4],
                       [1, 0],
                       [0, 1],
                       [2, 0],
                       [0, 2],
                       [3, 0],
                       [4, 0],
                       [0, 3],
                       [0, 4] ] )

    #
    BC_Lévy = tile(array( [0, 1] ), Situations_Lévy // 2)

    #
    error_Navier    = zeros( [mn_max, Situations_Navier] )
    error_Navier[0] = inf

    #
    error_Ritz    = zeros( [mn_max, Situations_Ritz] )
    error_Ritz[0] = inf

    #
    error_Lévy    = zeros( [mn_max, Situations_Lévy] )
    error_Lévy[0] = inf

    #
    delta_Navier = zeros( [mn_max, Situations_Navier] )

    #
    delta_Ritz = zeros( [mn_max, Situations_Ritz] )

    #
    delta_Lévy = zeros( [mn_max, Situations_Lévy] )

    #
    mn_Ritz = zeros(Situations_Ritz, dtype = int)

    #
    mn_Lévy = zeros(Situations_Lévy, dtype = int)

    #
    x_Ritz = zeros(Situations_Ritz)
    y_Ritz = zeros(Situations_Ritz)

    #
    x_Lévy = zeros(Situations_Lévy)
    y_Lévy = zeros(Situations_Lévy)

    ## Navier solution
    #
    M = n[0:mn_max, 0:mn_max].T

    # Equation 5.5 and 5.16 from [1]
    q_mn = 4 * F_z / (a * b) * sin(M * pi * xi / a) * sin(M.T * pi * eta / b)

    # Equation 5.7 from [1]
    D_mn = D[0, 0] * M**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * (M * M.T * a / b)**2 + D[1, 1] * (M.T * a / b)**4

    # Equation 5.6 and 5.7 from [1]
    A_mn = a**4 / pi**4 * q_mn / D_mn

    #
    counter = 0

    #
    mn = 1

    #
    while error_Navier[counter] >= margin and mn <= mn_max:

        #
        counter = (mn != 1) * (counter + 1)

        # Equation 5.6 from [1]
        w =     einsum('ij,ijkl->kl', A_mn[0:mn, 0:mn], \
            sin(einsum('ij,kl->ijkl', M[0:mn, 0:mn]   * pi / a, nodes_x) ) * \
            sin(einsum('ij,kl->ijkl', M[0:mn, 0:mn].T * pi / b, nodes_y) ) )

        # Maximum deflection
        delta_Navier[counter] = max(w.flatten(), key=abs)

        # Normalized, maximum norm
        if counter == 0:
            error_Navier[counter] = 100
        else:
            error_Navier[counter] = real(sqrt( (w - W).flatten() @ (w - W).flatten() / (W.flatten() @ W.flatten() ) ) ) * 100

        #
        W = w.copy()

        # Based on equation 5.16 from [1] it can be concluded that the even terms do not contribute to the deflection of the plate due
        # to a pressure load
        mn += 2

    #
    mn_Navier = mn - 2

    #
    x_Navier = median(nodes_x[w == delta_Navier[counter] ] )
    y_Navier = median(nodes_y[w == delta_Navier[counter] ] )


    ## Ritz method
    #
    Xm = zeros( [len(m), 5] )

    lambda_m = (M.flatten() == 1) * 4.712 \
             + (M.flatten() == 2) * 7.854 \
             + (M.flatten() > 2)  * (2 * M.flatten() + 1) * pi / 2
    gamma_m = (cosh(lambda_m) - cos(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )
    Xm[M.flatten() == 1, 0] = 1
    Xm[M.flatten() == 2, 0] = real(sqrt(3) ) * (1 - 2 * xi / a)
    Xm[M.flatten() > 2, 0]  =                            cosh(lambda_m[M.flatten() > 2] / a * xi) \
                            +                             cos(lambda_m[M.flatten() > 2] / a * xi) \
                            - gamma_m[M.flatten() > 2] * sinh(lambda_m[M.flatten() > 2] / a * xi) \
                            - gamma_m[M.flatten() > 2] *  sin(lambda_m[M.flatten() > 2] / a * xi)

    Xm[:, 1] = sin(M.flatten() * pi * xi / a)

    lambda_m = (M.flatten() + 0.25) * pi
    gamma_m  = (sin(lambda_m) - sinh(lambda_m) ) / (cosh(lambda_m) - cos(lambda_m) )
    Xm[:, 2] = gamma_m *  cos(lambda_m / a * xi) \
             - gamma_m * cosh(lambda_m / a * xi) \
             +            sin(lambda_m / a * xi) \
             -           sinh(lambda_m / a * xi)

    lambda_m = (M.flatten() == 1) * 4.712 \
             + (M.flatten() == 2) * 7.854 \
             + (M.flatten() > 2)  * (2 * M.flatten() + 1) * pi / 2
    gamma_m = (cos(lambda_m) - cosh(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )
    Xm[:, 3] = gamma_m *  cos(lambda_m / a * xi) \
             - gamma_m * cosh(lambda_m / a * xi) \
             +            sin(lambda_m / a * xi) \
             -           sinh(lambda_m / a * xi)

    Xm[:, 4] = (xi**2 - a * xi)**2 * xi**(M.flatten() - 1)

    #
    Yn = zeros( [len(M.flatten() ), 5] )

    lambda_n = (M.T.flatten() == 1) * 4.712 \
             + (M.T.flatten() == 2) * 7.854 \
             + (M.T.flatten() > 2)  * (2 * M.T.flatten() + 1) * pi / 2
    gamma_n = (cosh(lambda_n) - cos(lambda_n) ) / (sin(lambda_n) + sinh(lambda_n) )
    Yn[M.T.flatten() == 1, 0] = 1
    Yn[M.T.flatten() == 2, 0] = real(sqrt(3) ) * (1 - 2 * eta / b)
    Yn[M.T.flatten() > 2, 0]  =                           cosh(lambda_n[M.T.flatten() > 2] / b * eta) \
                           +                               cos(lambda_n[M.T.flatten() > 2] / b * eta) \
                           - gamma_n[M.T.flatten() > 2] * sinh(lambda_n[M.T.flatten() > 2] / b * eta) \
                           - gamma_n[M.T.flatten() > 2] *  sin(lambda_n[M.T.flatten() > 2] / b * eta)

    Yn[:, 1] = sin(M.T.flatten() * pi * eta / b)

    lambda_n = (M.T.flatten() + 0.25) * pi
    gamma_n  = (sin(lambda_n) - sinh(lambda_n) ) / (cosh(lambda_n) - cos(lambda_n) )
    Yn[:, 2] = gamma_n *  cos(lambda_n / b * eta) \
       - gamma_n * cosh(lambda_n / b * eta) \
       +            sin(lambda_n / b * eta) \
        -           sinh(lambda_n / b * eta)

    lambda_n = (M.T.flatten() == 1) * 4.712 \
             + (M.T.flatten() == 2) * 7.854 \
             + (M.T.flatten() > 2)  * (2 * M.T.flatten() + 1) * pi / 2
    gamma_n = (cos(lambda_n) - cosh(lambda_n) ) / (sin(lambda_n) + sinh(lambda_n) )
    Yn[:, 3] = gamma_n *  cos(lambda_n / b * eta) \
             - gamma_n * cosh(lambda_n / b * eta) \
             +            sin(lambda_n / b * eta) \
             -           sinh(lambda_n / b * eta)

    Yn[:, 4] = (eta**2 - b * eta)**2 * eta**(M.T.flatten() - 1)

    #
    for o in range(Situations_Ritz):

        #
        mn = 1

        #
        counter = 0

        #
        while error_Ritz[counter, o] >= margin and mn <= mn_max:

            #
            counter = (mn != 1) * (counter + 1)

            #
            index = (tile(linspace(0, mn - 1, mn), (mn, 1) ) + tile(linspace(0, mn_max * (mn - 1), mn), (mn, 1) ).T).flatten().astype(int)

            # Equation 5.44 from [1]
            B_mn =    D[0, 0] * (ddXkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]   ) \
                +     D[0, 1] * (  XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] \
                +                  XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T ) \
                +     D[1, 1] * (  XkXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]     * ddYlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] ) \
                + 4 * D[2, 2] * ( dXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    *  dYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]  )

            # Equation 5.44 from [1]
            #### Evaluationg of xm and yn at location xi and eta
            C_mn = Xm[index, BC_Ritz[o, 0] ] * Yn[index, BC_Ritz[o, 1] ]

            # Equation 5.44 from [1]
            A_mn = solve(B_mn, C_mn)
            if max( (B_mn @ A_mn - C_mn).flatten(), key=abs) > 1e-16:
                counter -= 1
                break

            # Equation 5.42 and 5.44 from [1]
            w = F_z * einsum('k,ijk->ij', A_mn, xm[:, :, index, BC_Ritz[o, 0] ] * yn[:, :, index, BC_Ritz[o, 1] ] )

            # Maximum deflection
            delta_Ritz[counter, o] = max(w.flatten(), key=abs)

            # Normalized, maximum norm []
            if counter == 0:
                error_Ritz[counter, o] = 100
            else:
                error_Ritz[counter, o] = real(sqrt( (w - W).flatten() @ (w - W).flatten() / (W.flatten() @ W.flatten() ) ) ) * 100

            #
            W = w.copy()

            # Because of symmetry considerations and based on the Fourier decomposition of the pressure load even numbers will
            # not contribute to the deformation
            mn += 2

        #
        mn_Ritz[o] = mn - 2

        #
        x_Ritz[o] = median(nodes_x[w == delta_Ritz[counter, o] ] )
        y_Ritz[o] = median(nodes_y[w == delta_Ritz[counter, o] ] )


    ## Lévy solution
    # Equation 5.26 from [1]
    alpha = D[0, 1] + 2 * D[2, 2]
    beta  = alpha**2 - D[0,0] * D[1,1]

    # Equation 5.26 from [1] (removed division by sqrt(D_11) for x and y-direction)
    lambda_1 = sqrt( (alpha - sqrt(beta) ) )
    lambda_2 = sqrt( (alpha + sqrt(beta) ) )

    # Page 94 from [1]
    if beta >= 0:
        lambda_1 = real(lambda_1)
        lambda_2 = real(lambda_2)
        if not isclose(lambda_1, lambda_2):
            lambda_1 = abs(lambda_1)
            lambda_2 = abs(lambda_2)
        else:
            lambda_1 = abs(lambda_1)
            lambda_2 = lambda_1
    else:
        lambda_2 = abs(imag(lambda_1) )
        lambda_1 = abs(real(lambda_1) )

    length = (BC_Lévy == 0) * a + (BC_Lévy == 1) * b
    width  = (BC_Lévy == 0) * b + (BC_Lévy == 1) * a

    # Equation 5.26 from [1]
    Lambda_1 = (BC_Lévy == 0) * lambda_1 / real(sqrt(D[0, 0] ) ) + (BC_Lévy == 1) * lambda_1 / real(sqrt(D[1, 1] ) )
    Lambda_2 = (BC_Lévy == 0) * lambda_2 / real(sqrt(D[0, 0] ) ) + (BC_Lévy == 1) * lambda_2 / real(sqrt(D[1, 1] ) )

    N = n[0, 0:mn_max]

    # Equation 5.5 and 5.16 from [1]
    q_mn = 4 * F_z / (a * b) * sin(M * pi * xi / a) * sin(M.T * pi * eta / b)

    #
    for o in range(Situations_Lévy):

        #
        mn = 1

        #
        counter = 0

        #
        W = ones(nodes_x.shape)

        #
        D_mn = (BC_Lévy[o] == 0) * (D[0, 0] * M**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * (M * M.T * a / b)**2 + D[1, 1] * (M * a / b)**4) \
             + (BC_Lévy[o] == 1) * (D[1, 1] * M**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * (M * M.T * b / a)**2 + D[0, 0] * (M * b / a)**4)

        #
        W_p = width[o]**4 / pi**4 * q_mn / D_mn

        #
        dw_p_0 = pi / width[o] * M * W_p

        #
        dw_p_a = dw_p_0 * (- 1)**M

        #
        dddw_p_0 = - pi**2 / width[o]**2 * M**2 * dw_p_0

        #
        dddw_p_a = - pi**2 / width[o]**2 * M**2 * dw_p_a

        #
        nodes_X = (BC_Lévy[o] == 0) * nodes_x + (BC_Lévy[o] == 1) * nodes_y
        nodes_Y = (BC_Lévy[o] == 0) * nodes_y + (BC_Lévy[o] == 1) * nodes_x

        #
        while error_Lévy[counter, o] >= margin and mn <= mn_max:

            #
            counter = (mn != 1) * (counter + 1)

            #
            if o == 0 or o == 1:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsSimplySupportedSimplySupported(N[0:mn] )
            elif o == 2 or o == 3:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsClampedSimplySupported(beta, Lambda_1[o], Lambda_2[o], length[o], width[o], sum(dw_p_0[0:mn, 0:mn], axis=1), N[0:mn] )
            elif o == 4 or o == 5:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsClampedClamped(beta, Lambda_1[o], Lambda_2[o], length[o], width[o], sum(dw_p_0[0:mn, 0:mn], axis=1), sum(dw_p_a[0:mn, 0:mn], axis=1), N[0:mn] )
            elif o == 6 or o == 7:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsFreeFree(beta, Lambda_1[o], Lambda_2[o], length[o], width[o], sum(dddw_p_0[0:mn, 0:mn], axis=1), sum(dddw_p_a[0:mn, 0:mn], axis=1), N[0:mn] )
            elif o == 8 or o == 9:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsFreeSimplySupported(beta, Lambda_1[o], Lambda_2[o], length[o], width[o], sum(dddw_p_0[0:mn, 0:mn], axis=1), N[0:mn] )
            elif o == 10 or o == 11:
                A_n, B_n, C_n, D_n \
                    = LévyCoefficientsClampedFree(beta, Lambda_1[o], Lambda_2[o], length[o], width[o], sum(dw_p_0[0:mn, 0:mn], axis=1), sum(dddw_p_a[0:mn, 0:mn], axis=1), N[0:mn] )

            if beta >= 0:
                # Equation 5.27 from [1]
                if not isclose(Lambda_1[o], Lambda_2[o] ):
                    phi_n = einsum('k,ijk->ijk', A_n[0:mn], cosh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) ) ) \
                          + einsum('k,ijk->ijk', B_n[0:mn], sinh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) ) ) \
                          + einsum('k,ijk->ijk', C_n[0:mn], cosh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) ) \
                          + einsum('k,ijk->ijk', D_n[0:mn], sinh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) )
                # Equation 5.29 from [1]
                else:
                    phi_n = einsum('k,ijk->ijk', A_n[0:mn]           , cosh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) ) ) \
                          + einsum('k,ij->ijk',  B_n[0:mn], nodes_X) * cosh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) )   \
                          + einsum('k,ijk->ijk', C_n[0:mn]           , sinh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) ) ) \
                          + einsum('k,ij->ijk',  D_n[0:mn], nodes_X) * sinh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) )
            # Equation 5.31 from [1]
            else:
                phi_n = (einsum('k,ijk->ijk', A_n[0:mn],  cos(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) )   \
                      +  einsum('k,ijk->ijk', B_n[0:mn],  sin(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) ) ) \
                      *                                  cosh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) )     \
                      + (einsum('k,ijk->ijk', C_n[0:mn],  cos(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) )   \
                      +  einsum('k,ijk->ijk', D_n[0:mn],  sin(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_2[o] / width[o], nodes_X) ) ) ) \
                      *                                  sinh(einsum('k,ij->ijk', N[0:mn] * pi * Lambda_1[o] / width[o], nodes_X) )

            # Equation 5.42 from [1]
            w =     einsum('ijk,ijk->ij', phi_n \
              +     einsum('lk,ijk->ijk', W_p[0:mn, 0:mn], \
                sin(einsum('k,ij->ijk',   N[0:mn] * pi / length[o], nodes_X) ) ), \
                sin(einsum('k,ij->ijk',   N[0:mn] * pi / width[o] , nodes_Y) ) )

            # Maximum deflection
            delta_Lévy[counter, o] = max(w.flatten(), key=abs)

            # Normalized, maximum norm
            if counter == 0:
                error_Lévy[counter, o] = 100
            else:
                error_Lévy[counter, o] = real(sqrt( (w - W).flatten() @ (w - W).flatten() / (W.flatten() @ W.flatten() ) ) ) * 100

            #
            W = w.copy()

            # Because of symmetry considerations and based on the Fourier decomposition of the pressure load even numbers will
            # not contribute to the deformation
            mn += 2

        #
        mn_Lévy[o] = mn - 2

        #
        x_Lévy[o] = median(nodes_x[w == delta_Lévy[counter, o] ] )
        y_Lévy[o] = median(nodes_y[w == delta_Lévy[counter, o] ] )


    # Tableau 30 (Tableau 10, Tableau 10 Medium & Tableau 10 Light)
    Colors = [ ( 31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
               (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 / 255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
               ( 44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 / 255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
               (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 / 255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
               (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 / 255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
               (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 / 255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
               (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 / 255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
               (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 / 255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
               (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 / 255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
               ( 23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.) ]

    Colors_Navier = [0]

    Colors_Ritz = [0, 3, 4, 6, 6, 7, 7, 9, 12, 12, 13, 13, 15, 15, 15, 15, 18, 19, 21, 22, 24, 24, 25, 25]

    Colors_Lévy = [0, 1, 3, 4, 6, 7, 18, 19, 27, 28, 21, 22]

    LineStyles = ['solid', 'dashed', 'dotted', 'dashdot']

    LineStyles_Navier = [0]

    LineStyles_Ritz = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 3, 1, 0, 0, 0, 0, 0, 1, 0, 1]

    LineStyles_Lévy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Initiate window
    plt.figure(0)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    plt.plot(linspace(1, mn_Navier, (mn_Navier + 1) // 2), error_Navier[0:(mn_Navier + 1) // 2], color=Colors[Colors_Navier[0] ], linewidth=1,
             linestyle=LineStyles[LineStyles_Navier[0] ], label=BoundaryConditions_Navier[0] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B5 [%]')
    plt.yscale('symlog', linthreshy = margin / 100)
    # Get min and max of y-axis
    y_b0, y_t0 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Initiate window
    plt.figure(1)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Ritz):
        plt.plot(linspace(1, mn_Ritz[o], (mn_Ritz[o] + 1) // 2), error_Ritz[0:(mn_Ritz[o] + 1) // 2, o], color=Colors[Colors_Ritz[o] ], linewidth=1,
                 linestyle=LineStyles[LineStyles_Ritz[o] ], label=BoundaryConditions_Ritz[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B5 [%]')
    plt.yscale('symlog', linthreshy = margin / 100)
    # Get min and max of y-axis
    y_b1, y_t1 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Initiate window
    plt.figure(2)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Lévy):
        plt.plot(linspace(1, mn_Lévy[o], (mn_Lévy[o] + 1) // 2), error_Lévy[0:(mn_Lévy[o] + 1) // 2, o],
                 color=Colors[Colors_Lévy[o] ], linewidth=1, linestyle=LineStyles[LineStyles_Lévy[o] ], label=BoundaryConditions_Lévy[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B5 [%]')
    plt.yscale('symlog', linthreshy = margin / 100)
    # Get min and max of y-axis
    y_b2, y_t2 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    #
    yb = min(y_b0, y_b1, y_b2)
    yt = max(y_t0, y_t1, y_t2)

    #
    plt.figure(0)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/ErrorConvergenceNavier.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(0)

    #
    plt.figure(1)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/ErrorConvergenceRitz.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(1)

    #
    plt.figure(2)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/ErrorConvergenceLévy.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(2)

    # Initiate window
    plt.figure(0)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    plt.plot(linspace(1, mn_Navier, (mn_Navier + 1) // 2), delta_Navier[0:(mn_Navier + 1) // 2], color=Colors[Colors_Navier[0] ], linewidth=1,
             linestyle=LineStyles[LineStyles_Navier[0] ], label=BoundaryConditions_Navier[0] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B4\u2098\u2090\u2093 [m]')
    plt.yscale('log')
    # Get min and max of y-axis
    y_b0, y_t0 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Initiate window
    plt.figure(1)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Ritz):
        plt.plot(linspace(1, mn_Ritz[o], (mn_Ritz[o] + 1) // 2), delta_Ritz[0:(mn_Ritz[o] + 1) // 2, o], color=Colors[Colors_Ritz[o] ], linewidth=1,
                 linestyle=LineStyles[LineStyles_Ritz[o] ], label=BoundaryConditions_Ritz[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B4\u2098\u2090\u2093 [m]')
    plt.yscale('log')
    # Get min and max of y-axis
    y_b1, y_t1 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Initiate window
    plt.figure(2)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Lévy):
        plt.plot(linspace(1, mn_Lévy[o], (mn_Lévy[o] + 1) // 2), delta_Lévy[0:(mn_Lévy[o] + 1) // 2, o],
                 color=Colors[Colors_Lévy[o] ], linewidth=1, linestyle=LineStyles[LineStyles_Lévy[o] ], label=BoundaryConditions_Lévy[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B4\u2098\u2090\u2093 [m]')
    plt.yscale('log')
    # Get min and max of y-axis
    y_b3, y_t3 = plt.ylim()
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) )
    plt.xticks(arange(1, (max(mn_Navier, max(mn_Ritz), max(mn_Lévy) ) + 2), 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad_SpeciallyOrthotropic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    #
    yb = min(y_b0, y_b1, y_b2)
    yt = max(y_t0, y_t1, y_t2)

    #
    plt.figure(0)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/DisplacementConvergenceNavier.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(0)

    #
    plt.figure(1)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/DisplacementConvergenceRitz.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(1)

    #
    plt.figure(2)
    #
    plt.ylim(yb, yt)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/DisplacementConvergenceLévy.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(2)

        #
    if data:
        #
        if exists(f'{simulation}/Data/PointLoad/MaximumDeformation.txt'):
            remove(f'{simulation}/Data/PointLoad/MaximumDeformation.txt')

        #
        textfile = open(f'{simulation}/Data/PointLoad/MaximumDeformation.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Maximum deformation due to a constant pressure load according to different analyses for various boundary conditions') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: PointLoad_SpeciallyOrthotopic.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        if max(abs(B.flatten() ) ) != 0 and (D[0,2] != 0 or D[1,2] != 0):
            textfile.write(fill(f'Warning: The laminate under consideration violates the assumed symmetry (B = 0) and vanishing of the ' \
                                f'bending-twisting coupling terms (D\u2081\u2086 = D\u2082\u2086 = 0).') )
            textfile.write(f'\n\n')
        elif (D[0,2] != 0 or D[1,2] != 0):
            textfile.write(fill(f'Warning: The laminate under consideration violates the assumed vanishing of the bending-twisting coupling ' \
                                f'terms (D\u2081\u2086 = D\u2082\u2086 = 0).') )
            textfile.write(f'\n\n')
        elif max(abs(B.flatten() ) ) != 0:
            textfile.write(fill(f'Warning: The laminate under consideration violates the assumed symmetry (B = 0).') )
            textfile.write(f'\n\n')

        #
        textfile.write(f'Navier solution \n')

        #
        textfile.write(tabulate( [ [BoundaryConditions_Navier[0], delta_Navier[(mn_Navier - 1) // 2], x_Navier, y_Navier, mn_Navier, error_Navier[(mn_Navier - 1) // 2] ],
                    [None, None, None, None, None, None] ],
                    headers=(f'Boundary conditions', f'\u03B4\u2098\u2090\u2093 [m]', f'x [m]', f'y [m]', f'm = n [-]', f'\u03B5 [%]'),
                    stralign=('left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.2e', '.2e', '.0f', '.5e') ) )

        #
        textfile.write(f'\n')

        #
        textfile.write(f'Ritz method \n')

        #
        textfile.write(tabulate( array([BoundaryConditions_Ritz, delta_Ritz[(mn_Ritz - 1) // 2, arange(delta_Ritz.shape[1]) ], x_Ritz, y_Ritz, mn_Ritz, error_Ritz[(mn_Ritz - 1) // 2, arange(error_Ritz.shape[1]) ] ] ).T.tolist(),
                    headers=(f'Boundary conditions', f'\u03B4\u2098\u2090\u2093 [m]', f'x [m]', f'y [m]', f'm = n [-]', f'\u03B5 [%]'),
                    stralign=('left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.2e', '.2e', '.0f', '.5e') ) )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Lévy solution \n')

        #
        textfile.write(tabulate( array([BoundaryConditions_Lévy, delta_Lévy[(mn_Lévy - 1) // 2, arange(delta_Lévy.shape[1] ) ], x_Lévy, y_Lévy, mn_Lévy, error_Lévy[(mn_Lévy - 1) // 2, arange(error_Lévy.shape[1] ) ] ] ).T.tolist(),
                    headers=(f'Boundary conditions', f'\u03B4\u2098\u2090\u2093 [m]', f'x [m]', f'y [m]', f'm = n [-]', f'\u03B5 [%]'),
                    stralign=('left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.2e', '.2e', '.0f', '.5e') ) )

        #
        textfile.write(f'\n')

        #
        textfile.close()

    # End the function
    return