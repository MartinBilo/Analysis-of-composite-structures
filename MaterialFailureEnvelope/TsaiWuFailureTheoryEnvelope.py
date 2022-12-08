def TsaiWuFailureTheoryEnvelope(laminate, material, settings, tqad, data=False, illustrations=False):
    """[summary]

    Parameters
    ----------
    laminate : [type]
        [description]
    material : [type]
        [description]
    settings : [type]
        [description]
    tqad : [type]
        [description]
    data : bool, optional
        [description], by default False
    illustrations : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    # Import packages into library
    from   datetime          import datetime
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as     plt
    from   numpy             import amax, amin, array, concatenate, copy, einsum, errstate, inf, linspace, maximum, ones, pi, repeat, sin, \
                                    sort, sqrt, tan, where, zeros
    from   os                import mkdir, remove
    from   os.path           import exists, isdir
    from   tabulate          import tabulate
    from   textwrap          import fill

    ## Import the laminate properties
    # The layup of the laminate in degrees
    theta = laminate['theta']

    ##
    #
    X_c = abs(material['X_c'] )
    #
    X_t = material['X_t']
    #
    Y_c = abs(material['Y_c'] )
    #
    Y_t = material['Y_t']
    #
    S   = material['S']

    ##
    #
    data_points = settings['data_points']
    #
    fileformat  = settings['fileformat']
    #
    simulation  = settings['simulation']

    ##
    def SingleFailureLoad(d, tqad, X_c, X_t, Y_c, Y_t, S):

        # In MaterialFailureEnvelopes.py it is shown that
        #
        #  σ₁  = tqad[0, d, :] ⋅ N ,
        #
        #  σ₂  = tqad[1, d, :] ⋅ N , and                                                                                        (MFE.X)
        #
        #  τ₁₂ = tqad[2, d, :] ⋅ N .
        #
        # The Tsai-Wu Failure Theory (equation 4.9 from [1]) is defined as
        #                                                _____________________________
        #  σ₁² / (Xᶜ ⋅ Xᵗ) + σ₂² / (Yᶜ ⋅ Yᵗ) - σ₁ ⋅ σ₂ ⋅ \/ 1 / (Xᶜ ⋅ Xᵗ) ⋅ 1 / (Yᶜ ⋅ Yᵗ)  + (1 / Xᵗ - 1 / Xᶜ) ⋅ σ₁
        #  + (1 / Yᵗ - 1 / Yᶜ) ⋅ σ₂ + τ₁₂² / S² = 1 ,                                                                            (SFL.5)
        #
        # where the first term of this expression has been corrected from σ₂² to σ₁². Combining and rearranging the previous two
        # expression results in
        #
        #  a ⋅ N₁² + b ⋅ N₁ - 1 = 0  -->                    (SFL.6)
        #

        # Specified compressive singular failure load of the laminate according to the Tsai-Wu Failure Theory (equation SFL.2)
        a = tqad[0, d, :] * tqad[0, d, :] / (X_c * X_t) \
          + tqad[1, d, :] * tqad[1, d, :] / (Y_c * Y_t) \
          + tqad[2, d, :] * tqad[2, d, :] / (S * S) \
          - tqad[0, d, :] * tqad[1, d, :] * sqrt(1 / (X_c * X_t) * 1 / (Y_c * Y_t) )
        b = (1 / X_t - 1/ X_c) * tqad[0, d, :] \
          + (1 / Y_t - 1/ Y_c) * tqad[1, d, :]

        # Initialize the array containing the specified compressive and tensile singular material material failure load of the laminate
        N = zeros(2)
        # Negative
        N[0] = min( (- b - sqrt(b**2 + 4 * a) ) / (2 * a), key = abs)
        # Positive
        N[1] = min( (- b + sqrt(b**2 + 4 * a) ) / (2 * a), key = abs)

        #
        return(N)

    def FailureEnvelope(data_points, F2, d1, d2, tqad, X_c, X_t, Y_c, Y_t, S):

        # The plane defined by the two independent loads under consideration (N₁ and N₂), which contains the corresponding material
        # failure envelope, is depicted in figure 1 below. In addition, the angle θ between both material failure loads is specified.
        #
        #                        N₁
        #                      θ = 0, 2π
        #                         ^
        #                         |
        #                     \ θ |
        #                      \<-|
        #                       \ |
        #                        \|
        #   θ = π / 2 ------------+------------> θ = 3π / 2, N₂
        #                         |
        #                         |
        #                         |
        #                         |
        #                         |
        #                       θ = π
        #
        # Figure FE.1: Illustration of the N₁N₂-plane and the angle
        #          θ between two material failure loads.
        #
        # The following relationships between two failure loads and the angle θ can be deduced from this figure:
        #           ___________
        #  N₁ =   \/ N₁² + N₂²  ⋅ cos θ  -->  N₁² = (N₁² + N₂²) ⋅ cos² θ , and                  (FE.1)
        #
        #           ___________
        #  N₂ = - \/ N₁² + N₂²  ⋅ sin θ  -->  N₂² = (N₁² + N₂²) ⋅ sin² θ ,                      (FE.2)
        #
        # which can be combined to yield
        #
        #                _____________             ______________
        #               /   sin² θ                /  1 - cos² θ
        #  N₂ = ± N₁   / ------------- =  ± N₁   / -------------- = ± N₁ ⋅ r₂₁ .                (FE.3)
        #            \/   1 - sin² θ           \/      cos² θ
        #
        # For the N₁N₂-plane, this expression can be used to reduce the number of unkown variables in a failure theory from two to one. It can
        # furthermore be surmised that this expression does not hold for θ = π / 2 or θ = 3π / 2 since equation FE.1 and FE.2
        # reduce to
        #
        #  N₁ = 0 , and
        #
        #  N₂ = N₂ ,
        #
        # for these conditions thereby decoupling both material failure loads.

        # Based on the specified number of data points, an array containing the angle θ, in radians, where the failure envelope in the
        # N₁N₂-plane is evaluated
        theta = 2 * pi / 360 * linspace(0, 360 - 360 / data_points, num = data_points)

        # The corresponding ratio between both material failure loads as defined in expression FE.3
        with errstate(divide='ignore'):
            ratio_force = where( (1 - (sin(theta) )**2) != 0, sqrt( (sin(theta) )**2 / (1 - (sin(theta) )**2) ), 0)

        # In MaterialFailureEnvelopes.py it is shown that
        #
        #  σ₁  = tqad[0, d1, :] ⋅ N₁ + tqad[0, d2, :] ⋅ N₂ ,
        #
        #  σ₂  = tqad[1, d1, :] ⋅ N₁ + tqad[1, d2, :] ⋅ N₂ , and                                (MFE.X)
        #
        #  τ₁₂ = tqad[2, d1, :] ⋅ N₁ + tqad[2, d2, :] ⋅ N₂ .
        #
        # From these expression in combination with equation FE.3 it follows that
        #
        #  σ₁  = N₁ ⋅ (tqad[0, d1, :] ± r₂₁ ⋅ tqad[0, d2, :] ) ,
        #
        #  σ₂  = N₁ ⋅ (tqad[1, d1, :] ± r₂₁ ⋅ tqad[1, d2, :] ) , and                            (FE.4)
        #
        #  τ₁₂ = N₁ ⋅ (tqad[2, d1, :] ± r₂₁ ⋅ tqad[2, d2, :] ) .
        #
        # The Tsai-Wu Failure Theory (equation 4.9 from [1]) is defined as
        #                                                _____________________________
        #  σ₁² / (Xᶜ ⋅ Xᵗ) + σ₂² / (Yᶜ ⋅ Yᵗ) - σ₁ ⋅ σ₂ ⋅ \/ 1 / (Xᶜ ⋅ Xᵗ) ⋅ 1 / (Yᶜ ⋅ Yᵗ)  + (1 / Xᵗ - 1 / Xᶜ) ⋅ σ₁
        #  + (1 / Yᵗ - 1 / Yᶜ) ⋅ σ₂ + τ₁₂² / S² = 1 ,                                                                      (FE.5)
        #
        # where the first term of this expression has been corrected from σ₂² to σ₁². Combining and rearranging the previous two
        # expression results in
        #
        #  (a₀ ± r₂₁ ⋅ a₁ + r₂₁² ⋅ a₂) ⋅ N₁² + (b₀ ± r₂₁ ⋅ b₁) ⋅ N₁ - 1 = 0  -->  Aᶠ ⋅ N₁² + Bᶠ ⋅ N₁ - 1 = 0                  (FE.6)
        #

        # The(se) coefficients of the Tsai-Wu Failure Theory are given by
        a0 = tqad[0, d1, :] * tqad[0, d1, :] / (X_c * X_t) \
           + tqad[1, d1, :] * tqad[1, d1, :] / (Y_c * Y_t) \
           + tqad[2, d1, :] * tqad[2, d1, :] / (S * S) \
           - tqad[0, d1, :] * tqad[1, d1, :] * sqrt(1 / (X_c * X_t) * 1 / (Y_c * Y_t) )
        a1 = 2 * tqad[0, d1, :] * tqad[0, d2, :] / (X_c * X_t) \
           + 2 * tqad[1, d1, :] * tqad[1, d2, :] / (Y_c * Y_t) \
           -     tqad[0, d1, :] * tqad[1, d2, :] * sqrt(1 / (X_c * X_t) * 1 / (Y_c * Y_t) ) \
           -     tqad[1, d1, :] * tqad[0, d2, :] * sqrt(1 / (X_c * X_t) * 1 / (Y_c * Y_t) ) \
           + 2 * tqad[2, d1, :] * tqad[2, d2, :] / (S * S)
        a2 = tqad[0, d2, :] * tqad[0, d2, :] / (X_c * X_t) \
           + tqad[1, d2, :] * tqad[1, d2, :] / (Y_c * Y_t) \
           + tqad[2, d2, :] * tqad[2, d2, :] / (S * S) \
           - tqad[0, d2, :] * tqad[1, d2, :] * sqrt(1 / (X_c * X_t) * 1 / (Y_c * Y_t) )
        b0 = tqad[0, d1, :] * (1 / X_t - 1 / X_c) \
           + tqad[1, d1, :] * (1 / Y_t - 1 / Y_c)
        b1 = tqad[0, d2, :] * (1 / X_t - 1 / X_c) \
           + tqad[1, d2, :] * (1 / Y_t - 1 / Y_c)

        # The sign of load N₂, and therefore of the coefficients a₁ and b₁ in expression FE.6, can be determined with figure FE.1
        #
        #   0     <= θ <=  π / 2  -->  N₁ > 0  -->  N₂ < 0  -->  N₂ = - r₂₁ ⋅ N₁  -->  - a₁ & - b₁ ,
        #   π / 2 <= θ <=  π      -->  N₁ < 0  -->  N₂ < 0  -->  N₂ = + r₂₁ ⋅ N₁  -->  + a₁ & + b₁ ,
        #   π     <= θ <= 3π / 2  -->  N₁ < 0  -->  N₂ > 0  -->  N₂ = - r₂₁ ⋅ N₁  -->  - a₁ & - b₁ , and                   (FE.7)
        #  3π / 2 <= θ <= 2π      -->  N₁ > 0  -->  N₂ > 0  -->  N₂ = + r₂₁ ⋅ N₁  -->  + a₁ & + b₁ ,
        #
        # resulting in the following expression for the force coefficients (Aᶠ and Bᶠ)
        A_f = einsum('i,j->ij', a0, ones(data_points) ) \
            + ( (    0      <= theta) & (theta <       pi / 2) * (- 1) \
            +   (    pi / 2 <  theta) & (theta <       pi)     * (  1) \
            +   (    pi     <= theta) & (theta <   3 * pi / 2) * (- 1) \
            +   (3 * pi / 2 <  theta) & (theta <=  2 * pi)     * (  1) ) \
            * einsum('i,j->ij', a1, ratio_force) \
            + einsum('i,j->ij', a2, ratio_force**2)
        B_f = einsum('i,j->ij', b0, ones(data_points) ) \
            + ( (    0      <= theta) & (theta <       pi / 2) * (- 1) \
            +   (    pi / 2 <  theta) & (theta <       pi)     * (  1) \
            +   (    pi     <= theta) & (theta <   3 * pi / 2) * (- 1) \
            +   (3 * pi / 2 <  theta) & (theta <=  2 * pi)     * (  1) ) \
            * einsum('i,j->ij', b1, ratio_force)

        # Initialize the array containing in each row the combination of failure loads of the laminate for each angle theta
        MN = zeros( [data_points, 2] )

        # Based on equation FE.6, the first failure load for each ply interfaces is given by
        #                 _____________
        #        - Bᶠ ± \/ Bᶠ² + 4 ⋅ Aᶠ
        #  N₁ = ------------------------ .                                                                                  (FE.8)
        #                2 ⋅ Aᶠ
        #
        # The sign in this expression, and thus of the first failure load, is based on figure FE.1
        #
        #   0     <= θ <   π / 2  -->  N₁ > 0
        #            θ =   π / 2  -->  N₁ = 0
        #   π / 2 <  θ <  3π / 2  -->  N₁ < 0                                                                     (FE.9)
        #            θ =  3π / 2  -->  N₁ = 0
        #  3π / 2 <  θ <= 2π      -->  N₁ > 0
        #
        MN[:, 0] = ( (theta <  pi / 2) | (theta >  3 * pi / 2) ) * (amin( (- B_f + sqrt(B_f**2 + 4 * A_f) ) / (2 * A_f), axis=0) ) \
                 + ( (theta >  pi / 2) & (theta <  3 * pi / 2) ) * (amax( (- B_f - sqrt(B_f**2 + 4 * A_f) ) / (2 * A_f), axis=0) )

        # The second failure load is given by
        #
        #  N₂ = - N₁ ⋅ tan θ                                                                                       (FE.10)
        #
        # as derived from figure FE.1. Like equation FE.3, this expression does not hold for θ = π / 2 or θ = 3π / 2. For these angles
        # the failure loads as obtained with SingularFailureLoad are substituted.
        MN[:, 1] = ( (theta !=     pi / 2) \
                 &   (theta != 3 * pi / 2) ) * (- MN[:, 0] * tan(theta) ) \
                 +   (theta ==     pi / 2)   * (F2[0] ) \
                 +   (theta == 3 * pi / 2)   * (F2[1] )

        # Append the array containing the combination of failure loads for each angle theta with the first row of this array to obtain a
        # closed failure envelope graphically
        MN       = concatenate( (MN, MN[0, :].reshape(1, 2) ), axis=0)

        # End the function and return the array describing the material failure envelope
        return(MN)

    # Number of plies in the laminate
    number_of_plies = len(theta)

    #
    X_c = repeat(X_c * ones(number_of_plies), 2)
    #
    X_t = repeat(X_t * ones(number_of_plies), 2)
    #
    Y_c = repeat(Y_c * ones(number_of_plies), 2)
    #
    Y_t = repeat(Y_t * ones(number_of_plies), 2)
    #
    S   = repeat(S * ones(number_of_plies), 2)

    #
    forces = [f'Nx', f'Ny', f'Nxy', f'Mx', f'My', f'Mxy']

    #
    number_of_forces = len(forces)

    #
    TsaiWu = {}

    #
    for i in range(0, number_of_forces):
        #
        locals()[forces[i] ] = SingleFailureLoad(i, tqad, X_c, X_t, Y_c, Y_t, S)
        #
        TsaiWu[forces[i] ] = locals()[forces[i] ]

    #
    for i in range(0, number_of_forces - 1):
        #
        for j in range(i + 1, number_of_forces):
            #
            locals()[f'{forces[i]}{forces[j]}'] = FailureEnvelope(data_points, locals()[forces[j] ], i, j, tqad, X_c, X_t, Y_c, Y_t, S)
            #
            TsaiWu[forces[i] + forces[j] ] = locals()[f'{forces[i]}{forces[j]}']

    #
    if data:
        #
        if exists(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiWuFailureTheory.txt'):
            remove(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiWuFailureTheory.txt')

        #
        textfile = open(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiWuFailureTheory.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Singular failure loads according to the Tsai-Wu Failure Theory') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: TsaiWuFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Singular failure loads \n')

        #
        textfile.write(tabulate( array( [ [f'N\u2093  [N/m]', locals()[forces[0] ][0], locals()[forces[0] ][1] ], \
                                          [f'N\u1D67  [N/m]', locals()[forces[1] ][0], locals()[forces[1] ][1] ], \
                                          [f'N\u2093\u1D67 [N/m]', locals()[forces[2] ][0], locals()[forces[2] ][1] ], \
                                          [f'M\u2093  [N]', locals()[forces[3] ][0], locals()[forces[3] ][1] ], \
                                          [f'M\u1D67  [N]', locals()[forces[4] ][0], locals()[forces[4] ][1] ], \
                                          [f'M\u2093\u1D67 [N]', locals()[forces[5] ][0], locals()[forces[5] ][1] ] ] ).tolist(),
                    headers=(f'Force', f'Tsai-Wu', f'Failure Theory'),
                    colalign=('left', 'right', 'left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.5e') ) )

        #
        textfile.close()

    if illustrations:
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

        # Labels
        Labels = [f'N\u2093 [N/m]', f'N\u1D67 [N/m]', f'N\u2093\u1D67 [N/m]', f'M\u2093 [N]', f'M\u1D67 [N]', f'M\u2093\u1D67 [N]']

        #
        if not isdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiWuFailureTheory'):
            mkdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiWuFailureTheory')

        #
        for i in range(0, number_of_forces - 1):
            #
            for j in range(i + 1, number_of_forces):
                #
                if exists(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiWuFailureTheory/{forces[i]}{forces[j]}.{fileformat}'):
                    remove(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiWuFailureTheory/{forces[i]}{forces[j]}.{fileformat}')

                # Initiate window
                plt.figure(0)
                # Full screen
                plt.get_current_fig_manager().full_screen_toggle()
                # Create vertical and horizontal line through origin
                plt.axvline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                plt.axhline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                # Create plots
                plt.plot(locals()[f'{forces[i]}{forces[j]}'][:, 1], locals()[f'{forces[i]}{forces[j]}'][:, 0], color=Colors[9], linewidth=1)
                plt.plot(0, locals()[forces[i] ][0], color=Colors[9], marker='.')
                plt.plot(0, locals()[forces[i] ][1], color=Colors[9], marker='.')
                plt.plot(locals()[forces[j] ][0], 0, color=Colors[9], marker='.')
                plt.plot(locals()[forces[j] ][1], 0, color=Colors[9], marker='.')
                # Format axes
                plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0) )
                # Format xlabel
                plt.xlabel(Labels[j] )
                # Format ylabel
                plt.ylabel(Labels[i] )
                # Create text wit simulation name and source
                plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0, -0.15, f'Source: TsaiWuFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                # Save figure
                plt.savefig(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiWuFailureTheory/{forces[i]}{forces[j]}.{fileformat}', bbox_inches='tight')
                # Close figure
                plt.close(0)

    #
    return(TsaiWu)