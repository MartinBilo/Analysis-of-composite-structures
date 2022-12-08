def AssumedDeflection(geometry, mesh, settings):
    """[summary]

    Args:
        geometry ([type]): [description]
        mesh ([type]): [description]
        settings ([type]): [description]
    """

    # Import packages into library
    from numpy import linspace, repeat, tile, zeros, longdouble

    # Import functions into library
    from AssumedDeflection.BeamApproximationFreeFree                       import BeamApproximationFreeFree
    from AssumedDeflection.BeamApproximationSimplySupportedSimplySupported import BeamApproximationSimplySupportedSimplySupported
    from AssumedDeflection.BeamApproximationClampedSimplySupported         import BeamApproximationClampedSimplySupported
    from AssumedDeflection.BeamApproximationClampedClamped                 import BeamApproximationClampedClamped
    from AssumedDeflection.PolynomialApproximationClampedClamped           import PolynomialApproximationClampedClamped


    ##
    #
    a = geometry['a']
    #
    b = geometry['b']

    ##
    #
    number_of_elements_x = mesh['number_of_elements_x']
    #
    number_of_elements_y = mesh['number_of_elements_y']

    ##
    #
    mn_max = settings['mn_max']

    #
    nodes_x = tile(linspace(0, a, number_of_elements_x + 1), (number_of_elements_y + 1, 1) )
    #
    nodes_y = tile(linspace(0, b, number_of_elements_y + 1), (number_of_elements_x + 1, 1) ).T

    #
    m = tile(repeat(linspace(1, mn_max, mn_max), mn_max), (mn_max**2, 1) )
    #
    n = tile(tile(linspace(1, mn_max, mn_max), mn_max), (mn_max**2, 1) )

    #
    number_of_approximations = 5

    #
    xm       = zeros( [nodes_x.shape[0], nodes_x.shape[1], m.shape[0], number_of_approximations], dtype=longdouble )
    Xm       = zeros( [m.shape[0], number_of_approximations] )
    XkXm     = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    XkXm_m   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    XkddXm   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    dXkdXm   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    ddXkddXm = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    XkdXm    = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    ddXkdXm  = zeros( [m.shape[0], m.shape[1], number_of_approximations] )

    #
    yn       = zeros( [nodes_y.shape[0], nodes_y.shape[1], m.shape[0], number_of_approximations] )
    Yn       = zeros( [m.shape[0], number_of_approximations] )
    YlYn     = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    YlYn_m   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    YlddYn   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    dYldYn   = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    ddYlddYn = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    YldYn    = zeros( [m.shape[0], m.shape[1], number_of_approximations] )
    ddYldYn  = zeros( [m.shape[0], m.shape[1], number_of_approximations] )

    #
    xm[:,:,:,0], Xm[:,0], XkXm[:,:,0], XkXm_m[:,:,0], XkddXm[:,:,0], dXkdXm[:,:,0], ddXkddXm[:,:,0], XkdXm[:,:,0], ddXkdXm[:,:,0] \
        = BeamApproximationFreeFree(a, m, nodes_x)

    #
    yn[:,:,:,0], Yn[:,0], YlYn[:,:,0], YlYn_m[:,:,0], YlddYn[:,:,0], dYldYn[:,:,0], ddYlddYn[:,:,0], YldYn[:,:,0], ddYldYn[:,:,0] \
        = BeamApproximationFreeFree(b, n, nodes_y)

    #
    xm[:,:,:,1], Xm[:,1], XkXm[:,:,1], XkXm_m[:,:,1], XkddXm[:,:,1], dXkdXm[:,:,1], ddXkddXm[:,:,1], XkdXm[:,:,1], ddXkdXm[:,:,1] \
        = BeamApproximationSimplySupportedSimplySupported(a, m, nodes_x)

    #
    yn[:,:,:,1], Yn[:,1], YlYn[:,:,1], YlYn_m[:,:,1], YlddYn[:,:,1], dYldYn[:,:,1], ddYlddYn[:,:,1], YldYn[:,:,1], ddYldYn[:,:,1] \
        = BeamApproximationSimplySupportedSimplySupported(b, n, nodes_y)

    #
    xm[:,:,:,2], Xm[:,2], XkXm[:,:,2], XkXm_m[:,:,2], XkddXm[:,:,2], dXkdXm[:,:,2], ddXkddXm[:,:,2], XkdXm[:,:,2], ddXkdXm[:,:,2] \
        = BeamApproximationClampedSimplySupported(a, m, nodes_x)

    #
    yn[:,:,:,2], Yn[:,2], YlYn[:,:,2], YlYn_m[:,:,2], YlddYn[:,:,2], dYldYn[:,:,2], ddYlddYn[:,:,2], YldYn[:,:,2], ddYldYn[:,:,2] \
        = BeamApproximationClampedSimplySupported(b, n, nodes_y)

    #
    xm[:,:,:,3], Xm[:,3], XkXm[:,:,3], XkXm_m[:,:,3], XkddXm[:,:,3], dXkdXm[:,:,3], ddXkddXm[:,:,3], XkdXm[:,:,3], ddXkdXm[:,:,3] \
        = BeamApproximationClampedClamped(a, m, nodes_x)

    #
    yn[:,:,:,3], Yn[:,3], YlYn[:,:,3], YlYn_m[:,:,3], YlddYn[:,:,3], dYldYn[:,:,3], ddYlddYn[:,:,3], YldYn[:,:,3], ddYldYn[:,:,3] \
        = BeamApproximationClampedClamped(b, n, nodes_y)

    #
    xm[:,:,:,4], Xm[:,4], XkXm[:,:,4], XkXm_m[:,:,4], XkddXm[:,:,4], dXkdXm[:,:,4], ddXkddXm[:,:,4], XkdXm[:,:,4], ddXkdXm[:,:,4] \
        = PolynomialApproximationClampedClamped(a, m, nodes_x)

    #
    yn[:,:,:,4], Yn[:,4], YlYn[:,:,4], YlYn_m[:,:,4], YlddYn[:,:,4], dYldYn[:,:,4], ddYlddYn[:,:,4], YldYn[:,:,4], ddYldYn[:,:,4] \
        = PolynomialApproximationClampedClamped(b, n, nodes_y)

    ##
    assumeddeflection             = {}
    assumeddeflection['xm']       = xm
    assumeddeflection['yn']       = yn
    assumeddeflection['Xm']       = Xm
    assumeddeflection['Yn']       = Yn
    assumeddeflection['XkXm']     = XkXm
    assumeddeflection['YlYn']     = YlYn
    assumeddeflection['XkXm_m']   = XkXm_m
    assumeddeflection['YlYn_m']   = YlYn_m
    assumeddeflection['XkddXm']   = XkddXm
    assumeddeflection['YlddYn']   = YlddYn
    assumeddeflection['dXkdXm']   = dXkdXm
    assumeddeflection['dYldYn']   = dYldYn
    assumeddeflection['ddXkddXm'] = ddXkddXm
    assumeddeflection['ddYlddYn'] = ddYlddYn
    assumeddeflection['XkdXm']    = XkdXm
    assumeddeflection['YldYn']    = YldYn
    assumeddeflection['ddXkdXm']  = ddXkdXm
    assumeddeflection['ddYldYn']  = ddYldYn

    ##
    mesh['nodes_x'] = nodes_x
    mesh['nodes_y'] = nodes_y

    ##
    settings['m'] = m
    settings['n'] = n

    return(assumeddeflection, mesh, settings)