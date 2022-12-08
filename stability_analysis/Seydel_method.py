# Import modules
from datetime import datetime
from numpy    import argmax, array, finfo, sqrt
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List, Tuple


# The function 'Seydel_method'
def Seydel_method(geometry : Dict[str, float], settings : Dict[str, float or str],
                  stiffness : Dict[str, List[float] ], data : bool = False ) -> Tuple[List[float], List[float] ]:
    """[summary]

    Parameters
    ----------
    geometry : Dict[str, float]
        [description]
    settings : Dict[str, float or str]
        [description]
    stiffness : Dict[str, List[float] ]
        [description]
    data : bool, optional
        [description], by default False

    Returns
    -------
    Tuple[List[float], List[float] ]
        [description]
    """
    # Infinite strip (Seydel (section 5.9 from [1] ) )

    # The plate dimension (width) in the x-direction
    a = geometry['a']
    # The plate dimension (length) in the y-direction
    b = geometry['b']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # The quantity T dictating the governing equation, and the coefficients β₁ and β₂ for the laminate under consideration (equation 5.126 from
    # [1])
    T = sqrt(D[0, 0] * D[1, 1] ) / (D[0, 1] + 2 * D[2, 2] )

    # An array/table of shape-(5, 11) containing the coefficients β₁ and β₂ for simply supported and clamped sides respectively for the entire
    # range of the parameter T (taken from tables 5.6 and 5.7 from [1]). The rows of the table correspond to: 0) the quantity T, 1) β₁ for
    # simply supported sides, 2) β₁ for clamped sides, 3) β₂ for simply supported sides, and 4) β₂ for clamped sides. For values of the
    # parameter T for which a value of the coefficient β₂ is not given, the value of this coefficient is taken from the nearest value of the
    # quantity T for which it is available. The value of infinity for the parameter T in tables 5.6 and 5.7 is replaced by the largest
    # representable float to guarantee linear interpolation for the entire range of the quantity T.
    beta = array( [ [0,       0.2,  0.5,     1,     2,     3,    5,   10,     20,    40, finfo(float).max], \
                    [11.71,  11.8, 12.2, 13.17,  10.8,  9.95, 9.25,  8.7,    8.4,  8.25,            8.125], \
                    [18.59, 18.85, 19.9, 22.15, 18.75, 17.55, 16.6, 15.85, 15.45, 15.25,            15.07], \
                    [ 1.94,  1.94, 2.07,  2.49,  2.28,  2.16, 2.13,  2.08,  2.08,  2.08,             2.08], \
                    [ 1.20,  1.20, 1.36,  1.66,  1.54,  1.48, 1.44,  1.41,  1.41,  1.41,             1.41] ] )

    # An index indicating the lower boundary of the bin encompassing the value of the quantity T for the layup under consideration in the
    # previous table
    index = argmax(beta[0, beta[0, :] <= T] )

    # An array of shape-(2) containing the coefficient β₁ for simply supported and clamped sides respectively based on linear interpolation of
    # the values at the boundaries of the bin encompassing the value of the quantity T for the layup under consideration in the previous table
    beta_1 = ( (beta[0, index + 1] - T) * beta[1:3, index] + (T - beta[0, index] ) * beta[1:3, index + 1] ) \
           / (beta[0, index + 1] - beta[0, index] )

    # An array of shape-(2) containing the coefficient β₂ for simply supported and clamped sides respectively based on linear interpolation of
    # the values at the boundaries of the bin encompassing the value of the quantity T for the layup under consideration in the previous table
    beta_2 = ( (beta[0, index + 1] - T) * beta[3:, index] + (T - beta[0, index] ) * beta[3:, index + 1] ) \
           / (beta[0, index + 1] - beta[0, index] )

    # Select the relevant equations for the in-plane, shear, buckling load and the length between the successive buckling waves in the plate
    # for simply supported and clamped sides based on quantity T
    if 1 <= T:
        # An array of shape-(2) containing the in-plane, shear, buckling load for simply supported and clamped sides respectively (equation
        # 5.127 from [1])
        N_Seydel = 4 * beta_1 / ( (a >= b) * b**2 + (b > a) * a**2) \
                 * ( ( (a >= b) * D[0, 0] * D[1, 1]**3 + (b > a) * D[0, 0]**3 * D[1, 1] ) )**(1 / 4)
        # An array of shape-(2) containing the length between the successive buckling waves in the plate (expression 5.128 from [1])
        xi   = beta_2 * ( (a >= b) * b + (b > a) * a) / 2 * ( ( (a >= b) * D[0, 0] + (b > a) * D[1, 1] ) \
             / ( (a >= b) * D[1, 1] + (b > a) * D[0, 0] ) )**(1 / 4)
    else:
        # An array of shape-(2) containing the in-plane, shear, buckling load for simply supported and clamped sides respectively (equation
        # 5.129 from [1])
        N_Seydel = 4 * beta_1 / ( (a >= b) * b + (b > a) * a) \
                 * sqrt( ( (a >= b) * D[1, 1] + (b > a) * D[0, 0] ) * (D[0, 1] + 2 * D[2, 2] )  )
        # An array of shape-(2) containing the length between the successive buckling waves in the plate (expression 5.130 from [1])
        xi   = beta_2 * ( (a >= b) * b + (b > a) * a) / 2 * sqrt( (D[0, 1] + 2 * D[2, 2] ) / ( (a >= b) * D[1, 1] + (b > a) * D[0, 0] ) )

    # If a text file containing the in-plane, shear, buckling load and the length between the successive buckling waves in the plate should be
    # generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/stability_analysis/Seydel_method.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The in-plane, shear, buckling load of an infinite, rectangular, specially orthotropic strip according to '
                                f'the Seydel method') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: stability_analysis/Seydel_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the in-plane, shear, buckling load and the length between the successive buckling waves in the plate for simply supported
            # and clamped sides
            textfile.write(tabulate(array( [ [f'Simply supported', f'\u00B1{N_Seydel[0]:.5e}', xi[0] ],
                                             [f'Clamped',          f'\u00B1{N_Seydel[1]:.5e}', xi[1] ] ] ).tolist(),
                        headers = [f'Boundary conditions', f'N\u1D6A\u1D67 [N/m]', f'Half-wave length [m]'],
                        colalign = ('left', 'center', 'center'), floatfmt = ('.5e', '.5e', '.2f') ) )

    # End the function and return the arrays of shape-(2) containing the in-plane, shear, buckling load and the length between the successive
    # buckling waves in the plate for simply supported and clamped sides
    return(N_Seydel, xi)
