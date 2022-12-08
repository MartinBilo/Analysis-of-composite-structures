# Import modules
from datetime import datetime
from numpy    import array, allclose, ceil, isclose, ones, vstack, zeros
from pandas   import DataFrame
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List, Tuple


# The funtion 'laminate_classification'
def laminate_classification(laminate : Dict[str, float or List[float] ], material : Dict[str, List[str] ], settings : Dict[str, str],
                            stiffness : Dict[str, List[float] ], data : bool = False) -> Tuple[Dict[str, str], Dict[str, List[float] ] ]:
    """Computes the local stress at the top and bottom of each ply of a laminate.

    Returns the ``stress`` dictionary appended with a shape-(2N, 3) array containing the three local stresses at the top and bottom of each ply
    based on the global stress (``sigma_12``) for a laminate characterized by the lay-up (``theta``) and the z-coordinate of each ply (``z``).
    This computation is denoted by a moniker (``simulation``). Two optional booleans (``data`` and ``illustrations``) can be defined which
    respectively indicate if a text file (``local_strain.txt``) containing the local stresses or if an illustration (``local_strain``) showing
    the variation of these stresses throughout the laminate should be generated. This illustration uses the stipulated format (``fileformat``)
    and resolution (``resolution``).

    Parameters
    ----------
    - ``laminate`` : Dict | Dictionary containing:
        - ``theta`` : numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - ``z`` : numpy.ndarray, shape = (N + 1) | The z-coordinate of each of the N + 1 ply interfaces.
    - ``settings`` : Dict | Dictionary containing:
        - ``fileformat`` : str | The format of the files as which the generated illustrations are saved.
        - ``resolution`` : float | The resolution of the generated illustrations in dots-per-inch.
        - ``simulation`` : str | The moniker of the simulation.
    - ``stress`` : Dict | Dictionary containing:
        - ``sigma_12`` : numpy.ndarray, shape = (2N, 3) | The stresses at the top and bottom of each ply in the global 1-2 coordinate system.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the three local stresses at the top and bottom of each ply should
    be generated, by default ``False``.
    - ``illustrations`` : bool, optional | Boolean indicating if an illustration containing the variation of the three local stresses throughout
    the laminate should be generated, by default ``False``.

    Returns
    -------
    - ``stress`` : Dict | Dictionary appended with:
        - ``sigma_xy`` : numpy.ndarray, shape = (2N, 3) | The three stresses at the top and bottom of each ply in the local x-y coordinate
        system from the top to the bottom of the laminate.

    Output
    ------
    - ``local_stress.txt`` : text file | A text file containing the three local stresses at the top and bottom of each ply from the top to the
    bottom of the laminate.
    - ``local_stress`` : illustration | An illustration showing the variation of the three local stresses throughout the laminate.

    Assumptions
    -----------

    Version
    -------
    - v1.0 :
        - Initial version (06/08/2020) | M. Bilo

    References
    ----------
    [1]: Python Software Foundation. August 6 2020. accessed 6 August 2020. <https://docs.python.org/3/library/functions.html#chr>

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [3]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Section 5.2
    """

    # The lay-up of the laminate in degrees
    theta = laminate['theta']
    # The thickness of a/each ply
    t = laminate['t']

    # The classification of the material(s) comprising the laminate
    composite = material['composite']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the extensional stiffness (A-)matrix
    A = stiffness['A']
    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix
    B = stiffness['B']
    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # An array of shape-(N) containing the thickness of each of the N plies
    t = t * ones(number_of_plies)

    # If the number of plies is unequal to the indicated number of materials in the laminate
    if number_of_plies != len(composite):
        # Expand the list containing the classification of the material(s) comprising the laminate to all plies
        composite = array( [composite for i in range(number_of_plies) ] ).T

    # To simplify the classification, wrap the lay-up of the laminate to the interval -90° < θ < ∞° by shifting the direction of relevant plies
    # with integer multiples of 180°
    theta[theta <= - 90] = theta[theta <= - 90] + ceil(theta[theta <= - 90] / - 180) * 180
    # Subsequently, wrap the lay-up to the interval -90° < θ <= 90° by shifting the direction of relevant plies with integer multiples of -180°
    theta[theta >    90] = theta[theta >    90] - ceil(theta[theta >    90] /   180) * 180

    # 6 separate classes are defined, namely
    #
    # 1) category_1; indicates if the laminate consist of a single ply,
    # 2) category_2; indicates if the laminate is symmetric/antisymmetric/asymmetric,
    # 3) category_3; indicates if the laminate is an cross- or angle ply,
    # 4) category_4; indicates if the laminate is balanced,
    # 5) category_5; indicates the relative coupling between the normal and shear forces, and
    # 6) category_6; indicates the relative coupling between the bending and twisting moments,
    #
    # which combined at the end of this function yield a complete classification of the laminate.

    # Initiate an alpabetical counter, based on the corresponding unicode integer, to sort the various relevant categories successively in the
    # classification of the laminate [1]
    counter = 97

    # A dataframe of shape-(N, 3) containing the lay-up of the laminate, the thickness of each ply, and the material of each ply
    laminate_properties = DataFrame(vstack( (theta, t, composite) ).T, columns = ['theta', 't', 'material'] )

    # Cast the columns of the dataframe containing the lay-up of the laminate and the thickness of each ply to float
    laminate_properties[ ['theta', 't'] ] = laminate_properties[ ['theta', 't'] ].astype(float)

    # If the laminate consists of a single ply
    if number_of_plies == 1:
        # Substitute the appropriate classifications in categories 1 to 5
        category_1 = f'({chr(counter)}) single ply'
        category_2 = f''
        category_3 = f''
        category_4 = f''

        # Adjust the coupling stiffness (B-)matrix to remove possible, numerical round-off errors
        B = zeros( [3, 3] )

    # Otherwise, if the laminate consists of multiple plies
    else:
        # Substitute the appropriate classification in category 1
        category_1 = f''

        # If the laminate is symmetric (section 5.2.1 from [2])
        if laminate_properties[ ['theta', 't'] ].equals(laminate_properties[ ['theta', 't'] ][::-1].reset_index(drop=True) ) and \
           laminate_properties['material'].equals(laminate_properties['material'][::-1].reset_index(drop=True) ):
            # Substitute the appropriate classification in category 2
            category_2 = f'({chr(counter)}) symmetric'

            # Adjust the coupling stiffness (B-)matrix to remove possible, numerical round-off errors
            B = zeros( [3, 3] )

            # Increase the alpabetical counter by 1
            counter += 1

        # If the laminate is antisymmetric (section 5.2.4 from [2])
        elif laminate_properties['theta'].equals(- laminate_properties['theta'][::-1].reset_index(drop=True) ) and \
             laminate_properties['t'].equals(laminate_properties['t'][::-1].reset_index(drop=True) ) and \
             laminate_properties['material'].equals(laminate_properties['material'][::-1].reset_index(drop=True) ):
            # Substitute the appropriate classification in category 2
            category_2 = f'({chr(counter)}) antisymmetric'

            # Adjust the components of the extensional stiffness (A-)matrix to remove possible, numerical round-off errors
            # Element A₁₆
            A[0, 2] = 0
            # Element A₆₁
            A[2, 0] = 0
            # Element A₂₆
            A[1, 2] = 0
            # Element A₆₂
            A[2, 1] = 0

            # Adjust the components of the bending stiffness (D-)matrix to remove possible, numerical round-off errors
            # Element D₁₆
            D[0, 2] = 0
            # Element D₆₁
            D[2, 0] = 0
            # Element D₂₆
            D[1, 2] = 0
            # Element D₆₂
            D[2, 1] = 0

            # Increase the alpabetical counter by 1
            counter += 1

        # Otherwise, if the laminate is asymmetric
        else:
            # Substitute the appropriate classification in category 2
            category_2 = f'({chr(counter)}) asymmetric'

            # Increase the alpabetical counter by 1
            counter += 1

        # Determine the unique ply directions in the laminate sorted in an ascending manner as a function of the ply direction
        laminate_properties_unique_layup = laminate_properties.drop_duplicates(subset = ['theta'] ).sort_values(by = ['theta'], \
            ignore_index = True )

        # Determine the number of unique ply directions in the laminate
        number_of_plies_unique_layup = len(laminate_properties_unique_layup.index)

        # Determine the unique plies in the laminate sorted in an ascending manner as a function of the ply direction
        laminate_properties_unique = laminate_properties.drop_duplicates().sort_values(by = ['theta'], \
            ignore_index = True )

        # Determine the number of unique plies in the laminate
        number_of_plies_unique = len(laminate_properties_unique.index)

        # If the laminate is a cross-ply (section 5.2.2 from [2])
        if number_of_plies_unique_layup == 2 and isclose(laminate_properties_unique_layup['theta'][0], 0) and \
           isclose(laminate_properties_unique_layup['theta'][1], 90):
            # Substitute the appropriate classification in category 3
            category_3 = f', ({chr(counter)}) cross-ply'

            # Adjust the components of the extensional stiffness (A-)matrix to remove possible, numerical round-off errors
            # Element A₁₆
            A[0, 2] = 0
            # Element A₆₁
            A[2, 0] = 0
            # Element A₂₆
            A[1, 2] = 0
            # Element A₆₂
            A[2, 1] = 0

            # Adjust the components of the coupling stiffness (B-)matrix to remove possible, numerical round-off errors
            # Element B₁₆
            B[0, 2] = 0
            # Element B₆₁
            B[2, 0] = 0
            # Element B₂₆
            B[1, 2] = 0
            # Element B₆₂
            B[2, 1] = 0

            # Adjust the components of the bending stiffness (D-)matrix to remove possible, numerical round-off errors
            # Element D₁₆
            D[0, 2] = 0
            # Element D₆₁
            D[2, 0] = 0
            # Element D₂₆
            D[1, 2] = 0
            # Element D₆₂
            D[2, 1] = 0

            # Increase the alpabetical counter by 1
            counter += 1

        # If the laminate is an angle ply (section 5.2.3 from [2])
        elif number_of_plies_unique == 2 and isclose(laminate_properties_unique['theta'][0], - laminate_properties_unique['theta'][1] ) and \
             isclose(laminate_properties_unique['t'][0], laminate_properties_unique['t'][1] ) and \
             laminate_properties_unique['material'][0] == laminate_properties_unique['material'][1]:
            # Substitute the appropriate classification in category 3
            category_3 = f', ({chr(counter)}) angle ply'

            # Determine the number of times each ply direction of the angle ply occurs in the laminate
            count_plies_unique = laminate_properties.value_counts().tolist()

            # If the laminate has an even number of plies (section 5.2.3 from [2])
            if (number_of_plies_unique % 2) == 0 and isclose(count_plies_unique[0], count_plies_unique[1] ):
                # Adjust the components of the extensional stiffness (A-)matrix to remove possible, numerical round-off errors
                # Element A₁₆
                A[0, 2] = 0
                # Element A₆₁
                A[2, 0] = 0
                # Element A₂₆
                A[1, 2] = 0
                # Element A₆₂
                A[2, 1] = 0

            # Increase the alpabetical counter by 1
            counter += 1

        # Otherwise, if the laminate is neither a cross-ply nor an angle ply
        else:
            # Substitute the relevant classification in the appropriate category
            category_3 = f''

        # Determine unique ply directions other than 0° and 90° in the laminate sorted in an ascending manner as a function of the ply
        # direction
        laminate_properties_unique_balanced = laminate_properties[(laminate_properties.theta != 0) & \
            (laminate_properties.theta != 90) ].drop_duplicates().sort_values(by = ['theta'], ignore_index = True ).sort_values(by = ['theta'], \
            ignore_index = True )

        # Determine the number of unique ply directions other than 0° and 90°
        number_of_plies_unique_balanced = len(laminate_properties_unique_balanced.index)

        # Determine the number of times each ply direction other than 0° and 90° occurs in the laminate
        count_plies_unique_balanced = laminate_properties[(laminate_properties.theta != 0) & \
            (laminate_properties.theta != 90) ].value_counts(sort=False).tolist()

        # If the laminate is balanced (section 5.2.5 from [2])
        if number_of_plies_unique_balanced != 0 and allclose(count_plies_unique_balanced, count_plies_unique_balanced[::-1] ) and \
           laminate_properties_unique_balanced['theta'].equals(- laminate_properties_unique_balanced['theta'][::-1].reset_index(drop=True) ) and \
           laminate_properties_unique_balanced['t'].equals(laminate_properties_unique_balanced['t'][::-1].reset_index(drop=True) ) and \
           laminate_properties_unique_balanced['material'].equals(laminate_properties_unique_balanced['material'][::-1].reset_index(drop=True) ):
            # Substitute the appropriate classification in category 4
            category_4 = f', ({chr(counter)}) balanced'

            # Adjust the components of the extensional stiffness (A-)matrix to remove possible, numerical round-off errors
            # Element A₁₆
            A[0, 2] = 0
            # Element A₆₁
            A[2, 0] = 0
            # Element A₂₆
            A[1, 2] = 0
            # Element A₆₂
            A[2, 1] = 0

            # Increase the alpabetical counter by 1
            counter += 1

        # Otherwise, if the laminate is not balanced
        else:
            # Substitute the appropriate classification in category 4
            category_4 = f''

    # The relative coupling between the normal and shear forces
    force_coupling = max(A[0,2], A[1,2], key=abs) / min(A[0,0], A[1,1], A[0,1], key=abs) * 100

    # The relative coupling between the bending and twisting moments
    moment_coupling = max(D[0,2], D[1,2], key=abs) / min(D[0,0], D[1,1], D[0,1], key=abs) * 100

    # If the normal and shear forces, and the bending and twisting moments are coupled for the laminate
    if force_coupling != 0 and moment_coupling != 0:
        # Substitute the appropriate classification in category 5
        category_5 = f', with a relative coupling between the normal and shear forces of {force_coupling:.3e}%'
        # Substitute the appropriate classification in category 6
        category_6 = f', and with a relative coupling between the bending and twisting moments of {moment_coupling:.3e}%'
    # If the normal and shear forces are coupled for the laminate
    elif force_coupling != 0:
        # Substitute the appropriate classification in category 5
        category_5 = f', with a relative coupling between the normal and shear forces of {force_coupling:.3e}%'
        # Substitute the appropriate classification in category 6
        category_6 = f''
    # If the bending and twisting moments are coupled for the laminate
    elif moment_coupling != 0:
        # Substitute the appropriate classification in category 5
        category_5 = f''
        # Substitute the appropriate classification in category 6
        category_6 = f', with a relative coupling between the bending and twisting moments of {moment_coupling:.3e}%'
    # Otherwise, if the normal and shear forces, and the bending and twisting moments are uncoupled for the laminate
    else:
        # Substitute the appropriate classification in category 5
        category_5 = f''
        # Substitute the appropriate classification in category 6
        category_6 = f''

    # Construct the classification of the laminate based on the various evaluated categories
    classification = f'The laminate is {category_1}{category_2}{category_3}{category_4}{category_5}{category_6}.'

    # If text files containing the classification of the laminate and the components of the adjusted ABD-matrix of the laminate should be
    # generated
    if data:
        # Create and open the text file
        textfile = open(f'{simulation}/data/classical_laminated_plate_theory/laminate_classification.txt', 'w', encoding = 'utf8')

        # Add the classification of the laminate
        textfile.write(fill(f'{classification}') )

        # Add the source and a timestamp
        textfile.write(f'\n\n')
        textfile.write(fill(f'Source: laminate_classification.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        # Close the text file
        textfile.close()

        # Create and open the text file
        textfile = open(f'{simulation}/data/classical_laminated_plate_theory/ABD_matrix_adjusted.txt', 'w', encoding = 'utf8')

        # Add a description of the contents
        textfile.write(fill(f'The components of the ABD-matrix, adjusted based on the classification of the laminate') )

        # Add the source and a timestamp
        textfile.write(f'\n\n')
        textfile.write(fill(f'Source: laminate_classification.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        # Add the adjusted extensional stiffness (A-)matrix
        textfile.write(f'\n\n')
        textfile.write(f'A [N/m] =')
        textfile.write(f'\n')
        textfile.write(tabulate(A, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

        # Add the adjusted coupling stiffness (B-)matrix
        textfile.write(f'\n\n')
        textfile.write(f'B [N] =')
        textfile.write(f'\n')
        textfile.write(tabulate(B, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

        # Add the adjusted bending stiffness (D-)matrix
        textfile.write(f'\n\n')
        textfile.write(f'D [Nm] =')
        textfile.write(f'\n')
        textfile.write(tabulate(D, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

        # Close the text file
        textfile.close()

    # Append the laminate dictionary with the classification of the laminate
    laminate['classification'] = classification

    # Append the stiffness dictionary with the adjusted extensional stiffness (A-)matrix
    stiffness['A'] = A
    # Append the stiffness dictionary with the adjusted coupling stiffness (B-)matrix
    stiffness['B'] = B
    # Append the stiffness dictionary with the adjusted bending stiffness (D-)matrix
    stiffness['D'] = D

    # End the function and return the appended laminate and stiffness dictionary
    return(laminate, stiffness)
