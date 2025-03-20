# Import modules
from datetime  import datetime
from jax.numpy import allclose
from tabulate  import tabulate
from textwrap  import fill

# The function 'laminate_stiffness'
def laminate_stiffness(laminate : dict[float], settings : dict[str], stiffness : dict[list[float], list[float], list[float] ],
                       data : bool = False) -> dict[list[float] ]:
    """Computes the in-plane and flexural engineering constants for a laminate.

    Returns the ``laminate`` dictionary appended with the in-plane (``E_1m``, ``E_2m``, ``G_12m``, ``nu_12m``, and ``nu_21m``) and flexural
    (``E_1b``, ``E_2b``, ``G_12b``, ``nu_12b``, and ``nu_21b``) engineering constants for a laminate characterized by: the height of the
    laminate (``h``), the coupling stiffness (``B``-)matrix, the extensional compliance matrix (``alpha``), and the bending compliance matrix
    (``delta``). The computation is denoted by a moniker (``simulation``). Lastly, an optional boolean (``data``) can be defined which
    indicates if a text file (``laminate_stiffness.txt``) containing the in-plane and flexural engineering constants should be generated.

    Parameters
    ----------
    - ``laminate`` : Dict | Dictionary containing:
        - ``h`` : float | The height of the laminate.
    - ``settings`` : Dict | Dictionary containing:
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary appended with:
        - ``B`` : jax.numpy.ndarray, shape = (3, 3) | The coupling stiffness (B-)matrix of the laminate.
        - ``alpha`` : jax.numpy.ndarray, shape = (3, 3) | The extensional compliance matrix of the laminate.
        - ``delta`` : jax.numpy.ndarray, shape = (3, 3) | The bending compliance matrix of the laminate.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the in-plane and flexural engineering constants of the laminate
    should be generated, by default `False`.

    Returns
    -------
    - ``laminate`` : Dict | Dictionary appended with:
        - ``E_1m`` : float | The effective in-plane longitudinal modulus.
        - ``E_2m`` : float | The effective in-plane transverse modulus.
        - ``G_12m`` : float | The effective in-plane shear modulus.
        - ``nu_12m`` : float | The effective in-plane Poisson's ratio in the longitudinal direction for the transverse direction.
        - ``nu_21m`` : float | The effective in-plane Poisson's ratio in the transverse direction for the longitudinal direction.
        - ``E_1b`` : float | The effective flexural longitudinal modulus.
        - ``E_2b`` : float | The effective flexural transverse modulus.
        - ``G_12b`` : float | The effective flexural shear modulus.
        - ``nu_12b`` : float | The effective flexural Poisson's ratio in the longitudinal direction for the transverse direction.
        - ``nu_21b`` : float | The effective flexural Poisson's ratio in the transverse direction for the longitudinal direction.

    Output
    ------
    - ``laminate_stiffness.txt`` : text file | A text file containing the in-plane and flexural engineering constants of the laminate.

    Raises
    ------
    - ``None`` : None | The function does not create any output if the laminate is not symmetric as required for the validity of the
    in-plane and flexural engineering constants.

    Assumptions
    -----------
    - The lay-up of the laminate is assumed to be symmetric to ensure meaningfull in-plane and flexural stiffness constants.

    Version
    -------
    - v1.0 :
        - Initial version (06/08/2020) | M. Bilo
    - v1.1 :
        - Updated documentation (13/11/2024) | M. Bilo

    References
    ----------
    [1]: Kassapoglou, Christos. 2013. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 2nd ed.
    Chichester: John Wiley & Sons

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.4
    """

    # Version of the script
    version = f'v1.1'

    # The height, h, of the laminate
    h = laminate['h']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix
    B = stiffness['B']
    # An array of shape-(3, 3) containing the extensional compliance matrix
    alpha = stiffness['alpha']
    # An array of shape-(3, 3) containing the bending compliance matrix
    delta = stiffness['delta']

    # Substantiate the symmetry assumed for the validaty of the in-plane and flexural stiffness constants of the laminate (chapter 3.3 of [1]
    # and chapter 4.4 of [2])
    if not allclose(B, 0):
        # End the function and return the laminate dictionary
        return(laminate)

    # The effective in-plane longitudinal modulus (equation 3.65 from [1] and equation 4.35 from [2])
    #
    #          1
    # E₁ₘ = ---------
    #        h ⋅ α₁₁
    #
    E_1m = 1 / (h * alpha[0, 0] )

    # The effective in-plane transverse modulus (equation 3.65 from [1] and equation 4.37 from [2])
    #
    #          1
    # E₂ₘ = ---------
    #        h ⋅ α₂₂
    #
    E_2m = 1 / (h * alpha[1, 1] )

    # The effective in-plane shear modulus (equation 3.65 from [1] and equation 4.39 from [2])
    #
    #           1
    # G₁₂ₘ = ---------
    #         h ⋅ α₆₆
    #
    G_12m = 1 / (h * alpha[2, 2] )

    # The effective in-plane Poisson's ratio in the longitudinal direction for the transverse direction (equation 3.65 from [1] and
    # equation 4.42 from [2])
    #
    #           α₁₂
    # ν₁₂ₘ = - ------
    #           α₁₁
    #
    nu_12m = - alpha[0, 1] / alpha[0, 0]

    # The effective in-plane Poisson's ratio in the transverse direction for the longitudinal direction (equation 3.65 from [1] and
    # equation 4.45 from [2])
    #
    #           α₁₂
    # ν₂₁ₘ = - ------
    #           α₂₂
    #
    nu_21m = - alpha[0, 1] / alpha[1, 1]

    # The effective flexural longitudinal modulus (equation 3.65 from [1] and equation 4.51 from [2])
    #
    #          12
    # E₁ₕ = -----------
    #        h³ ⋅ δ₁₁
    #
    E_1b = 12 / (h**3 * delta[0, 0] )

    # The effective flexural transverse modulus (equation 3.65 from [1] and equation 4.52 from [2])
    #
    #          12
    # E₂ₕ = -----------
    #        h³ ⋅ δ₂₂
    #
    E_2b = 12 / (h**3 * delta[1, 1] )

    # The effective flexural shear modulus (equation 3.65 from [1] and equation 4.53 from [2])
    #
    #           12
    # G₁₂ₕ = ----------
    #         h³ ⋅ δ₆₆
    #
    G_12b = 12 / (h**3 * delta[2, 2] )

    # The effective flexural Poisson's ratio in the longitudinal direction for the transverse direction (equation 3.65 from [1] and
    # equation 4.54 from [2])
    #
    #           δ₁₂
    # ν₁₂ₕ = - ------
    #           δ₁₁
    #
    nu_12b = - delta[0, 1] / delta[0, 0]

    # The effective flexural Poisson's ratio in the transverse direction for the longitudinal direction (equation 3.65 from [1] and
    # equation 4.55 from [2])
    #
    #           δ₁₂
    # ν₂₁ₕ = - ------
    #           δ₂₂
    #
    nu_21b = - delta[0, 1] / delta[1, 1]

    # If a text file containing the in-plane and flexural engineering constants of the laminate should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/laminate_stiffness.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The in-plane and flexural engineering constants of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: laminate_stiffness.py [{version}] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the in-plane engineering constants of the laminate
            textfile.write(f'In-plane engineering constants\n')
            textfile.write(tabulate( [ [E_1m, E_2m, G_12m, nu_12m, nu_21m], [None, None, None] ],
                                    headers = (f'E\u2081\u2098 [N/m\u00B2]',    f'E\u2082\u2098 [N/m\u00B2]', f'G\u2081\u2082\u2098 [N/m\u00B2]',
                                               f'\u03BD\u2081\u2082\u2098 [-]', f'\u03BD\u2082\u2081\u2098 [-]'),
                                    stralign = 'left', numalign = 'decimal', floatfmt = '.5e') )

            # Add an empty line
            textfile.write(f'\n')

            # Add the flexural engineering constants of the laminate
            textfile.write(f'Flexural engineering constants\n')
            textfile.write(tabulate( [ [E_1b, E_2b, G_12b, nu_12b, nu_21b], [None, None, None] ],
                                    headers = (f'E\u2081\u2095 [N/m\u00B2]',    f'E\u2082\u2095 [N/m\u00B2]', f'G\u2081\u2082\u2095 [N/m\u00B2]',
                                               f'\u03BD\u2081\u2082\u2095 [-]', f'\u03BD\u2082\u2081\u2095 [-]'),
                                    stralign = 'left', numalign = 'decimal', floatfmt = '.5e') )

    # Append the laminate dictionary with the effective in-plane longitudinal modulus
    laminate['E_1m'] = E_1m
    # Append the laminate dictionary with the effective in-plane transverse modulus
    laminate['E_2m'] = E_2m
    # Append the laminate dictionary with the effective in-plane shear modulus
    laminate['G_12m'] = G_12m
    # Append the laminate dictionary with the effective in-plane Poisson's ratio in the longitudinal direction for the transverse direction
    laminate['nu_12m'] = nu_12m
    # Append the laminate dictionary with the effective in-plane Poisson's ratio in the transverse direction for the longitudinal direction
    laminate['nu_21m'] = nu_21m

    # Append the laminate dictionary with the effective flexural longitudinal modulus
    laminate['E_1b'] = E_1b
    # Append the laminate dictionary with the effective flexural transverse modulus
    laminate['E_2b'] = E_2b
    # Append the laminate dictionary with the effective flexural shear modulus
    laminate['G_12b'] = G_12b
    # Append the laminate dictionary with the effective flexural Poisson's ratio in the longitudinal direction for the transverse direction
    laminate['nu_12b'] = nu_12b
    # Append the laminate dictionary with the effective flexural Poisson's ratio in the transverse direction for the longitudinal direction
    laminate['nu_21b'] = nu_21b

    # End the function and return the appended laminate dictionary
    return(laminate)
