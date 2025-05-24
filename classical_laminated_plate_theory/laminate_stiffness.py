# Import modules
from datetime         import datetime
from jax.numpy        import allclose, hstack, vstack, ix_
from jax.numpy.linalg import det
from tabulate         import tabulate
from textwrap         import fill

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
    - ``stiffness`` : Dict | Dictionary containing:
        - ``A`` : jax.numpy.ndarray, shape = (3, 3) | The extensional stiffness (A-)matrix of the laminate.
        - ``B`` : jax.numpy.ndarray, shape = (3, 3) | The coupling stiffness (B-)matrix of the laminate.
        - ``D`` : jax.numpy.ndarray, shape = (3, 3) | The bending stiffness (D-)matrix of the laminate.
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

    Assumptions
    -----------

    Version
    -------
    - v1.0 :
        - Initial version (06/08/2020) | M. Bilo
    - v1.1 :
        - Updated documentation (13/11/2024) | M. Bilo
    - v1.2 :
        - Extended the function to include asymmetric laminates based on [3].

    References
    ----------
    [1]: Kassapoglou, Christos. 2013. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 2nd ed.
    Chichester: John Wiley & Sons

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    [3]: Nettles, A.T. 1994. Basic Mechanics of Laminated Composite Plates. Alabama: National Aeronautics and Space Administration

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.4

    [3]: Nettles, A.T. 1994. Basic Mechanics of Laminated Composite Plates. Alabama: National Aeronautics and Space Administration, Chapter V.
    Section C. Example 2 | NOTE: 1) Unlike the statement in example 2 that the extensional stiffness is identical to the A-matrix determined in
    example 1, this is not the case since the number of plies differs for both examples (Example 1: [0, 45, 45, 0]; Example 2: [0, 45]).
    2) The variation in the effective in-plane longitudinal modulus as determined in this example (5839000 lb/in²) versus by this script
    (5.86715e+06 lb/in²) is minor ( (5.86715e+06 - 5839000) / 5839000 * 100% = 0.482%) and a result of the round of intermediate values /
    parameters in the example.
    """

    # Version of the script
    version = f'v1.2'

    # The height, h, of the laminate
    h = laminate['h']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the extensional stiffness (A-)matrix
    A = stiffness['A']
    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix
    B = stiffness['B']
    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']
    # An array of shape-(3, 3) containing the extensional compliance matrix
    alpha = stiffness['alpha']
    # An array of shape-(3, 3) containing the bending compliance matrix
    delta = stiffness['delta']

    # If the laminate is symmetric
    if allclose(B, 0):
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

    # Else, if the laminate is thus asymmetric
    else:
        # The ABD-matrix of the laminate (equation 3.49 from [1], expression 4.29 from [2] and equation 48 from [3])
        #              _                         _
        #             |  A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆  |
        #  _     _    |  A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆  |
        # |  A B  | _ |  A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆  |
        # |_ B D _| ‾ |  B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆  |
        #             |  B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆  |
        #             |_ B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ _|
        #
        ABD = vstack( (hstack( (A, B) ), hstack( (B, D) ) ) )

        # The determinant of the ABD-matrix
        #
        #           | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #           | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        # | A B | _ | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        # | B D | ‾ | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #           | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #           | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        detABD = det(ABD)

        # The effective in-plane longitudinal modulus (equation 84 from [3])
        #
        #             | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #             | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #             | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #             | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #             | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        1    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₁ₘ = --- ⋅ -----------------------------
        #        h      | A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #               | A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #               | B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #               | B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #               | B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        E_1m  = detABD / (h * det(ABD[ [1, 2, 3, 4, 5], :][:, [1, 2, 3, 4, 5] ] ) )

        # The effective in-plane transverse modulus (equation 85 from [3])
        #
        #             | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #             | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #             | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #             | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #             | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        1    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₂ₘ = --- ⋅ -----------------------------
        #        h      | A₁₁ A₁₆ B₁₁ B₁₂ B₁₆ |
        #               | A₁₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #               | B₁₁ B₁₆ D₁₁ D₁₂ D₁₆ |
        #               | B₁₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #               | B₁₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        E_2m  = detABD / (h * det(ABD[ [0, 2, 3, 4, 5], :][:, [0, 2, 3, 4, 5] ] ) )

        # The effective in-plane shear modulus (equation 86 from [3])
        #
        #              | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #              | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #              | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #              | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #              | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #         1    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # G₁₂ₘ = --- ⋅ -----------------------------
        #         h      | A₁₁ A₁₂ B₁₁ B₁₂ B₁₆ |
        #                | A₁₂ A₂₂ B₁₂ B₂₂ B₂₆ |
        #                | B₁₁ B₁₂ D₁₁ D₁₂ D₁₆ |
        #                | B₁₂ B₂₂ D₁₂ D₂₂ D₂₆ |
        #                | B₁₆ B₂₆ D₁₆ D₂₆ D₆₆ |
        #
        G_12m = detABD / (h * det(ABD[ [0, 1, 3, 4, 5], :][:, [0, 1, 3, 4, 5] ] ) )

        # The effective in-plane Poisson's ratio in the longitudinal direction for the transverse direction (equation 88 from [3])
        #
        #           | A₁₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #           | A₁₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #           | B₁₁ B₁₆ D₁₁ D₁₂ D₁₆ |
        #           | B₁₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #           | B₁₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # ν₁₂ₘ = - --------------------------
        #           | A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #           | A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #           | B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #           | B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #           | B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        nu_12m = det(ABD[ [1, 2, 3, 4, 5], :][:, [0, 2, 3, 4, 5] ] ) / det(ABD[ [1, 2, 3, 4, 5], :][:, [1, 2, 3, 4, 5] ] )

        # The effective in-plane Poisson's ratio in the transverse direction for the longitudinal direction (equation 89 from [3])  | NOTE:
        # A₁₆ and B₁₆ in the first column of the top determinate are erroneous and should be A₂₆ and B₂₆ as verified analytically.
        #
        #           | A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #           | A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #           | B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #           | B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #           | B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # ν₂₁ₘ = - --------------------------
        #           | A₁₁ A₁₆ B₁₁ B₁₂ B₁₆ |
        #           | A₁₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #           | B₁₁ B₁₆ D₁₁ D₁₂ D₁₆ |
        #           | B₁₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #           | B₁₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        nu_21m = det(ABD[ [0, 2, 3, 4, 5], :][:, [1, 2, 3, 4, 5] ] ) / det(ABD[ [0, 2, 3, 4, 5], :][:, [0, 2, 3, 4, 5] ] )

        # To derive expressions for the effective flexural moduli and Poisson's ratios, the same basic procedure as for the in-plane engineering
        # constants in [3] is followed and demonstrated for the effective flexural longitudinal modulus (E₁ₕ) below.
        #
        # The constitutive equations in matrix form are given by (equation 3.49 from [1], expression 4.29 from [2] and equation 48 from [3])
        #
        #  _     _     _                         _   _     _
        # |  N₁   |   |  A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆  | |  ε⁰₁  |
        # |  N₂   |   |  A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆  | |  ε⁰₂  |
        # |  N₁₂  | _ |  A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆  | |  γ⁰₁₂ |
        # |  M₁   | ‾ |  B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆  | |  κ₁   |
        # |  M₂   |   |  B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆  | |  κ₂   |
        # |_ M₁₂ _|   |_ B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ _| |_ κ₁₂ _|
        #
        #
        # To determine an expression for the effective flexural longitudinal modulus (E₁ₕ), only the moment resultant related to the x-axis (M₁)
        # is applied reducing the constitutive equation to (in line with equation 81 from [3])
        #
        #  _   _     _                         _   _     _
        # |  0  |   |  A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆  | |  ε⁰₁  |
        # |  0  |   |  A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆  | |  ε⁰₂  |
        # |  0  | _ |  A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆  | |  γ⁰₁₂ |
        # |  M₁ | ‾ |  B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆  | |  κ₁   |
        # |  0  |   |  B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆  | |  κ₂   |
        # |_ 0 _|   |_ B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ _| |_ κ₁₂ _|
        #
        #
        # Using Cramer's rule to solve for κ₁ (in line with equation 82 from [3]) results in
        #
        #       | A₁₁ A₁₂ A₁₆ 0   B₁₂ B₁₆ |
        #       | A₁₂ A₂₂ A₂₆ 0   B₂₂ B₂₆ |
        #       | A₁₆ A₂₆ A₆₆ 0   B₂₆ B₆₆ |
        #       | B₁₁ B₁₂ B₁₆ M₁  D₁₂ D₁₆ |
        #       | B₁₂ B₂₂ B₂₆ 0   D₂₂ D₂₆ |
        #       | B₁₆ B₂₆ B₆₆ 0   D₂₆ D₆₆ |
        # κ₁ = -----------------------------
        #       | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #       | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #       | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #       | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #       | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #       | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        # To simplify this expressions, cofactor expansion is used in line with equation 83 from [3]
        #
        #            | A₁₁ A₁₂ A₁₆ B₁₂ B₁₆ |
        #            | A₁₂ A₂₂ A₂₆ B₂₂ B₂₆ |
        #            | A₁₆ A₂₆ A₆₆ B₂₆ B₆₆ |
        #            | B₁₂ B₂₂ B₂₆ D₂₂ D₂₆ |
        #            | B₁₆ B₂₆ B₆₆ D₂₆ D₆₆ |
        # κ₁ = M₁ -----------------------------
        #          | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #          | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #          | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #          | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #          | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #          | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        #
        # The effective flexural longitudinal modulus (E₁ₕ) is defined as (equation 4.51 from [2])
        #
        #        12 ⋅ M₁      12     M₁
        # E₁ₕ = ---------- = ---- ⋅ ----
        #        h³ ⋅ κ₁      h³     κ₁
        #
        # In line with equation 84 from [3], combining the previous two equations results in the following expressions for this modules
        #
        #              | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #              | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #              | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #              | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #              | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        12    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₁ₕ = ---- ⋅ -----------------------------
        #        h³      | A₁₁ A₁₂ A₁₆ B₁₂ B₁₆ |
        #                | A₁₂ A₂₂ A₂₆ B₂₂ B₂₆ |
        #                | A₁₆ A₂₆ A₆₆ B₂₆ B₆₆ |
        #                | B₁₂ B₂₂ B₂₆ D₂₂ D₂₆ |
        #                | B₁₆ B₂₆ B₆₆ D₂₆ D₆₆ |
        #
        # For the derivation of an expression for the effective flexural Poisson's ratio, the following two definitions are furthermore employed
        # in line with equations 78 and 80 from [3]
        #
        #           κ₁
        # ν₁₂ₕ = - ----, and
        #           κ₂
        #
        #           κ₂
        # ν₂₁ₕ = - ----
        #           κ₁
        #

        # The effective flexural longitudinal modulus
        #
        #              | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #              | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #              | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #              | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #              | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        12    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₁ₕ = ---- ⋅ -----------------------------
        #        h³      | A₁₁ A₁₂ A₁₆ B₁₂ B₁₆ |
        #                | A₁₂ A₂₂ A₂₆ B₂₂ B₂₆ |
        #                | A₁₆ A₂₆ A₆₆ B₂₆ B₆₆ |
        #                | B₁₂ B₂₂ B₂₆ D₂₂ D₂₆ |
        #                | B₁₆ B₂₆ B₆₆ D₂₆ D₆₆ |
        #
        E_1b  = 12 * detABD / (h**3 * det(ABD[ [0, 1, 2, 4, 5], :][:, [0, 1, 2, 4, 5] ] ) )

        # The effective flexural transverse modulus
        #
        #              | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #              | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #              | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #              | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #              | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        12    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₁ₕ = ---- ⋅ -----------------------------
        #        h³      | A₁₁ A₁₂ A₁₆ B₁₁ B₁₆ |
        #                | A₁₂ A₂₂ A₂₆ B₁₂ B₂₆ |
        #                | A₁₆ A₂₆ A₆₆ B₁₆ B₆₆ |
        #                | B₁₁ B₁₂ B₁₆ D₁₁ D₁₆ |
        #                | B₁₆ B₂₆ B₆₆ D₁₆ D₆₆ |
        #
        E_2b  = 12 * detABD / (h**3 * det(ABD[ [0, 1, 2, 3, 5], :][:, [0, 1, 2, 3, 5] ] ) )

        # The effective flexural shear modulus
        #
        #              | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ B₁₆ |
        #              | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ B₂₆ |
        #              | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ B₆₆ |
        #              | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ D₁₆ |
        #              | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ D₂₆ |
        #        12    | B₁₆ B₂₆ B₆₆ D₁₆ D₂₆ D₆₆ |
        # E₁ₕ = ---- ⋅ -----------------------------
        #        h³      | A₁₁ A₁₂ A₁₆ B₁₁ B₁₂ |
        #                | A₁₂ A₂₂ A₂₆ B₁₂ B₂₂ |
        #                | A₁₆ A₂₆ A₆₆ B₁₆ B₂₆ |
        #                | B₁₁ B₁₂ B₁₆ D₁₁ D₁₂ |
        #                | B₁₂ B₂₂ B₂₆ D₁₂ D₂₂ |
        #
        G_12b = 12 * detABD / (h**3 * det(ABD[ [0, 1, 2, 3, 4], :][:, [0, 1, 2, 3, 4] ] ) )

        # The effective flexural Poisson's ratio in the longitudinal direction for the transverse direction
        #
        #           | A₁₁ A₁₂ A₁₆ B₁₁ B₁₆ |
        #           | A₁₂ A₂₂ A₂₆ B₁₂ B₂₆ |
        #           | A₁₆ A₂₆ A₆₆ B₁₆ B₆₆ |
        #           | B₁₂ B₂₂ B₂₆ D₁₂ D₂₆ |
        #           | B₁₆ B₂₆ B₆₆ D₁₆ D₆₆ |
        # ν₁₂ₕ = - --------------------------
        #           | A₁₁ A₁₂ A₁₆ B₁₂ B₁₆ |
        #           | A₁₂ A₂₂ A₂₆ B₂₂ B₂₆ |
        #           | A₁₆ A₂₆ A₆₆ B₂₆ B₆₆ |
        #           | B₁₂ B₂₂ B₂₆ D₂₂ D₂₆ |
        #           | B₁₆ B₂₆ B₆₆ D₂₆ D₆₆ |
        #
        nu_12b = det(ABD[ [0, 1, 2, 4, 5], :][:, [0, 1, 2, 3, 5] ] ) / det(ABD[ [0, 1, 2, 4, 5], :][:, [0, 1, 2, 4, 5] ] )

        # The effective flexural Poisson's ratio in the transverse direction for the longitudinal direction
        #
        #           | A₁₁ A₁₂ A₁₆ B₁₂ B₁₆ |
        #           | A₁₂ A₂₂ A₂₆ B₂₂ B₂₆ |
        #           | A₁₆ A₂₆ A₆₆ B₂₆ B₆₆ |
        #           | B₁₁ B₁₂ B₁₆ D₁₂ D₁₆ |
        #           | B₁₆ B₂₆ B₆₆ D₂₆ D₆₆ |
        # ν₁₂ₕ = - --------------------------
        #           | A₁₁ A₁₂ A₁₆ B₁₁ B₁₆ |
        #           | A₁₂ A₂₂ A₂₆ B₁₂ B₂₆ |
        #           | A₁₆ A₂₆ A₆₆ B₁₆ B₆₆ |
        #           | B₁₁ B₁₂ B₁₆ D₁₁ D₁₆ |
        #           | B₁₆ B₂₆ B₆₆ D₁₆ D₆₆ |
        #
        nu_21b = det(ABD[ [0, 1, 2, 3, 5], :][:, [0, 1, 2, 4, 5] ] ) / det(ABD[ [0, 1, 2, 3, 5], :][:, [0, 1, 2, 3, 5] ] )


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