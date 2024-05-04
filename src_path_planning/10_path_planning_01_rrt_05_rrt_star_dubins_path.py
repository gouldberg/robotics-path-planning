# Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

import random

import matplotlib.pyplot as plt
import numpy as np

import sys
import copy

import math
from math import sin, cos, atan2, sqrt, acos, pi, hypot
from scipy.spatial.transform import Rotation as Rot


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Sobol Sampler
# -----------------------------------------------------------------------------------------------------

"""
  Licensing:
    This code is distributed under the MIT license.

  Authors:
    Original FORTRAN77 version of i4_sobol by Bennett Fox.
    MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    Original Python version of is_prime by Corrado Chisari

    Original MATLAB versions of other functions by John Burkardt.
    PYTHON versions by Corrado Chisari

    Original code is available at
    http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html

    Note: the i4 prefix means that the function takes a numeric argument or
          returns a number which is interpreted inside the function as a 4
          byte integer
    Note: the r4 prefix means that the function takes a numeric argument or
          returns a number which is interpreted inside the function as a 4
          byte float
"""

atmost = None
dim_max = None
dim_num_save = None
initialized = None
lastq = None
log_max = None
maxcol = None
poly = None
recipd = None
seed_save = None
v = None


def i4_bit_hi1(n):
    """
     I4_BIT_HI1 returns the position of the high 1 bit base 2 in an I4.

      Discussion:

        An I4 is an integer ( kind = 4 ) value.

      Example:

           N    Binary    Hi 1
        ----    --------  ----
           0           0     0
           1           1     1
           2          10     2
           3          11     2
           4         100     3
           5         101     3
           6         110     3
           7         111     3
           8        1000     4
           9        1001     4
          10        1010     4
          11        1011     4
          12        1100     4
          13        1101     4
          14        1110     4
          15        1111     4
          16       10000     5
          17       10001     5
        1023  1111111111    10
        1024 10000000000    11
        1025 10000000001    11

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        26 October 2014

      Author:

        John Burkardt

      Parameters:

        Input, integer N, the integer to be measured.
        N should be nonnegative.  If N is nonpositive, the function
        will always be 0.

        Output, integer BIT, the position of the highest bit.

    """
    i = n
    bit = 0

    while True:

        if i <= 0:
            break

        bit = bit + 1
        i = i // 2

    return bit


def i4_bit_lo0(n):
    """
     I4_BIT_LO0 returns the position of the low 0 bit base 2 in an I4.

      Discussion:

        An I4 is an integer ( kind = 4 ) value.

      Example:

           N    Binary    Lo 0
        ----    --------  ----
           0           0     1
           1           1     2
           2          10     1
           3          11     3
           4         100     1
           5         101     2
           6         110     1
           7         111     4
           8        1000     1
           9        1001     2
          10        1010     1
          11        1011     3
          12        1100     1
          13        1101     2
          14        1110     1
          15        1111     5
          16       10000     1
          17       10001     2
        1023  1111111111    11
        1024 10000000000     1
        1025 10000000001     2

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        John Burkardt

      Parameters:

        Input, integer N, the integer to be measured.
        N should be nonnegative.

        Output, integer BIT, the position of the low 1 bit.

    """
    bit = 0
    i = n

    while True:

        bit = bit + 1
        i2 = i // 2

        if i == 2 * i2:
            break

        i = i2

    return bit


def i4_sobol_generate(m, n, skip):
    """


     I4_SOBOL_GENERATE generates a Sobol dataset.

      Licensing:

        This code is distributed under the MIT license.

      Modified:

        22 February 2011

      Author:

        Original MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

      Parameters:

        Input, integer M, the spatial dimension.

        Input, integer N, the number of points to generate.

        Input, integer SKIP, the number of initial points to skip.

        Output, real R(M,N), the points.

    """
    r = np.zeros((m, n))
    for j in range(1, n + 1):
        seed = skip + j - 2
        [r[0:m, j - 1], seed] = i4_sobol(m, seed)
    return r


def i4_sobol(dim_num, seed):
    """


     I4_SOBOL generates a new quasirandom Sobol vector with each call.

      Discussion:

        The routine adapts the ideas of Antonov and Saleev.

      Licensing:

        This code is distributed under the MIT license.

      Modified:

        22 February 2011

      Author:

        Original FORTRAN77 version by Bennett Fox.
        MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

      Reference:

        Antonov, Saleev,
        USSR Computational Mathematics and Mathematical Physics,
        olume 19, 19, pages 252 - 256.

        Paul Bratley, Bennett Fox,
        Algorithm 659:
        Implementing Sobol's Quasirandom Sequence Generator,
        ACM Transactions on Mathematical Software,
        Volume 14, Number 1, pages 88-100, 1988.

        Bennett Fox,
        Algorithm 647:
        Implementation and Relative Efficiency of Quasirandom
        Sequence Generators,
        ACM Transactions on Mathematical Software,
        Volume 12, Number 4, pages 362-376, 1986.

        Ilya Sobol,
        USSR Computational Mathematics and Mathematical Physics,
        Volume 16, pages 236-242, 1977.

        Ilya Sobol, Levitan,
        The Production of Points Uniformly Distributed in a Multidimensional
        Cube (in Russian),
        Preprint IPM Akad. Nauk SSSR,
        Number 40, Moscow 1976.

      Parameters:

        Input, integer DIM_NUM, the number of spatial dimensions.
        DIM_NUM must satisfy 1 <= DIM_NUM <= 40.

        Input/output, integer SEED, the "seed" for the sequence.
        This is essentially the index in the sequence of the quasirandom
        value to be generated.    On output, SEED has been set to the
        appropriate next value, usually simply SEED+1.
        If SEED is less than 0 on input, it is treated as though it were 0.
        An input value of 0 requests the first (0-th) element of the sequence.

        Output, real QUASI(DIM_NUM), the next quasirandom vector.

    """

    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if not initialized or dim_num != dim_num_save:
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1
        #
        #    Initialize (part of) V.
        #
        v = np.zeros((dim_max, log_max))
        v[0:40, 0] = np.transpose([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ])

        v[2:40, 1] = np.transpose([
            1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3,
            3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3
        ])

        v[3:40, 2] = np.transpose([
            7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7,
            5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3
        ])

        v[5:40, 3] = np.transpose([
            1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9,
            13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9
        ])

        v[7:40, 4] = np.transpose([
            9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31,
            11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9
        ])

        v[13:40, 5] = np.transpose([
            37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9,
            49, 33, 19, 29, 11, 19, 27, 15, 25
        ])

        v[19:40, 6] = np.transpose([
            13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3,
            113, 61, 89, 45, 107
        ])

        v[37:40, 7] = np.transpose([7, 23, 39])
        #
        #    Set POLY.
        #
        poly = [
            1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109,
            103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299
        ]

        atmost = 2**log_max - 1
        #
        #    Find the number of bits in ATMOST.
        #
        maxcol = i4_bit_hi1(atmost)
        #
        #    Initialize row 1 of V.
        #
        v[0, 0:maxcol] = 1

        # Things to do only if the dimension changed.

    if dim_num != dim_num_save:
        #
        #    Check parameters.
        #
        if (dim_num < 1 or dim_max < dim_num):
            print('I4_SOBOL - Fatal error!')
            print('    The spatial dimension DIM_NUM should satisfy:')
            print('        1 <= DIM_NUM <= %d' % dim_max)
            print('    But this input value is DIM_NUM = %d' % dim_num)
            return None

        dim_num_save = dim_num
        #
        #    Initialize the remaining rows of V.
        #
        for i in range(2, dim_num + 1):
            #
            #    The bits of the integer POLY(I) gives the form of polynomial
            #    I.
            #
            #    Find the degree of polynomial I from binary encoding.
            #
            j = poly[i - 1]
            m = 0
            while True:
                j = math.floor(j / 2.)
                if (j <= 0):
                    break
                m = m + 1
            #
            #    Expand this bit pattern to separate components of the logical
            #    array INCLUD.
            #
            j = poly[i - 1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = math.floor(j / 2.)
                includ[k - 1] = (j != 2 * j2)
                j = j2
            #
            #    Calculate the remaining elements of row I as explained
            #    in Bratley and Fox, section 2.
            #
            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l_var = 1
                for k in range(1, m + 1):
                    l_var = 2 * l_var
                    if (includ[k - 1]):
                        newv = np.bitwise_xor(
                            int(newv), int(l_var * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv
#
#    Multiply columns of V by appropriate power of 2.
#
        l_var = 1
        for j in range(maxcol - 1, 0, -1):
            l_var = 2 * l_var
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l_var
#
#    RECIPD is 1/(common denominator of the elements in V).
#
        recipd = 1.0 / (2 * l_var)
        lastq = np.zeros(dim_num)

    seed = int(math.floor(seed))

    if (seed < 0):
        seed = 0

    if (seed == 0):
        l_var = 1
        lastq = np.zeros(dim_num)

    elif (seed == seed_save + 1):
        #
        #    Find the position of the right-hand zero in SEED.
        #
        l_var = i4_bit_lo0(seed)

    elif (seed <= seed_save):

        seed_save = 0
        lastq = np.zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l_var = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l_var - 1]))

        l_var = i4_bit_lo0(seed)

    elif (seed_save + 1 < seed):

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l_var = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l_var - 1]))

        l_var = i4_bit_lo0(seed)
#
#    Check that the user is not calling too many times!
#
    if maxcol < l_var:
        print('I4_SOBOL - Fatal error!')
        print('    Too many calls!')
        print('    MAXCOL = %d\n' % maxcol)
        print('    L =            %d\n' % l_var)
        return None


#
#    Calculate the new components of QUASI.
#
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(
            int(lastq[i - 1]), int(v[i - 1, l_var - 1]))

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]


def i4_uniform_ab(a, b, seed):
    """


     I4_UNIFORM_AB returns a scaled pseudorandom I4.

      Discussion:

        The pseudorandom number will be scaled to be uniformly distributed
        between A and B.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        05 April 2013

      Author:

        John Burkardt

      Reference:

        Paul Bratley, Bennett Fox, Linus Schrage,
        A Guide to Simulation,
        Second Edition,
        Springer, 1987,
        ISBN: 0387964673,
        LC: QA76.9.C65.B73.

        Bennett Fox,
        Algorithm 647:
        Implementation and Relative Efficiency of Quasirandom
        Sequence Generators,
        ACM Transactions on Mathematical Software,
        Volume 12, Number 4, December 1986, pages 362-376.

        Pierre L'Ecuyer,
        Random Number Generation,
        in Handbook of Simulation,
        edited by Jerry Banks,
        Wiley, 1998,
        ISBN: 0471134031,
        LC: T57.62.H37.

        Peter Lewis, Allen Goodman, James Miller,
        A Pseudo-Random Number Generator for the System/360,
        IBM Systems Journal,
        Volume 8, Number 2, 1969, pages 136-143.

      Parameters:

        Input, integer A, B, the minimum and maximum acceptable values.

        Input, integer SEED, a seed for the random number generator.

        Output, integer C, the randomly chosen integer.

        Output, integer SEED, the updated seed.

    """

    i4_huge = 2147483647

    seed = int(seed)

    seed = (seed % i4_huge)

    if seed < 0:
        seed = seed + i4_huge

    if seed == 0:
        print('')
        print('I4_UNIFORM_AB - Fatal error!')
        print('  Input SEED = 0!')
        sys.exit('I4_UNIFORM_AB - Fatal error!')

    k = (seed // 127773)

    seed = 167 * (seed - k * 127773) - k * 2836

    if seed < 0:
        seed = seed + i4_huge

    r = seed * 4.656612875E-10
    #
    #  Scale R to lie between A-0.5 and B+0.5.
    #
    a = round(a)
    b = round(b)

    r = (1.0 - r) * (min(a, b) - 0.5) \
        + r * (max(a, b) + 0.5)
    #
    #  Use rounding to convert R to an integer between A and B.
    #
    value = round(r)

    value = max(value, min(a, b))
    value = min(value, max(a, b))
    value = int(value)

    return value, seed


def prime_ge(n):
    """


     PRIME_GE returns the smallest prime greater than or equal to N.

      Example:

          N    PRIME_GE

        -10     2
          1     2
          2     2
          3     3
          4     5
          5     5
          6     7
          7     7
          8    11
          9    11
         10    11

      Licensing:

        This code is distributed under the MIT license.

      Modified:

        22 February 2011

      Author:

        Original MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

      Parameters:

        Input, integer N, the number to be bounded.

        Output, integer P, the smallest prime number that is greater
        than or equal to N.

    """
    p = max(math.ceil(n), 2)
    while not isprime(p):
        p = p + 1

    return p


def isprime(n):
    """


     IS_PRIME returns True if N is a prime number, False otherwise

      Licensing:

        This code is distributed under the MIT license.

      Modified:

        22 February 2011

      Author:

        Corrado Chisari

      Parameters:

        Input, integer N, the number to be checked.

        Output, boolean value, True or False

    """
    if n != int(n) or n < 1:
        return False
    p = 2
    while p < n:
        if n % p == 0:
            return False
        p += 1

    return True


def r4_uniform_01(seed):
    """


     R4_UNIFORM_01 returns a unit pseudorandom R4.

      Discussion:

        This routine implements the recursion

          seed = 167 * seed mod ( 2^31 - 1 )
          r = seed / ( 2^31 - 1 )

        The integer arithmetic never requires more than 32 bits,
        including a sign bit.

        If the initial seed is 12345, then the first three computations are

          Input     Output      R4_UNIFORM_01
          SEED      SEED

             12345   207482415  0.096616
         207482415  1790989824  0.833995
        1790989824  2035175616  0.947702

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        04 April 2013

      Author:

        John Burkardt

      Reference:

        Paul Bratley, Bennett Fox, Linus Schrage,
        A Guide to Simulation,
        Second Edition,
        Springer, 1987,
        ISBN: 0387964673,
        LC: QA76.9.C65.B73.

        Bennett Fox,
        Algorithm 647:
        Implementation and Relative Efficiency of Quasirandom
        Sequence Generators,
        ACM Transactions on Mathematical Software,
        Volume 12, Number 4, December 1986, pages 362-376.

        Pierre L'Ecuyer,
        Random Number Generation,
        in Handbook of Simulation,
        edited by Jerry Banks,
        Wiley, 1998,
        ISBN: 0471134031,
        LC: T57.62.H37.

        Peter Lewis, Allen Goodman, James Miller,
        A Pseudo-Random Number Generator for the System/360,
        IBM Systems Journal,
        Volume 8, Number 2, 1969, pages 136-143.

      Parameters:

        Input, integer SEED, the integer "seed" used to generate
        the output random number.  SEED should not be 0.

        Output, real R, a random value between 0 and 1.

        Output, integer SEED, the updated seed.  This would
        normally be used as the input seed on the next call.

    """

    i4_huge = 2147483647

    if (seed == 0):
        print('')
        print('R4_UNIFORM_01 - Fatal error!')
        print('  Input SEED = 0!')
        sys.exit('R4_UNIFORM_01 - Fatal error!')

    seed = (seed % i4_huge)

    if seed < 0:
        seed = seed + i4_huge

    k = (seed // 127773)

    seed = 167 * (seed - k * 127773) - k * 2836

    if seed < 0:
        seed = seed + i4_huge

    r = seed * 4.656612875E-10

    return r, seed


def r8mat_write(filename, m, n, a):
    """


     R8MAT_WRITE writes an R8MAT to a file.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        12 October 2014

      Author:

        John Burkardt

      Parameters:

        Input, string FILENAME, the name of the output file.

        Input, integer M, the number of rows in A.

        Input, integer N, the number of columns in A.

        Input, real A(M,N), the matrix.
    """

    with open(filename, 'w') as output:
        for i in range(0, m):
            for j in range(0, n):
                s = '  %g' % (a[i, j])
                output.write(s)
            output.write('\n')


def tau_sobol(dim_num):
    """


     TAU_SOBOL defines favorable starting seeds for Sobol sequences.

      Discussion:

        For spatial dimensions 1 through 13, this routine returns
        a "favorable" value TAU by which an appropriate starting point
        in the Sobol sequence can be determined.

        These starting points have the form N = 2**K, where
        for integration problems, it is desirable that
                TAU + DIM_NUM - 1 <= K
        while for optimization problems, it is desirable that
                TAU < K.

      Licensing:

        This code is distributed under the MIT license.

      Modified:

        22 February 2011

      Author:

        Original FORTRAN77 version by Bennett Fox.
        MATLAB version by John Burkardt.
        PYTHON version by Corrado Chisari

      Reference:

        IA Antonov, VM Saleev,
        USSR Computational Mathematics and Mathematical Physics,
        Volume 19, 19, pages 252 - 256.

        Paul Bratley, Bennett Fox,
        Algorithm 659:
        Implementing Sobol's Quasirandom Sequence Generator,
        ACM Transactions on Mathematical Software,
        Volume 14, Number 1, pages 88-100, 1988.

        Bennett Fox,
        Algorithm 647:
        Implementation and Relative Efficiency of Quasirandom
        Sequence Generators,
        ACM Transactions on Mathematical Software,
        Volume 12, Number 4, pages 362-376, 1986.

        Stephen Joe, Frances Kuo
        Remark on Algorithm 659:
        Implementing Sobol's Quasirandom Sequence Generator,
        ACM Transactions on Mathematical Software,
        Volume 29, Number 1, pages 49-57, March 2003.

        Ilya Sobol,
        USSR Computational Mathematics and Mathematical Physics,
        Volume 16, pages 236-242, 1977.

        Ilya Sobol, YL Levitan,
        The Production of Points Uniformly Distributed in a Multidimensional
        Cube (in Russian),
        Preprint IPM Akad. Nauk SSSR,
        Number 40, Moscow 1976.

      Parameters:

                Input, integer DIM_NUM, the spatial dimension.    Only values
                of 1 through 13 will result in useful responses.

                Output, integer TAU, the value TAU.

    """
    dim_max = 13

    tau_table = [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35]

    if 1 <= dim_num <= dim_max:
        tau = tau_table[dim_num]
    else:
        tau = -1

    return tau

#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# helpers for angle
# -----------------------------------------------------------------------------------------------------

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Dubins Path Planning
# -----------------------------------------------------------------------------------------------------

def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                     step_size=0.1, selected_types=None):
    """
    Plan dubins path

    Parameters
    ----------
    s_x : float
        x position of the start point [m]
    s_y : float
        y position of the start point [m]
    s_yaw : float
        yaw angle of the start point [rad]
    g_x : float
        x position of the goal point [m]
    g_y : float
        y position of the end point [m]
    g_yaw : float
        yaw angle of the end point [rad]
    curvature : float
        curvature for curve [1/m]
    step_size : float (optional)
        step size between two path points [m]. Default is 0.1
    selected_types : a list of string or None
        selected path planning types. If None, all types are used for
        path planning, and minimum path length result is returned.
        You can select used path plannings types by a string list.
        e.g.: ["RSL", "RSR"]

    Returns
    -------
    x_list: array
        x positions of the path
    y_list: array
        y positions of the path
    yaw_list: array
        yaw angles of the path
    modes: array
        mode list of the path
    lengths: array
        arrow_length list of the path segments.

    Examples
    --------
    You can generate a dubins path.

    >>> start_x = 1.0  # [m]
    >>> start_y = 1.0  # [m]
    >>> start_yaw = np.deg2rad(45.0)  # [rad]
    >>> end_x = -3.0  # [m]
    >>> end_y = -3.0  # [m]
    >>> end_yaw = np.deg2rad(-45.0)  # [rad]
    >>> curvature = 1.0
    >>> path_x, path_y, path_yaw, mode, _ = plan_dubins_path(
                start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
    >>> plt.plot(path_x, path_y, label="final course " + "".join(mode))
    >>> plot_arrow(start_x, start_y, start_yaw)
    >>> plot_arrow(end_x, end_y, end_yaw)
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.show()

    .. image:: dubins_path.jpg
    """
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size,
        planning_funcs)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                      step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                      curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw


# -----------------------------------------------------------------------------------------------------
# helpers for plot
# -----------------------------------------------------------------------------------------------------

def plot_arrow(x, y, yaw, arrow_length=1.0,
               origin_point_plot_style="xr",
               head_width=0.1, fc="r", ec="k", **kwargs):
    """
    Plot an arrow or arrows based on 2D state (x, y, yaw)

    All optional settings of matplotlib.pyplot.arrow can be used.
    - matplotlib.pyplot.arrow:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.arrow.html

    Parameters
    ----------
    x : a float or array_like
        a value or a list of arrow origin x position.
    y : a float or array_like
        a value or a list of arrow origin y position.
    yaw : a float or array_like
        a value or a list of arrow yaw angle (orientation).
    arrow_length : a float (optional)
        arrow length. default is 1.0
    origin_point_plot_style : str (optional)
        origin point plot style. If None, not plotting.
    head_width : a float (optional)
        arrow head width. default is 0.1
    fc : string (optional)
        face color
    ec : string (optional)
        edge color
    """
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, head_width=head_width,
                       fc=fc, ec=ec, **kwargs)
    else:
        plt.arrow(x, y,
                  arrow_length * cos(yaw),
                  arrow_length * sin(yaw),
                  head_width=head_width,
                  fc=fc, ec=ec,
                  **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# class RRT
#   - this class includes also:  class Node, class AreaBounds
# -----------------------------------------------------------------------------------------------------

class RRT:
    class Node:
        def __init__(self, x, y, yaw):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            # ---------------------
            # RRT star
            self.cost = 0.0
            # ---------------------
            # Dubins Path
            self.yaw = yaw
            self.path_yaw = []

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 sobol_sampler=True,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 curvature=1.0,
                 goal_yaw_th=np.deg2rad(1.0),
                 goal_xy_th=0.5,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        # ---------------------
        # sobol sampler
        self.sobol_sampler = sobol_sampler
        self.sobol_inter_ = 0
        # ---------------------
        # RRT star
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        # ---------------------
        # for dubins path
        self.curvature = curvature  # for dubins path
        self.goal_yaw_th = goal_yaw_th
        self.goal_xy_th = goal_xy_th

    def planning(self, animation=True, search_until_max_iter=True):
        """
        RRT Star planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            # if self.check_if_outside_play_area(new_node, self.play_area) and \
            #     self.check_collision(new_node, self.obstacle_list, self.robot_radius):
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = \
            plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw,
                to_node.x, to_node.y, to_node.yaw, self.curvature)

        if len(px) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_length = plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        return from_node.cost + course_length

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return hypot(dx, dy)

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-pi, pi)
                            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    # Sobol Sampler
    def get_random_node_sobol(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            # sobol sampler
            rand_coordinates, n = i4_sobol(3, self.sobol_inter_)
            rand_coordinates_0 = rand_coordinates[:2]
            rnd = rand_coordinates[2]

            rand_coordinates_0 = self.min_rand + \
                rand_coordinates_0 * (self.max_rand - self.min_rand)
            rnd = -math.pi + rnd * math.pi
            rand_coordinates = [rand_coordinates_0[0], rand_coordinates_0[1], rnd]

            self.sobol_inter_ = n
            rnd = self.Node(*rand_coordinates)

        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = hypot(dx, dy)
        theta = atan2(dy, dx)
        return d, theta

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and \
                self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                # self.check_collision(t_node, self.obstacle_list, self.robot_radius) and \
                # self.check_if_outside_play_area(t_node, self.play_area):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None


    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            # no_collision = self.check_collision(edge_node, self.obstacle_list, self.robot_radius) and \
            #     self.check_if_outside_play_area(edge_node, self.play_area)
            no_collision = self.check_collision(edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)



#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# RRT Star
#  + Durbin Path
#  + Sobol Sampler
# -----------------------------------------------------------------------------------------------------

_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL, "RLR": _RLR, "LRL": _LRL, }

# [x, y, radius]
# obstacleList = [
#     (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1)
# ] 

obstacleList = [
    (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2)
] 

start = [0.0, 0.0, np.deg2rad(0.0)]
goal = [10.0, 10.0, np.deg2rad(0.0)]

robot_radius = 0.0

# Area Bounds
play_area = None
# play_area = [0, 10, 0, 14]

# limit of expanding distance at one time
expand_dis = 3.0
path_resolution = 0.5
# expand_dis = 1.0
# path_resolution = 0.1

curvature = 1.0
goal_yaw_th = np.deg2rad(1.0)
goal_xy_th = 0.5

# if goal_sample_rate = 5:
# 5%:  sampled at goal
# 95%:  sampled from x: random.uniform(rand_area[0], rand_area[1]), y: random.uniform(rand_area[0], rand_area[1])
goal_sample_rate = 10
rand_area = [-2, 15]

# sobol sampler
sobol_sampler = True

max_iter = 500

# RRT star
connect_circle_dist = 50.0
search_until_max_iter = True

rrt = RRT(
    start = start,
    goal = goal,
    obstacle_list = obstacleList,
    rand_area = rand_area,
    expand_dis = expand_dis,
    path_resolution = path_resolution,
    goal_sample_rate = goal_sample_rate,
    max_iter = max_iter,
    play_area = play_area,
    robot_radius = robot_radius,
    sobol_sampler = sobol_sampler,
    connect_circle_dist = connect_circle_dist,
    search_until_max_iter = search_until_max_iter,
    curvature = curvature,
    goal_yaw_th = goal_yaw_th,
    goal_xy_th = goal_xy_th
    )


# ----------
show_animation = True
path = rrt.planning(animation=show_animation)

print(path)


# ----------
# show final path
rrt.draw_graph()
plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
plt.grid(True)
# Need for Mac
plt.pause(0.01)
plt.show()


# ----------
# created nodes
len(rrt.node_list)

# node info
idx = 1
rrt.node_list[idx].path_x
rrt.node_list[idx].path_y



# ----------
# show final path (step by step)
plt.clf()
for node in rrt.node_list:
    if node.parent:
        plt.plot(node.path_x, node.path_y, "-g")

for (ox, oy, size) in rrt.obstacle_list:
    plt.plot(ox, oy, "ok", ms=30 * size)

if rrt.play_area is not None:
    plt.plot([rrt.play_area.xmin, rrt.play_area.xmax,
                rrt.play_area.xmax, rrt.play_area.xmin,
                rrt.play_area.xmin],
                [rrt.play_area.ymin, rrt.play_area.ymin,
                rrt.play_area.ymax, rrt.play_area.ymax,
                rrt.play_area.ymin],
                "-k")

plt.plot(rrt.start.x, rrt.start.y, "xr")
plt.plot(rrt.end.x, rrt.end.y, "xr")
plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
plt.axis("equal")
plt.axis([-2, 15, -2, 15])
plt.grid(True)
plt.pause(0.01)
