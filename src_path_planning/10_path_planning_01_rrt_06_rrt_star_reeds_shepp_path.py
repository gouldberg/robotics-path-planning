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
# Reeds Shepp Path
# -----------------------------------------------------------------------------------------------------

class Path:
    """
    Path data container
    """

    def __init__(self):
        # course segment length  (negative value is backward segment)
        self.lengths = []
        # course segment type char ("S": straight, "L": left, "R": right)
        self.ctypes = []
        self.L = 0.0  # Total lengths of the path
        self.x = []  # x positions
        self.y = []  # y positions
        self.yaw = []  # orientations [rad]
        self.directions = []  # directions (1:forward, -1:backward)


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc,
                  ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def pi_2_pi(x):
    return angle_mod(x)

def mod2pi(x):
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v

def set_path(paths, lengths, ctypes, step_size):
    path = Path()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = (i_path.ctypes == path.ctypes)
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # check path is long enough
    if path.L <= step_size:
        return paths  # too short, so do not insert path

    paths.append(path)
    return paths


def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']

    return False, [], []


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ['L', 'S', 'R']

    return False, [], []


def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']

    return False, [], []


def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2*A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi/2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi/2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2*u)
        if ((t >= 0) and (u >= 0) and (v >= 0)):
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if (0 <= u2 <= 1):
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi + math.pi/2)
        if (t >= 0) and (v >= 0):
           return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'L']

    return False, [], []


def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi/2)
        v = mod2pi(t - phi - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'R', 'L']

    return False, [], []


def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi/2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'R']

    return False, [], []


def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -math.pi/2, v], ['L', 'R', 'S', 'L', 'R']

    return False, [], []


def timeflip(travel_distances):
    return [-x for x in travel_distances]


def reflect(steering_directions):
    def switch_dir(dirn):
        if dirn == 'L':
            return 'R'
        elif dirn == 'R':
            return 'L'
        else:
            return 'S'
    return[switch_dir(dirn) for dirn in steering_directions]


def generate_path(q0, q1, max_curvature, step_size):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    paths = []
    path_functions = [left_straight_left, left_straight_right,                          # CSC
                      left_x_right_x_left, left_x_right_left, left_right_x_left,        # CCC
                      left_right_x_left_right, left_x_right_left_x_right,               # CCCC
                      left_x_right90_straight_left, left_x_right90_straight_right,      # CCSC
                      left_straight_right90_x_left, left_straight_left90_x_right,       # CSCC
                      left_x_right90_straight_left90_x_right]                           # CCSCC

    for path_func in path_functions:
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths


def calc_interpolate_dists_list(lengths, step_size):
    interpolate_dists_list = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size
        interp_dists = np.arange(0.0, length, d_dist)
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list


def generate_local_course(lengths, modes, max_curvature, step_size):
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0

    xs, ys, yaws, directions = [], [], [], []
    for (interp_dists, mode, length) in zip(interpolate_dists_list, modes,
                                            lengths):

        for dist in interp_dists:
            x, y, yaw, direction = interpolate(dist, length, mode,
                                               max_curvature, origin_x,
                                               origin_y, origin_yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions


def interpolate(dist, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw):
    if mode == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:  # curve
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = origin_yaw - dist
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, 1 if length > 0.0 else -1


def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc, step_size)
    for path in paths:
        xs, ys, yaws, directions = generate_local_course(path.lengths,
                                                         path.ctypes, maxc,
                                                         step_size)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for
                  (ix, iy) in zip(xs, ys)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for
                  (ix, iy) in zip(xs, ys)]
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc

    return paths


def reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=0.2):
    paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)
    if not paths:
        return None, None, None, None, None  # could not generate any path

    # search minimum cost path
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    return b_path.x, b_path.y, b_path.yaw, b_path.ctypes, b_path.lengths


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
            # Reeds Shepp Path
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
                 step_size=0.2
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
        # Reeds Shepp Path
        self.curvature = curvature
        self.goal_yaw_th = goal_yaw_th
        self.goal_xy_th = goal_xy_th
        self.step_size = step_size

    def set_random_seed(self, seed):
        random.seed(seed)

    def planning(self, animation=True, search_until_max_iter=True):
        """
        planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)
                    self.try_goal_path(new_node)

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

    def try_goal_path(self, node):

        goal = self.Node(self.end.x, self.end.y, self.end.yaw)

        new_node = self.steer(node, goal)
        if new_node is None:
            return

        if self.check_collision(
                new_node, self.obstacle_list, self.robot_radius):
            self.node_list.append(new_node)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw, to_node.x,
            to_node.y, to_node.yaw, self.curvature, self.step_size)

        if not px:
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(l) for l in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw, to_node.x,
            to_node.y, to_node.yaw, self.curvature, self.step_size)
        if not course_lengths:
            return float("inf")

        return from_node.cost + sum([abs(l) for l in course_lengths])

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)
        print("goal_indexes:", len(goal_indexes))

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        print("final_goal_indexes:", len(final_goal_indexes))

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        print("min_cost:", min_cost)
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy, iyaw) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_yaw)):
                path.append([ix, iy, iyaw])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.yaw])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return hypot(dx, dy)

    def get_random_node(self):

        rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand),
                        random.uniform(-math.pi, math.pi)
                        )

        return rnd

    # Sobol Sampler
    def get_random_node_sobol(self):
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

# -----------------------------------------------------------------------------------------------------
# function for smoothing
# -----------------------------------------------------------------------------------------------------

def get_path_length(path):
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d

    return le


def get_target_point(path, targetL):
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break

    partRatio = (le - targetL) / lastPairLen

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

    return [x, y, ti]


def line_collision_check(first, second, obstacleList):
    # Line Equation

    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]

    try:
        a = y2 - y1
        b = -(x2 - x1)
        c = y2 * (x2 - x1) - x2 * (y2 - y1)
    except ZeroDivisionError:
        return False

    for (ox, oy, size) in obstacleList:
        d = abs(a * ox + b * oy + c) / (math.hypot(a, b))
        if d <= size:
            return False

    return True  # OK


def path_smoothing(path, max_iter, obstacle_list):
    le = get_path_length(path)

    for i in range(max_iter):
        # Sample two points
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])

        if first[2] <= 0 or second[2] <= 0:
            continue

        if (second[2] + 1) > len(path):
            continue

        if second[2] == first[2]:
            continue

        # collision check
        if not line_collision_check(first, second, obstacle_list):
            continue

        # Create New path
        newPath = []
        newPath.extend(path[:first[2] + 1])
        newPath.append([first[0], first[1]])
        newPath.append([second[0], second[1]])
        newPath.extend(path[second[2] + 1:])
        path = newPath
        le = get_path_length(path)

    return path


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# RRT Star
#  + Reeds Shepp Path
#  + Sobol Sampler
# -----------------------------------------------------------------------------------------------------

# [x, y, radius]
obstacleList = [
    (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1)
] 

# obstacleList = [
#     (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2)
# ] 

# obstacleList = [
#     (5, 5, 1), (4, 6, 1), (4, 8, 1), (4, 10, 1), (6, 5, 1), (7, 5, 1), (8, 6, 1), (8, 8, 1), (8, 10, 1)
# ]


start = [0.0, 0.0, np.deg2rad(0.0)]
goal = [10.0, 9.0, np.deg2rad(0.0)]
# goal = [6.0, 7.0, np.deg2rad(90.0)]

robot_radius = 0.6

# Area Bounds
play_area = None
# play_area = [0, 10, 0, 14]

# limit of expanding distance at one time
expand_dis = 3.0
path_resolution = 0.5
# expand_dis = 1.0
# path_resolution = 0.1

curvature = 1.0 * 2
goal_yaw_th = np.deg2rad(1.0)
goal_xy_th = 0.5

# if goal_sample_rate = 5:
# 5%:  sampled at goal
# 95%:  sampled from x: random.uniform(rand_area[0], rand_area[1]), y: random.uniform(rand_area[0], rand_area[1])
goal_sample_rate = 10
rand_area = [-2, 15]

# sobol sampler
sobol_sampler = True

max_iter = int(500 * 1.5)
step_size = 0.1

# RRT star
connect_circle_dist = 50.0
search_until_max_iter = False


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
    goal_xy_th = goal_xy_th,
    step_size = step_size,
    )


# ----------
show_animation = True
path = rrt.planning(animation=show_animation)

print(path)



# ----------
# smoothing path:  this is optional
path_xy = [[x, y] for (x, y, yaw) in path]
maxIter = 1000
smoothedPath = path_smoothing(path_xy, maxIter, obstacleList)



# ----------
# show final path
rrt.draw_graph()
plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
plt.plot([x for (x, y) in smoothedPath], [y for (x, y) in smoothedPath], '-k')
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

