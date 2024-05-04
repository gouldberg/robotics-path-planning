# Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

import random

import matplotlib.pyplot as plt
import numpy as np

import sys
import copy

import math
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
# class RRT
# -----------------------------------------------------------------------------------------------------

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

class RRT:

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5,
                 goal_sample_rate=10, max_iter=200, sobol_sampler=False):

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None
        self.sobol_sampler = sobol_sampler
        self.sobol_inter_ = 0

    def informed_rrt_star_search(self, animation=True):

        self.node_list = [self.start]
        # max length we expect to find in our 'informed' sample space,
        # starts as infinite
        c_best = float('inf')
        solution_set = set()
        path = None

        # Computing the sampling space
        c_min = math.hypot(self.start.x - self.goal.x,
                           self.start.y - self.goal.y)
        x_center = np.array([[(self.start.x + self.goal.x) / 2.0],
                             [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / c_min],
                       [(self.goal.y - self.start.y) / c_min], [0]])

        e_theta = math.atan2(a1[1, 0], a1[0, 0])
        # first column of identity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        m = a1 @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag(
            [1.0, 1.0,
             np.linalg.det(u) * np.linalg.det(np.transpose(vh))]) @ vh

        for i in range(self.max_iter):
            # Sample space is defined by c_best
            # c_min is the minimum distance between the start point and
            # the goal x_center is the midpoint between the start and the
            # goal c_best changes when a new path is found

            rnd = self.informed_sample(c_best, c_min, x_center, c)
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]
            # steer
            theta = math.atan2(rnd[1] - nearest_node.y,
                               rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)
            d = self.line_cost(nearest_node, new_node)

            no_collision = self.check_collision(nearest_node, theta, d)

            if no_collision:
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                if self.is_near_goal(new_node):
                    if self.check_segment_collision(new_node.x, new_node.y,
                                                    self.goal.x, self.goal.y):
                        solution_set.add(new_node)
                        last_index = len(self.node_list) - 1
                        temp_path = self.get_final_course(last_index)
                        temp_path_len = self.get_path_len(temp_path)
                        if temp_path_len < c_best:
                            path = temp_path
                            c_best = temp_path_len
            if animation:
                self.draw_graph(x_center=x_center, c_best=c_best, c_min=c_min,
                                e_theta=e_theta, rnd=rnd)

        return path

    def choose_parent(self, new_node, near_inds):
        if len(near_inds) == 0:
            return new_node

        d_list = []
        for i in near_inds:
            dx = new_node.x - self.node_list[i].x
            dy = new_node.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                d_list.append(self.node_list[i].cost + d)
            else:
                d_list.append(float('inf'))

        min_cost = min(d_list)
        min_ind = near_inds[d_list.index(min_cost)]

        if min_cost == float('inf'):
            print("min cost is inf")
            return new_node

        new_node.cost = min_cost
        new_node.parent = min_ind

        return new_node

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt(math.log(n_node) / n_node)
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for
                  node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    def informed_sample(self, c_max, c_min, x_center, c):
        if c_max < float('inf'):
            r = [c_max / 2.0, math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            rl = np.diag(r)
            x_ball = self.sample_unit_ball()
            rnd = np.dot(np.dot(c, rl), x_ball) + x_center
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            if self.sobol_sampler:
                rnd = self.sample_free_space_sobol()
            else:
                rnd = self.sample_free_space()

        return rnd

    @staticmethod
    def sample_unit_ball():
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def sample_free_space(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand),
                   random.uniform(self.min_rand, self.max_rand)]
        else:
            rnd = [self.goal.x, self.goal.y]

        return rnd

    def sample_free_space_sobol(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            # sobol sampler
            rnd, n = i4_sobol(2, self.sobol_inter_)
            rnd = self.min_rand + rnd * (self.max_rand - self.min_rand)
            self.sobol_inter_ = n
        else:
            rnd = [self.goal.x, self.goal.y]

        return rnd

    @staticmethod
    def get_path_len(path):
        path_len = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_len += math.hypot(node1_x - node2_x, node1_y - node2_y)

        return path_len

    @staticmethod
    def line_cost(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in
                  nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        new_node = copy.deepcopy(nearest_node)

        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)

        new_node.cost += self.expand_dis
        new_node.parent = n_ind
        return new_node

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def rewire(self, new_node, near_inds):
        n_node = len(self.node_list)
        for i in near_inds:
            near_node = self.node_list[i]

            d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            s_cost = new_node.cost + d

            if near_node.cost > s_cost:
                theta = math.atan2(new_node.y - near_node.y,
                                   new_node.x - near_node.x)
                if self.check_collision(near_node, theta, d):
                    near_node.parent = n_node - 1
                    near_node.cost = s_cost

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)

    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]), np.array([x2, y2]), np.array([ox, oy]))
            if dd <= size ** 2:
                return False  # collision
        return True

    def check_collision(self, near_node, theta, d):
        tmp_node = copy.deepcopy(near_node)
        end_x = tmp_node.x + math.cos(theta) * d
        end_y = tmp_node.y + math.sin(theta) * d
        return self.check_segment_collision(tmp_node.x, tmp_node.y,
                                            end_x, end_y)

    def get_final_course(self, last_index):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph(self, x_center=None, c_best=None, c_min=None, e_theta=None,
                   rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if c_best != float('inf'):
                self.plot_ellipse(x_center, c_best, c_min, e_theta)

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_ellipse(x_center, c_best, c_min, e_theta):  # pragma: no cover

        a = math.sqrt(c_best ** 2 - c_min ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        fx = rot_mat_2d(-angle) @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")


#######################################################################################################
# -----------------------------------------------------------------------------------------------------
# Informed RRT Star
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

# obstacleList = [
#     (5, 5, 0.5), (9, 6, 1), (7, 5, 1), (1, 5, 1), (3, 6, 1), (7, 9, 1)
# ]


start = [0.0, 0.0]
goal = [6.0, 10.0]

goal_sample_rate = 10
rand_area = [-2, 15]

expand_dis = 0.5

# sobol sampler
sobol_sampler = False

max_iter = 200

rrt = RRT(
    start = start,
    goal = goal,
    obstacle_list = obstacleList,
    rand_area = rand_area,
    expand_dis = expand_dis,
    goal_sample_rate = goal_sample_rate,
    max_iter = max_iter,
    sobol_sampler = sobol_sampler,
    )


# ----------
show_animation = True
path = rrt.informed_rrt_star_search(animation=show_animation)

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

