"""Linear Feedback Shift Register toolkit.

References:
http://www.eng.auburn.edu/~strouce/class/elec6250/LFSRs.pdf
https://www.xilinx.com/support/documentation/application_notes/xapp210.pdf
https://users.ece.cmu.edu/~koopman/lfsr/
https://en.wikipedia.org/wiki/Linear-feedback_shift_register
"""


def lfsr_if(poly, init=1):
    """ Linear Feedback Shift Register with internal feedback.

    Example: poly = 0x0D (x**3+x**2+1), init=0x01:

    +----<-----+-------<-------+
    |          |               |
    +-----[0]-XOR-[0]-----[1]--+-->-- out

    x**3     x**2    x**1    x**0

    cycle state
      0    001
      1    110
      2    011
      3    111
      4    101
      5    100
      6    010

    >>> n = 3
    >>> taps = max_len_lfsr_min_taps[n]
    >>> poly = taps_to_poly(taps)
    >>> prng = lfsr_if(poly)
    >>> for i in range(10):
    ...     print(i, next(prng))
    0 1
    1 6
    2 3
    3 7
    4 5
    5 4
    6 2
    7 1
    8 6
    9 3

    Reference: http://www.eng.auburn.edu/~strouce/class/elec6250/LFSRs.pdf
    """

    assert(poly % 2)
    feed = poly // 2
    state = init
    while True:
        lsb = state & 1
        yield state

        state = (state >> 1)
        if lsb:
            state ^= feed


def lfsr_ef(poly, init=1):
    """ Linear Feedback Shift Register with external feedback.

    Implements LFSR counter for many use cases. For instance for certain
    polynomial and init, LSB of the output can be considered as
    pseudo-random sequence.

    Example: poly = 0x0D (x**3+x**2+1), init=0x01:

    +----<----XOR------<-------+
    |          |               |
    +-----[0]--+--[0]-----[1]--+-->-- out

    x**3     x**2    x**1    x**0

    cycle state
      0    001
      1    100
      2    110
      3    111
      4    011
      5    101
      6    010

    >>> n = 3
    >>> taps = max_len_lfsr_min_taps[n]
    >>> poly = taps_to_poly(taps)
    >>> prng = lfsr_ef(poly)
    >>> for i in range(10):
    ...     print(i, next(prng))
    0 1
    1 4
    2 6
    3 7
    4 3
    5 5
    6 2
    7 1
    8 4
    9 6

    Reference: http://www.eng.auburn.edu/~strouce/class/elec6250/LFSRs.pdf
    """

    def bits(x):
        return [int(b) for b in bin(x)[2:]]
    assert(poly % 2)
    state = init
    n = len(bits(poly)) - 2
    msb = 1 << n
    while True:
        yield state

        xor_in = bits(state & poly)
        xor_out = sum(xor_in) % 2
        state = (state >> 1)
        if xor_out:
            state += msb


def taps_to_poly(taps, append0=True):
    """Converts vector of taps to corresponding polynomial.

    Note that tap #0 defaults to 1, unless `append0` is False

    Example:
        Vector of taps [4,3,0] correspond to polynomial x**4+x**3+1
        which is represented as 0x19

    """
    taps = set(list(taps) + [0])
    if not append0:
        taps.remove(0)
    return sum([2**x for x in taps])


def measure_period(cntr):
    """Measures how many cycles it takes for the counter overflow.
    """
    buf = []
    while True:
        x = next(cntr)
        if x in buf:
            break
        buf.append(x)
    return len(buf)
