# -*- coding: utf-8 -*-
"""Linear Feedback Shift Register toolkit.

References:
http://www.eng.auburn.edu/~strouce/class/elec6250/LFSRs.pdf
https://www.xilinx.com/support/documentation/application_notes/xapp210.pdf
https://users.ece.cmu.edu/~koopman/lfsr/
https://en.wikipedia.org/wiki/Linear-feedback_shift_register
"""
from __future__ import print_function
import copy
import string
import itertools
import numpy


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


class LFSR(object):

    @property
    def algorithm(self):
        raise NotImplementedError('Use any of the subclasses')

    @property
    def taps(self):
        return self.poly_to_taps(self.poly)

    def __init__(self, taps, seed):
        if not isinstance(taps, (int, long)):
            taps = self.taps_to_poly(taps)
        if isinstance(seed, (str, bytes, type(u''))):
            seed = _string_to_binarray(seed)
        if not isinstance(seed, (int, long)):
            seed = _binarray_to_int(seed)
        self.__iter = None
        self.poly = taps
        self.seed = seed
        self.key_length = max(self.taps)
        self.state = seed
        self.lfsr = self.algorithm(self.poly, self.seed)
        assert self.seed != 0

    def poly_to_taps(self, poly):
        return list(reversed([i for i, x in enumerate(reversed(bin(poly)[2:])) if x == '1']))

    def taps_to_poly(self, taps):
        return taps_to_poly(taps)

    def get_restarted(self):
        return type(self)(self.poly, self.seed)

    def get_period(self):
        return measure_period(self.get_restarted().lfsr)

    def getchar(self):
        return chr(self.getbyte())

    def getnext(self):
        return int(''.join(map(str, self.getbits(self.key_length))), 2)

    def getbyte(self):
        return int(''.join(map(str, self.getbits(8))), 2)

    def getbits(self, num):
        return list(itertools.islice(self, num))

    def getbit(self):
        return next(self)

    def encrypt(self, plaintext):
        """Encrypt a message using XOR One Time Pad"""
        lfsr = self.get_restarted()
        ciphertext = ''
        for char in plaintext:
            ciphertext += chr(ord(char) ^ lfsr.getbyte())
        return ciphertext

    def decrypt(self, ciphertext):
        return self.encrypt(ciphertext)

    def __call__(self):
        while True:
            self.state = data = next(self.lfsr)
            for x in bin(data)[2:].zfill(self.key_length):
                yield int(x)

    def __iter__(self):
        return self()

    def __next__(self):
        if self.__iter is None:
            self.__iter = self.__iter__()
        try:
            return next(self.__iter)
        except StopIteration:
            self.__iter = None
            raise
    next = __next__

    def __repr__(self):
        return '<%s(seed=0x%X (%dbit), polynomial=0x%X, state=0x%X)>' % (type(self).__name__, self.seed, self.key_length, self.poly, self.state)


class Galois(LFSR):
    @property
    def algorithm(self):
        def _fixed(poly, seed):  # AFAIK: galois must not yield the initial state, fix the broken implementation of lfsr_if()
            s = lfsr_if(poly, seed)
            assert next(s) == seed
            while True:
                yield next(s)
        return _fixed


class BrokenGalois(LFSR):
    @property
    def algorithm(self):
        return lfsr_if


class Fibonacci(LFSR):
    """
    >>> x = Fibonacci([3, 2, 1, 0], 1)
    >>> x
    <Fibonacci(seed=0x1 (3bit), polynomial=0xF, state=0x1)>
    >>> x.getbits(8)
    [0, 0, 1, 1, 0, 0, 1, 1]
    >>> x.getbits(8)
    [0, 1, 1, 0, 0, 1, 1, 0]
    >>> x.getbits(8)
    [1, 1, 0, 0, 1, 1, 0, 0]
    >>> x.getbits(8)
    [1, 0, 0, 1, 1, 0, 0, 1]
    >>> x.getbits(8)
    [0, 0, 1, 1, 0, 0, 1, 1]
    >>> x.getbits(8)
    [0, 1, 1, 0, 0, 1, 1, 0]
    >>> x.getbits(8)
    [1, 1, 0, 0, 1, 1, 0, 0]
    >>> x.getbits(8)
    [1, 0, 0, 1, 1, 0, 0, 1]
    """
    @property
    def algorithm(self):
        return lfsr_ef


class BerlekampMasseyAlgorithm(object):
    """Calculates the shortest LFSR. Given any sequence of bytes or binary array.

    >>> BerlekampMasseyAlgorithm(Fibonacci([3, 2, 1, 0], 1).getbits(80))
    BMA(3bit, taps=[3, 2, 1, 0]) for data='3333333333'
    """

    def __init__(self, data_sequence):
        if isinstance(data_sequence, (str, bytes, type(u''))):
            data_sequence = _string_to_binarray(data_sequence)
        self.data_sequence = data_sequence
        self.bit_length, coefficients = self.berlekamp_massey_algorithm(data_sequence)
        self.taps = self.get_taps(coefficients)
        self.key_length = max(self.taps)

    @property
    def data(self):
        return _binarray_to_string(self.data_sequence, True)

    def get_taps(self, coefficients):
        return list(reversed([max(0, t) for t, v in enumerate(coefficients) if v == 1]))

    def __repr__(self):
        return 'BMA(%dbit, taps=%r) for data=%r' % (self.bit_length, self.taps, self.data)

    def berlekamp_massey_algorithm(self, block_data):
        """
        An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
        [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
        The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
        for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
        sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
        non-zero elements to have a multiplicative inverse."""
        n = len(block_data)
        c = numpy.zeros(n)
        b = numpy.zeros(n)
        c[0], b[0] = 1, 1
        ll, m, i = 0, -1, 0
        int_data = [int(el) for el in block_data]
        assert set(int_data) == set([0, 1])
        while i < n:
            v = int_data[(i - ll):i]
            v = v[::-1]
            cc = c[1:ll + 1]
            d = (int_data[i] + numpy.dot(v, cc)) % 2
            if d == 1:
                temp = copy.copy(c)
                p = numpy.zeros(n)
                for j in range(0, ll):
                    if b[j] == 1:
                        p[j + i - m] = 1
                c = (c + p) % 2
                if ll <= 0.5 * i:
                    ll = i + 1 - ll
                    m = i
                    b = temp
            i += 1
        return ll, c


class KnownPlaintextAttack(object):
    """Perform a known plaintext attack against a LFSR using berlekamp massey algorithm and some guessed plaintext chunks"""

    forbidden_plaintext_chars = set(map(chr, range(256))) - set(string.printable)

    def __init__(self, ciphertext, known_plaintexts=None, known_bit_length=None):
        self.ciphertext = ciphertext
        self.known_plaintexts = known_plaintexts or []
        self.known_bit_length = known_bit_length

    def crack(self):
        print()
        print('Trying to crack', self.ciphertext.encode('base64'))
        print()

        for word in self.get_plaintexts():
            print('Bruteforce using', word, len(word))
            for pos in self.get_positions(word):
                chunk = self.ciphertext[pos:pos + len(word)]
                state = self.xorword(chunk, word)
                bma = BerlekampMasseyAlgorithm(state)
                if not self.bma_suitable(bma):
                    print('ignored')
                    continue

                state = state[:int(bma.key_length / 8.0)]

                # for tap_ in (bma.taps, [max(0, x - 1) for x in bma.taps]):
                for LFSR in (Fibonacci, Galois, BrokenGalois):
                    lfsr = LFSR(bma.taps, state)
                    decrypted = lfsr.decrypt(self.ciphertext[pos:])

                    if not set(decrypted) & self.forbidden_plaintext_chars:
                        print('Found possible match: %r' % (decrypted,))
                        print(repr(lfsr))
                        print(repr(bma))
                        print()

    def bma_suitable(self, bma):
        print(repr(bma))
        if 0 not in bma.taps:  # polynomal must contain "+ x^0"
            return False
        if len(bma.taps) % 2:  # taps must be even!
            return False
        if self.known_bit_length and bma.bit_length != self.known_bit_length:
            return False
        return True

    def xorword(self, a, b):
        return _binarray_to_string(self.xor(_string_to_binarray(a), _string_to_binarray(b)))

    def xor(self, a, b):
        # assert len(a) == len(b), (len(a), len(b))
        return [b1 ^ b2 for b1, b2 in zip(a, b)]

    def get_plaintexts(self):
        for word in self.known_plaintexts:
            if self.known_bit_length and len(word) < (2 * self.known_bit_length / 8.0):
                print('Word %r too small! Ignoring.' % (word,))
                continue
            yield word

    def get_positions(self, word):
        for _ in range(len(self.ciphertext) - len(word)):
            yield _


def _binarray_to_string(data, ignore_padding=False):
    """Create a string from a binary array
    >>> _binarray_to_string([0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    'test'
    """
    return ''.join(map(chr, _binarray_to_bytes(data, ignore_padding)))


def _binarray_to_bytes(data, ignore_padding=False):
    if not ignore_padding:
        assert not len(data) % 8
    while data:
        x, data = data[:8], data[8:]
        yield _binarray_to_int(x)


def _binarray_to_int(data):
    """Create integer from binary array
    >>> _binarray_to_int(_string_to_binarray('test'))
    1952805748
    """
    return int(''.join(map(str, data)), 2)


def _string_to_binarray(data):
    """Create a binary array out of a string.
    >>> _string_to_binarray('test')
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    """
    bindata = ''.join(bin(ord(x))[2:].zfill(8) for x in data)
    return [0 if x == '0' else 1 for x in bindata]
