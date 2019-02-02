#[derive(Debug)]
pub struct BitWriter {
    out: Vec<u8>, // output buffer
    cache: u8,    // unwritten bits are stored here
    bits: u8,     // number of unwritten bits in cache
}

impl BitWriter {
    #[inline]
    pub fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            cache: 0,
            bits: 0,
        }
    }

    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        BitWriter {
            out: Vec::with_capacity(n),
            cache: 0,
            bits: 0,
        }
    }

    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.out.reserve(n / 8);
    }

    #[inline]
    fn push_unaligned(&mut self, b: u8) {
        self.out.push(self.cache | (b >> self.bits));
        self.cache = (b & ((1 << self.bits) - 1)) << (8 - self.bits);
    }

    #[inline]
    fn reset(&mut self) {
        self.cache = 0;
        self.bits = 0;
    }

    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push(0xef);
    /// w.push(0x4d);
    /// assert_eq!(&[0xef, 0x4d], w.align_ref());
    /// ```
    #[inline]
    pub fn push(&mut self, b: u8) {
        // self.bits will be the same after writing 8 bits,
        // so we don't need to update that.
        if self.bits == 0 {
            self.out.push(b);
        } else {
            self.push_unaligned(b);
        }
    }

    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push_bytes(&[0xef, 0x4d]);
    /// assert_eq!(&[0xef, 0x4d], w.align_ref());
    /// ```
    #[inline]
    pub fn push_bytes(&mut self, p: &[u8]) {
        // self.bits will be the same after writing 8 bits,
        // so we don't need to update that.
        if self.bits == 0 {
            self.out.extend_from_slice(p);
            return;
        }

        for &b in p {
            self.push_unaligned(b);
        }
    }

    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push_bit(true);
    /// w.push_bit(true);
    /// w.push_bit(true);
    /// w.push_bit(false);
    /// w.push_bit(true);
    /// w.push_bit(true);
    /// w.push_bit(true);
    /// w.push_bit(true);
    ///
    /// w.push_bit(false);
    /// w.push_bit(true);
    /// w.push_bit(false);
    /// w.push_bit(false);
    /// w.push_bit(true);
    /// w.push_bit(true);
    /// w.push_bit(false);
    /// w.push_bit(true);
    /// assert_eq!(&[0xef, 0x4d], w.align_ref());
    /// ```
    #[inline]
    pub fn push_bit(&mut self, b: bool) {
        if self.bits == 7 {
            if b {
                self.out.push(self.cache | 1);
            } else {
                self.out.push(self.cache);
            }
            self.reset();
            return;
        }

        self.bits += 1;
        if b {
            self.cache |= 1 << (8 - self.bits);
        }
    }

    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push_bits(0x08, 4);
    /// w.push_bits(0x07, 3);
    /// w.push_bits(0x05, 3);
    /// w.push_bits(0x15, 6);
    /// assert_eq!(&[0x8f, 0x55], w.align_ref());
    /// ```
    ///
    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    ///
    /// // 2 or 0b10
    /// let x: u64 = 2;
    /// let nz = x.leading_zeros(); // 62
    /// let nd = 63u32.saturating_sub(nz); // 1
    /// w.push_bits(std::u64::MAX, nd as u8);
    /// //assert_eq!(&[0b10000000], &w.as_ref());
    /// //w.push_bit(false);
    /// w.push_bit(false);
    /// //assert_eq!(&[0b10000000], &w.as_ref());
    /// w.push_bits(x, nd as u8);
    /// // assert_eq!(&[0b10000000], &w.as_ref());
    ///
    /// // 1 or 0b1
    /// w.push_bit(true);
    /// // assert_eq!(&[0b10010000], &w.align_ref());
    ///
    /// // 4 or 0b100
    /// let x: u64 = 4;
    /// let nz = x.leading_zeros(); // 61
    /// let nd = 63u32.saturating_sub(nz); // 2
    /// w.push_bits(std::u64::MAX, nd as u8);
    /// // assert_eq!(&[0b10011100], &w.align_ref());
    /// w.push_bit(false);
    /// // assert_eq!(&[0b10011100], &w.align_ref());
    /// w.push_bits(x, nd as u8);
    /// println!("{:?}", w);
    /// assert_eq!(&[0b10011100, 0b00000000], &w.align_ref());
    /// println!("{:?}", w);
    /// ```
    #[inline]
    pub fn push_bits(&mut self, r: u64, n: u8) {
        if n == 0 {
            return;
        }
        let mut n = n;

        // the bits higher than n are read as well,
        // so they must be zeroed.
        let r = (r << 64 - n) >> 64 - n;

        let new_bits = self.bits + n;
        if new_bits < 8 {
            // r fits into cache, no write will occur to out
            self.cache |= (r as u8) << (8 - new_bits);
            self.bits = new_bits;
            return;
        }

        if new_bits > 8 {
            // cache will be filled, and there will be more bits to write

            // fill cache and write it out
            let free = 8 - self.bits;
            self.out.push(self.cache | ((r >> (n - free)) as u8));
            n -= free;

            // write out whole bytes
            while n >= 8 {
                n -= 8;
                // no need to mask r, converting to u8 will mask
                // out higher bits
                self.out.push((r >> n) as u8);
            }

            // put remaining into cache
            if n > 0 {
                // note: n < 8 (in case of n=8, 1<<n would overflow u8)
                self.cache = ((r as u8) & ((1 << n) - 1)) << (8 - n);
                self.bits = n;
            } else {
                self.reset();
            }
            return;
        }

        // cache will be filled exactly with the bits to be written
        self.out.push(self.cache | r as u8);
        self.reset();
    }

    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push(0xc1);
    /// w.push_bit(false);
    /// w.push_bits(0x3f, 6);
    /// w.push_bit(true);
    /// w.push(0xac);
    /// w.push_bits(0x01, 1);
    /// w.push_bits(0x1248f, 20);
    /// assert_eq!(3, w.align());
    /// w.push_bytes(&[0x01, 0x02]);
    /// w.push_bits(0x0f, 4);
    /// w.push_bytes(&[0x80, 0x8f]);
    /// assert_eq!(4, w.align());
    /// assert_eq!(0, w.align());
    /// w.push_bits(0x01, 1);
    /// w.push(0xff);
    /// assert_eq!(
    ///     &[0xc1, 0x7f, 0xac, 0x89, 0x24, 0x78, 0x01, 0x02, 0xf8, 0x08, 0xf0, 0xff, 0x80],
    ///     w.align_ref(),
    /// );
    /// ```
    #[inline]
    pub fn align(&mut self) -> u8 {
        let mut skipped: u8 = 0;
        if self.bits > 0 {
            self.out.push(self.cache);
            skipped = 8 - self.bits;
            self.reset();
        }
        skipped
    }

    /// Not safe to call unless aligned (!)
    ///
    /// ```rust
    /// use treeid::bitter::*;
    ///
    /// let mut w = BitWriter::new();
    /// w.push(1);
    /// assert_eq!(&[1], w.align_ref());
    /// w.trailing_pad(4);
    /// assert_eq!(&[1, 0, 0, 0], w.align_ref());
    /// ```
    #[inline]
    pub fn trailing_pad(&mut self, boundary: usize) -> usize {
        let pad_bytes = self.out.len() % boundary;
        if pad_bytes > 0 {
            for _ in 0..boundary - pad_bytes {
                self.out.push(0x0);
            }
        }
        pad_bytes
    }

    #[inline]
    pub fn to_vec(self) -> Vec<u8> {
        self.out
    }

    #[inline]
    pub fn align_vec(mut self) -> Vec<u8> {
        self.align();
        self.out
    }

    #[inline]
    pub fn align_ref(&mut self) -> &[u8] {
        self.align();
        &self.out
    }
}

impl AsRef<[u8]> for BitWriter {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.out
    }
}

extern crate byteorder;
use bitvec::Bits;
use byteorder::{BigEndian, ByteOrder};

/// Returns the `k`'th bit of `x` as a `bool`.
///
/// Behavior is undefined if `k >= 64`.
/// ```rust
/// use treeid::bitter::*;
/// assert_eq!(false, kth_bit(0, 0));
/// assert_eq!(false, kth_bit(0, 4));
/// assert_eq!(false, kth_bit(0, 8));
/// assert_eq!(false, kth_bit(0, 23));
/// assert_eq!(false, kth_bit(1, 0));
/// assert_eq!(true, kth_bit(1, 63));
/// assert_eq!(false, kth_bit(1, 62));
/// assert_eq!(false, kth_bit(1, 23));
/// ```
#[inline]
pub fn kth_bit(x: u64, k: u8) -> bool {
    ((1 << (u64::MASK - k)) & x) != 0
}

/// Returns the `k`'th bit of the next element in `it` as a `bool`,
/// without consuming the element.
///
/// If there is no next element, returns `false`.
///
/// Behavior is undefined if `k >= 64`.
#[inline]
pub fn kth_bit_iter<'a, I>(it: &mut std::iter::Peekable<I>, k: u8) -> bool
where
    I: Iterator<Item = &'a u64>,
{
    match it.peek() {
        Some(&&seg) => kth_bit(seg, k),
        _ => false,
    }
}

/// Adds `y` to `x`, rotating back to `0` if the result
/// exceeds `64`.
///
/// No distinction is made in the number of rotations.
///
/// ```rust
/// use treeid::bitter::*;
/// assert_eq!(1, rotate_add(0, 1));
/// assert_eq!(63, rotate_add(0, 63));
/// assert_eq!(0, rotate_add(0, 64));
/// assert_eq!(1, rotate_add(0, 65));
/// assert_eq!(63, rotate_add(0, 127));
/// assert_eq!(0, rotate_add(0, 128));
/// ```
#[inline]
pub fn rotate_add(x: u8, y: u8) -> u8 {
    (x + y) & u64::MASK
}

/// Advances `cursor` by `bits`, consuming an element of `it` if
/// a rotation is made.
#[inline]
pub fn rotate_consume<'a, I>(it: &mut I, cursor: &mut u8, bits: u8) -> Option<bool>
where
    I: Iterator<Item = &'a u64>,
{
    *cursor = rotate_add(*cursor, bits);
    let rotated = *cursor == 0 && bits != 0;
    if rotated {
        it.next()?;
    }
    Some(rotated)
}

/// Advances `cursor` by `1`, consuming an element of `it` if a
/// rotation is made.
#[inline]
pub fn rotate_incr<'a, I>(it: &mut I, cursor: &mut u8) -> Option<bool>
where
    I: Iterator<Item = &'a u64>,
{
    *cursor = rotate_add(*cursor, 1);
    let rotated = *cursor == 0;
    if rotated {
        it.next()?;
    }
    Some(rotated)
}

/// Pack an mLCF encoded slice of `u8` into chunks of `u64`.
#[inline]
pub fn pack_mlcf(src: &[u8]) -> Vec<u64> {
    let mut dst = vec![0; src.len() / 8];
    BigEndian::read_u64_into(src, &mut dst);
    dst
}
