pub const U8_MASK: u8 = 7;
pub const U8_WIDTH: u8 = 8;

/// Returns the `k`'th bit of `x` as a `bool`.
///
/// Behavior is undefined if `k >= 8`.
/// ```rust
/// use treeid::bits::*;
/// assert_eq!(false, kth_bit(0, 0));
/// assert_eq!(false, kth_bit(0, 4));
/// assert_eq!(false, kth_bit(0, 7));
/// assert_eq!(false, kth_bit(0, 6));
/// assert_eq!(false, kth_bit(1, 0));
/// assert_eq!(true, kth_bit(1, 7));
/// assert_eq!(false, kth_bit(1, 6));
/// assert_eq!(false, kth_bit(1, 5));
/// ```
#[inline]
pub fn kth_bit(x: u8, k: u8) -> bool {
    ((1 << (U8_MASK - k)) & x) != 0
}

/// Returns the `k`'th bit of the next element in `it` as a `bool`,
/// without consuming the element.
///
/// If there is no next element, returns `false`.
///
/// Behavior is undefined if `k >= 8`.
#[inline]
pub fn kth_bit_iter<'a, I>(it: &mut std::iter::Peekable<I>, k: u8) -> bool
where
    I: Iterator<Item = &'a u8>,
{
    match it.peek() {
        Some(&&seg) => kth_bit(seg, k),
        _ => false,
    }
}

/// Adds `y` to `x`, rotating back to `0` if the result
/// exceeds `8`.
///
/// No distinction is made in the number of rotations.
///
/// ```rust
/// use treeid::bits::*;
/// assert_eq!(1, rotate_add(0, 1));
/// assert_eq!(7, rotate_add(0, 7));
/// assert_eq!(0, rotate_add(0, 8));
/// assert_eq!(1, rotate_add(0, 9));
/// assert_eq!(7, rotate_add(0, 15));
/// assert_eq!(0, rotate_add(0, 16));
/// ```
#[inline]
pub fn rotate_add(x: u8, y: u8) -> u8 {
    (x + y) & U8_MASK
}

/// Advances `cursor` by `bits`, consuming an element of `it` if
/// a rotation is made.
#[inline]
pub fn rotate_consume<'a, I>(it: &mut I, cursor: &mut u8, bits: u8) -> Option<bool>
where
    I: Iterator<Item = &'a u8>,
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
    I: Iterator<Item = &'a u8>,
{
    *cursor = rotate_add(*cursor, 1);
    let rotated = *cursor == 0;
    if rotated {
        it.next()?;
    }
    Some(rotated)
}
