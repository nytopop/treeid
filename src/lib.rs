// Copyright 2019 Eric Izoita (nytopop)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

//! An implementation of rational buckets for lexically ordered
//! collections.
//!
//! References
//! - [0] Dan Hazel
//!   [Using rational numbers to key nested sets](https://arxiv.org/abs/0806.3115)
//! - [1] David W. Matula, Peter Kornerup
//!   [An order preserving finite binary encoding of the rationals](https://www.researchgate.net/publication/261204300_An_order_preserving_finite_binary_encoding_of_the_rationals)

#![feature(test)]
#![feature(specialization)]

use std::{cmp, cmp::Ordering, iter};

pub mod bits;
pub mod bitter;

use self::bits::*;
use self::bitter::*;

/// Represents a location in the treeid hierarchy, and an arbitrary key.
///
/// Taken together, the primary use case is to allow for arbitrarily nested
/// ranges of keys in a flat, ordered collection like [BTreeMap](std::collections::BTreeMap).
///
/// Crucially, the sort order of treeid nodes remains stable even when serialized,
/// allowing for them to be used efficiently with on-disk collections that do not
/// support varying comparison operators. Even in collections that do, the lexicographic
/// sort offered by a serialized treeid node is typically faster (and simpler) than
/// having to deserialize keys for every comparison.
///
/// # Sort order
/// The location of each node in the hierarchy is represented as a sequence of
/// nonzero unsigned integers:
///
/// ```
/// // Hierarchical Structure
/// //
/// //                /------------[root]-------------\
/// //                |                               |
/// //       /-------[1]-------\             /-------[2]-------\
/// //       |                 |             |                 |
/// //  /---[1,1]---\   /---[1,2]---\   /---[2,1]---\   /---[2,2]---\
/// //  |           |   |           |   |           |   |           |
/// // [1,1,1] [1,1,2] [1,2,1] [1,2,2] [2,1,1] [2,1,2] [2,2,1] [2,2,2]
///
/// // Ascending Sort Order
/// //
/// // [root]
/// // [1]
/// // [1,1]
/// // [1,1,1]
/// // [1,1,2]
/// // [1,2]
/// // [1,2,1]
/// // [1,2,2]
/// // [2]
/// // [2,1]
/// // [2,1,1]
/// // [2,1,2]
/// // [2,2]
/// // [2,2,1]
/// // [2,2,2]
/// ```
///
/// Nodes in the same position, but with different keys will be ordered by the
/// key.
///
/// ```
/// use treeid::Node;
///
/// let a = Node::from_parts(&[1, 2], b"hello world");
/// let b = Node::from_parts(&[1, 2, 1], b"1st key");
/// let c = Node::from_parts(&[1, 2, 1], b"2nd key");
/// let d = Node::from_parts(&[1, 3], b"some other key");
///
/// assert!(a < b && b < c && c < d);
/// assert!(a.to_binary() < b.to_binary()
///      && b.to_binary() < c.to_binary()
///      && c.to_binary() < d.to_binary());
/// ```
///
/// # Encoding format
/// Nodes are encoded to binary in a modified form of LCF[1](https://www.researchgate.net/publication/261204300_An_order_preserving_finite_binary_encoding_of_the_rationals) (mLCF).
///
/// Technical deviations from LCF encoding as described by Matula et al:
///
/// - only suitable for rationals p/q where one (out of 2) of the
///   continued fraction forms has both of the following properties:
///   - composed of an odd number of natural terms
///   - terms at odd indices are always 1
///
/// - leading high bit / low bit is elided because (p >= q >= 1)
///   and we don't need to differentiate from (1 <= p <= q).
///
/// - a trailing zero byte is appended to allow for a suffix key
///
/// # Size
/// There is no limit to the length of a treeid position, other than practical
/// concerns w.r.t. space consumption. The total size of the positional portion
/// of an encoded treeid node can be found by taking the sum of 1 + the doubles
/// of minimum binary sizes of each term - 1, and adding the number of terms - 1.
/// The result rounded to the next byte boundary will be the total bitsize.
///
/// A single zero byte will follow to dilineate from the key portion, which is
/// appended unchanged.
///
/// For example, to find the encoded size of the position `&[7, 4, 2]`, we perform:
///
/// - minimum size: `[3 (111), 3 (100), 2 (10)]`
/// - subtract one: `[2, 2, 1]`
/// - double      : `[4, 4, 2]`
/// - add one     : `[5, 5, 3]`
/// - summate     : `13`
/// - add terms-1 : `15`
/// - round to 8  : `16`
/// - add a byte  : `24`
///
/// Which corresponds to the encoded form of:
///
/// `0b11011111 0b00011000 0x0`
///
/// ```
/// use treeid::Node;
///
/// let node = Node::from(&[7, 4, 2]);
/// assert_eq!(
///     // 7    |sep|4    |sep|2  |padding   |key
///     // 11011|1  |11000|1  |100|0000000000|
///
///     &[0b11011_1_11, 0b000_1_100_0, 0],
///     &*node.to_binary(),
/// );
/// ```
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Node {
    loc: Vec<u64>, // location in the tree
    key: Vec<u8>,  // arbitrary key
}

impl Default for Node {
    fn default() -> Self {
        Self::root()
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Node) -> Ordering {
        match self.loc.cmp(&other.loc) {
            Ordering::Equal => self.key.cmp(&other.key),
            o => o,
        }
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: AsRef<[u64]>> From<A> for Node {
    default fn from(loc: A) -> Self {
        assert!(!loc.as_ref().contains(&0));
        Node {
            loc: loc.as_ref().iter().map(|&x| x).collect(),
            key: Vec::new(),
        }
    }
}

impl From<Vec<u64>> for Node {
    fn from(loc: Vec<u64>) -> Self {
        Self::from_vec(loc)
    }
}

impl AsRef<[u8]> for Node {
    fn as_ref(&self) -> &[u8] {
        &self.key
    }
}

impl Node {
    /// Returns the root node.
    ///
    /// ```rust
    /// use treeid::*;
    /// assert_eq!(Node::from(&[]), Node::root());
    /// ```
    pub fn root() -> Self {
        Node {
            loc: Vec::new(),
            key: Vec::new(),
        }
    }

    /// Returns a reference to the tree position of this node.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from_parts(&[1, 2, 3], b"hello worldo");
    ///
    /// assert_eq!(node.position(), &[1, 2, 3]);
    /// ```
    pub fn position(&self) -> &[u64] {
        &self.loc
    }

    /// Returns a reference to the key of this node.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from_parts(&[1, 2, 3], b"hello worldo");
    /// assert_eq!(node.key(), b"hello worldo");
    /// ```
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Constructs a node from its tree position as a series of natural
    /// numbers.
    ///
    /// Panics if the input contains any zeros.
    pub fn from_vec(loc: Vec<u64>) -> Self {
        Self::from_vec_parts(loc, Vec::new())
    }

    /// Constructs a node from its tree position and key.
    ///
    /// Panics if the position contains any zeros.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let from_parts = Node::from_parts(&[1, 2, 3], b"a key of some sort");
    /// let another = Node::from(&[1, 2, 3]).with_key(b"a key of some sort");
    ///
    /// assert_eq!(from_parts, another);
    /// ```
    pub fn from_parts<A: AsRef<[u64]>, B: AsRef<[u8]>>(loc: A, key: B) -> Self {
        assert!(!loc.as_ref().contains(&0));
        Node {
            loc: loc.as_ref().iter().map(|&x| x).collect(),
            key: key.as_ref().iter().map(|&x| x).collect(),
        }
    }

    /// Constructs a node from its (owned) tree position and key.
    ///
    /// Panics if the position contains any zeros.
    pub fn from_vec_parts(loc: Vec<u64>, key: Vec<u8>) -> Self {
        assert!(!loc.contains(&0));
        Node { loc, key }
    }

    /// Returns a node at the same location as the current node, but using
    /// the provided key.
    pub fn with_key<K: AsRef<[u8]>>(&self, key: K) -> Self {
        Node {
            loc: self.loc.clone(),
            key: key.as_ref().iter().map(|&x| x).collect(),
        }
    }

    /// Returns a node at the same location as the current node, but using
    /// the provided (owned) key.
    pub fn with_vec_key(&self, key: Vec<u8>) -> Self {
        Node {
            loc: self.loc.clone(),
            key,
        }
    }

    /// Sets the key for this node.
    pub fn set_key<K: AsRef<[u8]>>(&mut self, key: K) {
        self.key = key.as_ref().iter().map(|&x| x).collect();
    }

    /// Sets the (owned) key for this node.
    pub fn set_vec_key(&mut self, key: Vec<u8>) {
        self.key = key
    }

    /// Get the parent of this node. Sorts before this node and any of its
    /// siblings/children.
    ///
    /// The parent of the root is the root.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from(&[1, 2, 3]);
    /// let par = node.parent();
    ///
    /// assert!(par < node);
    /// assert_eq!(par, Node::from(&[1, 2]));
    /// ```
    pub fn parent(&self) -> Self {
        let mut parent = self.clone();
        parent.parent_mut();
        parent
    }

    pub fn parent_mut(&mut self) {
        self.loc.pop();
    }

    /// Get the specified child of this node. Sorts after this node, but
    /// before any higher siblings.
    ///
    /// Panics if `id` is zero.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from(&[1, 2, 3]);
    /// let sibl = node.sibling(4).unwrap();
    /// let child = node.child(42);
    ///
    /// assert!(child > node);
    /// assert!(child < sibl);
    /// assert_eq!(child, Node::from(&[1, 2, 3, 42]));
    /// ```
    pub fn child(&self, id: u64) -> Self {
        let mut child = self.clone();
        child.child_mut(id);
        child
    }

    pub fn child_mut(&mut self, id: u64) {
        assert!(id != 0);
        self.loc.push(id);
    }

    /// Get the specified sibling of this node. Sort order is dependent on
    /// the value of `id`, relative to the current node's last term.
    ///
    /// Panics if `id` is zero, and returns None for the root.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from(&[1, 2, 3]);
    /// let up = node.sibling(44).unwrap();
    /// let down = node.sibling(2).unwrap();
    ///
    /// assert!(up > node);
    /// assert!(down < node);
    /// assert_eq!(up, Node::from(&[1, 2, 44]));
    /// assert_eq!(down, Node::from(&[1, 2, 2]));
    /// ```
    pub fn sibling(&self, id: u64) -> Option<Self> {
        if self.is_root() {
            return None;
        }

        let mut sibling = self.clone();
        sibling.sibling_mut(id);
        Some(sibling)
    }

    pub fn sibling_mut(&mut self, id: u64) {
        assert!(id != 0);
        if let Some(c) = self.loc.last_mut() {
            *c = id;
        }
    }

    /// Get the last sibling of this node. Sorts before this node.
    ///
    /// Returns None if this is a first child or the root.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from(&[1, 2, 3]);
    /// let pred = node.pred().unwrap();
    ///
    /// assert!(node > pred);
    /// assert_eq!(pred, Node::from(&[1, 2, 2]));
    /// ```
    pub fn pred(&self) -> Option<Self> {
        let mut pred = self.clone();
        let x = pred.loc.last_mut()?;
        if *x < 2 {
            return None;
        }
        *x -= 1;
        Some(pred)
    }

    /// Get the next sibling of this node. Sorts after this node.
    ///
    /// Returns None if this is the root.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from(&[1, 2, 3]);
    /// let succ = node.succ().unwrap();
    ///
    /// assert!(node < succ);
    /// assert_eq!(succ, Node::from(&[1, 2, 4]));
    /// ```
    pub fn succ(&self) -> Option<Self> {
        let mut succ = self.clone();
        (*succ.loc.last_mut()?) += 1;
        Some(succ)
    }

    /// Returns `true` if this is the root.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let root = Node::root();
    /// assert!(root.is_root());
    /// ```
    pub fn is_root(&self) -> bool {
        self.loc.is_empty()
    }

    /// Decode a node from its mLCF encoded form.
    ///
    /// ```rust
    /// use treeid::*;
    ///
    /// let node = Node::from_parts(&[1,2,32,4,32,5,7,5], b"i am a key");
    /// let serialized = node.to_binary();
    /// assert_eq!(node, Node::from_binary(&serialized).unwrap());
    /// ```
    pub fn from_binary<A: AsRef<[u8]>>(mlcf_encoded: A) -> Option<Self> {
        let mut loc: Vec<u64> = Vec::new();

        let mut it = mlcf_encoded.as_ref().iter().peekable();
        let mut cursor: u8 = 0;
        'chunker: loop {
            let mut nz_tot: u8 = 0;
            'prefixer: while let Some(&&seg) = it.peek() {
                let nz = (!(seg << cursor)).leading_zeros();
                nz_tot += nz as u8;

                // if cursor has rotated, we must at least attempt to
                // read some prefix from the next byte. it may or may
                // not actually contain any prefix.
                if rotate_consume(&mut it, &mut cursor, nz as u8)? {
                    continue 'prefixer;
                }

                guard(!kth_bit(seg, cursor))?;
                break 'prefixer;
            }

            // if we are here, we have read the entirety of a unit
            // prefix, and cursor points to the first low bit in the
            // next byte of 'it'.

            // advance the cursor by 1 bit to consume a zero bit
            // indicating a partition between the prefix and data
            // carrying component.
            rotate_incr(&mut it, &mut cursor)?;

            // initialize the term as 1 because we already consumed
            // the (inverted) leading payload bit.
            let mut term: u64 = 1;
            'payloader: while let Some(&&seg) = it.peek() {
                // extract the only bits in the current byte that
                // are part of the term we're reading.
                let until_end: u8 = U8_WIDTH - cursor;
                let mut data_mask = (seg << cursor) >> cursor;
                data_mask >>= until_end.saturating_sub(nz_tot);

                // copy safe_bits bits from data_mask to term.
                let safe_bits: u8 = cmp::min(nz_tot, until_end);
                term <<= safe_bits;
                term |= data_mask as u64;
                nz_tot -= safe_bits;

                rotate_consume(&mut it, &mut cursor, safe_bits)?;
                if nz_tot == 0 {
                    break 'payloader;
                }
            }

            // if we have gotten here, we have succesfully decoded a
            // term. the bit at cursor is set high if there are any
            // more terms to decode.
            loc.push(term);
            if !kth_bit_iter(&mut it, cursor) {
                // consume the current byte because the encoder aligns
                // to the next byte boundary.
                it.next()?;
                break 'chunker;
            }

            // advance the cursor to consume the high bit we just
            // checked for.
            rotate_incr(&mut it, &mut cursor)?;
        }

        guard(it.next()? == &0)?; // consume key separator byte
        let key = it.map(|&x| x).collect(); // key is the rest

        Some(Self::from_vec_parts(loc, key))
    }

    /// Writes this id into a `Vec<[u8]>` using mLCF encoding.
    ///
    /// ```rust
    /// use treeid::*;
    /// assert_eq!(&[0b00000000, 0], &*Node::from(&[1]).to_binary());
    /// assert_eq!(&[0b10000000, 0], &*Node::from(&[2]).to_binary());
    /// assert_eq!(&[0b10011000, 0], &*Node::from(&[2, 2]).to_binary());
    /// assert_eq!(
    ///     &[0b11000110, 0b11100111, 0b00100000, 0],
    ///     &*Node::from(&[4, 3, 2, 5]).to_binary(),
    /// );
    /// ```
    pub fn to_binary(&self) -> Vec<u8> {
        let evens = self.loc.iter();
        let odds = iter::repeat(&1).take(self.loc.len() - 1);
        let it = itertools::interleave(evens, odds);

        let mut stack = BitWriter::new();
        for (i, &x) in it.enumerate() {
            if i % 2 != 0 {
                stack.push_bit(true);
                continue;
            }

            let nz = x.leading_zeros() as u8;
            let nd = 63u8.saturating_sub(nz);
            stack.push_bits(std::u64::MAX, nd);
            stack.push_bit(false);
            stack.push_bits(x, nd);
        }

        stack.align();
        stack.push(0x00);
        stack.push_bytes(&self.key);
        stack.to_vec()
    }
}

fn guard(x: bool) -> Option<()> {
    if x {
        return Some(());
    }
    None
}

#[cfg(test)]
mod tests {
    extern crate rand;
    extern crate test;

    use self::test::Bencher;
    use super::*;
    use num_bigint::BigUint;
    use num_rational::Ratio;

    impl Node {
        fn to_ratio(&self) -> Ratio<BigUint> {
            Self::as_ratio(&self.cf_expansion())
        }

        fn as_ratio(ex: &[u64]) -> Ratio<BigUint> {
            let one = Ratio::new(BigUint::from(1usize), BigUint::from(1usize));
            let mut last = Ratio::from_integer(BigUint::from(0usize));
            for i in (0..ex.len()).rev() {
                let term = &one / (Ratio::new(BigUint::from(ex[i]), BigUint::from(1usize)) + last);
                last = term;
            }
            last.recip()
        }

        fn cf_expansion(&self) -> Vec<u64> {
            let evens = self.loc.iter();
            let odds = iter::repeat(&1).take(self.loc.len() - 1);
            itertools::interleave(evens, odds).map(|&x| x).collect()
        }
    }

    #[test]
    fn child_parent_eq() {
        let b = Node::from(&[1, 2, 3]);
        assert_eq!(b, b.child(4).parent());
    }

    #[bench]
    fn binary_lo_2(b: &mut Bencher) {
        let v: Vec<u64> = (1..=2).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_lo_4(b: &mut Bencher) {
        let v: Vec<u64> = (1..=4).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_lo_8(b: &mut Bencher) {
        let v: Vec<u64> = (1..=8).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_lo_16(b: &mut Bencher) {
        let v: Vec<u64> = (1..=16).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_lo_32(b: &mut Bencher) {
        let v: Vec<u64> = (1..=32).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_lo_64(b: &mut Bencher) {
        let v: Vec<u64> = (1..=64).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }

    #[bench]
    fn binary_hi_2(b: &mut Bencher) {
        let v: Vec<u64> = (1..=2).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_hi_4(b: &mut Bencher) {
        let v: Vec<u64> = (1..=4).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_hi_8(b: &mut Bencher) {
        let v: Vec<u64> = (1..=8).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_hi_16(b: &mut Bencher) {
        let v: Vec<u64> = (1..=16).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_hi_32(b: &mut Bencher) {
        let v: Vec<u64> = (1..=32).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }
    #[bench]
    fn binary_hi_64(b: &mut Bencher) {
        let v: Vec<u64> = (1..=64).map(|_| rand::random()).collect();
        let node = Node::from(&v);
        b.iter(|| Node::from_binary(&*node.to_binary()).unwrap());
    }

    #[test]
    fn edge_case() {
        let n1 = Node::from(&[2, 4, 1]); // 2, 1, 4, 1, 1
        let n2 = Node::from(&[2, 5]); //    2, 1, 4, 1
        println!("2 . 4 . 1 : {:?}", Node::from_binary(&*n1.to_binary()),);
        println!();
        println!("2 . 5     : {:?}", Node::from_binary(&*n2.to_binary()),);
        println!();
        assert!(n1 < n2);
        assert!(n1.to_ratio() < n2.to_ratio());
        assert!(n1.to_binary() < n2.to_binary());
    }

    struct BfsIter {
        stack: BigUint,
        radix: u32,
    }

    impl BfsIter {
        fn new(rdx: u32) -> BfsIter {
            BfsIter {
                stack: BigUint::new(vec![]),
                radix: rdx,
            }
        }
    }

    impl Iterator for BfsIter {
        type Item = Vec<u64>;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self
                .stack
                .to_radix_le(self.radix)
                .iter()
                .map(|&x| (x + 1) as u64)
                .collect();
            self.stack += BigUint::from(1u8);
            Some(item)
        }
    }

    #[test]
    fn bfs_iter_round_trip() {
        let mut it = BfsIter::new(15);
        it.next().unwrap();

        for i in 0..2u64.pow(14) {
            let v = it.next().unwrap();
            println!("raw input is: {:?}", v);

            let nd = Node::from_vec_parts(v, (1..=24).map(|_| rand::random()).collect());
            println!("roundtripping: #{} {:?}", i, nd);

            let bin = nd.to_binary();
            println!("binary: {:?}", bin);

            assert_eq!(nd, Node::from_binary(&*bin).unwrap());
        }
    }

    #[test]
    fn lcf_enc() {
        let mut node = Node::from_parts(&[1], b"hello worldo");
        let mut last = node.clone();

        // parent < children
        for i in 0..250 {
            node = node.succ().unwrap();
            if i % 100 == 0 {
                node.child_mut(rand::random());
            }

            assert!(last < node);
            // must be true as per proof in [0]
            assert!(last.to_ratio() < node.to_ratio());
            // must be true as per proof in [1]
            assert!(last.to_binary() < node.to_binary());

            last = node.clone();
        }

        // children < parent.succ()
        while node.loc.len() > 0 {
            for _ in 0..16 {
                node = node.succ().unwrap();

                assert!(last < node);
                // must be true as per proof in [0]
                assert!(last.to_ratio() < node.to_ratio());
                // must be true as per proof in [1]
                assert!(last.to_binary() < node.to_binary());

                last = node.clone();
            }

            node.parent_mut();
        }
    }
}
