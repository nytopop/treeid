// Copyright (C) 2018 Eric Izoita (nytopop)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! An implementation of rational buckets for lexically ordered
//! collections.
//!
//! References
//! - [0] Dan Hazel
//!   [Using rational numbers to key nested sets](https://arxiv.org/abs/0806.3115)
//! - [1] David W. Matula, Peter Kornerup
//!   [An order preserving finite binary encoding of the rationals](https://www.researchgate.net/publication/261204300_An_order_preserving_finite_binary_encoding_of_the_rationals)

#![feature(test)]

mod bitter;

use self::bitter::*;
use bitvec::Bits;
use std::{cmp, iter};

/// A position in the tree.
///
/// Nodes are encoded to binary in a modified form of LCF[1] (mLCF).
///
/// Deviations from the LCF encoding as described by Matula et al:
///
/// - only suitable for rationals p/q where one (out of 2) of the
///   continued fraction forms has both of the following properties:
///   - composed of an odd number of natural terms
///   - terms at odd indices are always 1
///
/// - leading high bit / low bit is elided because (p >= q >= 1)
///   and we don't need to differentiate from (1 <= p <= q).
/// ```
/// use treeid::Node;
///
/// let node = Node::from(&[2, 4]);
/// let child = Node::from(&[2, 4, 1]);
/// let sibling = Node::from(&[2, 5]);
/// assert!(sibling.to_binary().gt(&child.to_binary()));
/// assert!(child.to_binary().gt(&node.to_binary()));
/// assert_eq!(node, Node::from_binary(&*node.to_binary()).unwrap());
/// ```
#[derive(Debug, PartialEq)]
pub struct Node {
    loc: Vec<u64>,
}

impl Node {
    /// Returns the root node.
    ///
    /// ```rust
    /// use treeid::*;
    /// assert_eq!(Node::from(&[]), Node::root());
    /// ```
    pub fn root() -> Self {
        Node { loc: Vec::new() }
    }

    /// Constructs a node from its tree position as a series of
    /// natural numbers.
    pub fn from(v: &[u64]) -> Self {
        assert!(!v.contains(&0));
        Node {
            loc: v.iter().map(|&x| x).collect(),
        }
    }

    pub fn parent(&self) -> Self {
        let mut parent = self.loc.clone();
        parent.pop();
        Node { loc: parent }
    }

    pub fn parent_mut(&mut self) {
        self.loc.pop();
    }

    pub fn child(&self, id: u64) -> Self {
        let mut child = self.loc.clone();
        child.push(id + 1);
        Node { loc: child }
    }

    pub fn child_mut(&mut self, id: u64) {
        self.loc.push(id + 1);
    }

    pub fn sibling(&self, id: u64) -> Option<Self> {
        let mut sibling = self.loc.clone();
        (*sibling.last_mut()?) = id + 1;
        Some(Node { loc: sibling })
    }

    pub fn sibling_mut(&mut self, id: u64) {
        assert!(id != 0);
        match self.loc.last_mut() {
            None => self.loc.push(id + 1),
            Some(c) => *c = id + 1,
        }
    }

    pub fn pred(&self) -> Option<Self> {
        let mut pred = self.loc.clone();
        let x = pred.last_mut()?;
        if *x < 2 {
            return None;
        }
        *x -= 1;
        Some(Node { loc: pred })
    }

    pub fn succ(&self) -> Self {
        if self.loc.is_empty() {
            return Node { loc: vec![1] };
        }
        let mut succ = self.loc.clone();
        (*succ.last_mut().unwrap()) += 1;
        Node { loc: succ }
    }

    pub fn is_root(&self) -> bool {
        self.loc.is_empty()
    }

    /// Decode an id from its mLCF encoded form. The input must have a
    /// length that is an even multiple of 8 to allow unpacking into an
    /// `&mut [u64]` directly.
    // TODO: use variable width packs
    pub fn from_binary(mlcf_encoded: &[u8]) -> Option<Self> {
        let packed = pack_mlcf(mlcf_encoded);
        let mut it = packed.iter().peekable();

        let mut stack: Vec<u64> = Vec::new();
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
                let until_end: u8 = u64::WIDTH - cursor;
                let mut data_mask = (seg << cursor) >> cursor;
                data_mask >>= until_end.saturating_sub(nz_tot);

                // push them into term. repeated application of
                // this push-copy produces the final value.
                let safe_bits: u8 = cmp::min(nz_tot, until_end);
                term <<= safe_bits;
                term |= data_mask as u64;
                nz_tot -= safe_bits;

                rotate_consume(&mut it, &mut cursor, safe_bits)?;
                if nz_tot == 0 {
                    break 'payloader;
                }
            }
            stack.push(term);

            // if we have gotten here, we have succesfully decoded a
            // term. the bit at cursor is set high if there are any
            // more terms to decode.
            if !kth_bit_iter(&mut it, cursor) {
                break 'chunker;
            }

            // advance the cursor to consume the high bit we just
            // checked for.
            rotate_incr(&mut it, &mut cursor)?;
        }

        Some(Node::from(&stack))
    }

    /// Writes this id into a `Vec<[u8]>` using mLCF encoding. The output
    /// will be padded with trailing zero bytes such that its length is a
    /// multiple of 8 - from_binary() can then pack it directly into an
    /// `&mut [u64]` and decode up to 8 bytes per iter instead of up to 1.
    ///
    /// ```rust
    /// use treeid::*;
    /// assert_eq!(&[0, 0, 0, 0, 0, 0, 0, 0], &*Node::from(&[1]).to_binary());
    /// assert_eq!(&[0b10000000, 0, 0, 0, 0, 0, 0, 0], &*Node::from(&[2]).to_binary());
    /// assert_eq!(&[0b10011000, 0, 0, 0, 0, 0, 0, 0], &*Node::from(&[2, 2]).to_binary());
    /// assert_eq!(
    ///     &[0b11000110, 0b11100111, 0b00100000, 0, 0, 0, 0, 0],
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
        stack.trailing_pad(8);
        stack.to_vec()
    }

    /* A known good decoder that happens to be fairly slow.
     *
    /// Decodes the even indexed denominators of a modified LCF
    /// encoded rational.
    pub fn from_binary(bs: &[u8]) -> Option<Self> {
        let bits: BigVec = BitVec::from(bs);
        let mut it = bits.iter().peekable();
        let mut stack: Vec<u64> = Vec::new();

        for i in 0.. {
            if i % 2 != 0 {
                // odd indices should always be 1 high bit
                guard(it.next()?)?;
            } else {
                let mut k = 0;
                // consume k high bits, 1 low bit
                while let Some(true) = it.next() {
                    k += 1;
                }

                // assemble next term from the next k bits
                let mut term: u64 = 1;
                for _ in 0..k {
                    let bit = it.next()?;
                    term <<= 1;
                    term |= bit as u64;
                }
                stack.push(term);
            }

            match it.peek() {
                None => break,
                Some(false) if i % 2 == 0 => break,
                _ => continue,
            }
        }

        Some(Node::from(&stack))
    }
    */
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            loc: self.loc.clone(),
        }
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
        assert!(n1.to_ratio() < n2.to_ratio());
        assert!(n2.to_binary().gt(&n1.to_binary()));
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
            let nd = Node::from(&v);
            println!("roundtripping: #{} {:?}", i, nd);
            let bin = nd.to_binary();
            println!("{:?}", bin);
            assert_eq!(nd, Node::from_binary(&*bin).unwrap());
        }
    }

    #[test]
    fn lcf_enc() {
        let mut node = Node::from(&[1]);
        let mut last = node.clone();

        // parent < children
        for i in 0..250 {
            node = node.succ();
            if i % 100 == 0 {
                node = node.child(rand::random());
            }

            // num_gt must be true as per proof in [0]
            // lex_gt must be true as per proof in [1]
            let num_gt = node.to_ratio() > last.to_ratio();
            let lex_gt = node.to_binary().gt(&last.to_binary());
            assert_eq!(num_gt, lex_gt, "forward");
            assert!(num_gt || lex_gt, "forward");
            last = node.clone();
        }

        // children < parent.succ()
        while node.loc.len() > 0 {
            for _ in 0..16 {
                node = node.succ();

                // num_gt must be true as per proof in [0]
                // lex_gt must be true as per proof in [1]
                let num_gt = node.to_ratio() > last.to_ratio();
                let lex_gt = node.to_binary().gt(&last.to_binary());
                assert_eq!(num_gt, lex_gt, "backward");
                assert!(num_gt || lex_gt, "backward");

                last = node.clone();
            }

            node = node.parent();
        }
    }
}
