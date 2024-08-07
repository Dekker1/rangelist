//! A library for working with set of values represented as inclusive ranges.
//!
//! This library provides a [`RangeList`] struct that can be used to represent
//! sets of values as a collection of inclusive ranges. The ranges are stored
//! in a deduplicated sorted order.
//!
//! Additionally, the library provides a [`IntervalIter`] trait that can be
//! implemented by types that can provide an iterator of sorted inclusive
//! ranges. This trait then provides methods to perform set operations on
//! any combination of types that implement it.

use std::{
	collections::{BTreeSet, HashSet},
	fmt::{Debug, Display},
	iter::{Map, Peekable},
	ops::RangeInclusive,
};

use castaway::match_type;
use num::{cast::AsPrimitive, Integer};

/// An iterator combinator that given two iterators yielding ordered ranges,
/// yields the ordered ranges of elements that are in the ranges yielded by
/// `lhs` iterator, but does not include elements that are in the ranges yielded
/// by the `rhs` iterator.
#[derive(Debug)]
pub struct DiffIter<
	E: Clone + Integer,
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
> {
	/// Iterator yielding the ranges of the elements to be included
	lhs: Peekable<I>,
	/// Iterator yielding the ranges of the elements to be excluded
	rhs: Peekable<J>,
	/// The maximum of the last range yielded
	max: E,
}

/// An iterator combinator that given two iterators yielding ordered ranges,
/// yields the ordered ranges that are in the intersection of the ranges yielded
/// by the iterators.
#[derive(Debug)]
pub struct IntersectIter<
	E: PartialOrd,
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
> {
	/// Iterator yielding the ranges of the left-hand side of the intersection
	lhs: Peekable<I>,
	/// Iterator yielding the ranges of the right-hand side of the intersection
	rhs: Peekable<J>,
}

/// A trait that provides operations on iterators of orderdered intervals.
pub trait IntervalIterator<E: PartialOrd> {
	/// The type of the interval iterator.
	type IntervalIter: Iterator<Item = RangeInclusive<E>>;
	/// Returns an iterator over the ordered intervals.
	fn intervals(&self) -> Self::IntervalIter;

	/// Returns the number of elements contained within the RangeList.
	fn card(&self) -> usize
	where
		E: AsPrimitive<usize> + Integer,
	{
		self.intervals()
			.map(|r| {
				let mut diff = *r.end() - *r.start();
				diff.inc();
				diff.as_()
			})
			.sum()
	}

	/// Compute RangeList without any of the elements in the ranges of `other`.
	///
	/// # Warning
	///
	/// The implementation decrements the lowest value of `self` and increments
	/// the largest value of `self`. This could cause a panic if this causes
	/// overflow in `E`.
	fn diff<O, R>(&self, other: &O) -> R
	where
		E: Clone + Integer,
		O: IntervalIterator<E>,
		R: FromIterator<RangeInclusive<E>>,
	{
		DiffIter::from_iters(self.intervals(), other.intervals()).collect()
	}

	/// Returns whether `self` and `other` are disjoint sets
	fn disjoint<O: IntervalIterator<E> + ?Sized>(&self, other: &O) -> bool {
		let mut lhs = self.intervals().peekable();
		let mut rhs = other.intervals().peekable();
		while let (Some(l), Some(r)) = (lhs.peek(), rhs.peek()) {
			match overlap(l, r) {
				RangeOrdering::Less => {
					// Move to next "self range"
					let _ = lhs.next();
				}
				RangeOrdering::Overlap => return false,
				RangeOrdering::Greater => {
					// Move to next "other range"
					let _ = rhs.next();
				}
			}
		}
		true
	}

	/// Return the set intersection of two interval iterators.
	fn intersect<O, R>(&self, other: &O) -> R
	where
		E: Clone,
		O: IntervalIterator<E>,
		R: FromIterator<RangeInclusive<E>>,
	{
		IntersectIter::from_iters(self.intervals(), other.intervals()).collect()
	}

	/// Returns whether `self` is a subset of `other`
	fn subset<O: IntervalIterator<E> + ?Sized>(&self, other: &O) -> bool {
		let mut lhs = self.intervals().peekable();
		let mut rhs = other.intervals().peekable();
		while let (Some(l), Some(r)) = (lhs.peek(), rhs.peek()) {
			match overlap(l, r) {
				RangeOrdering::Overlap if r.start() <= l.start() && l.end() <= r.end() => {
					// Current "self range" is included in the current other range
					// Move to next "self range" that needs to be covered
					let _ = lhs.next();
				}
				RangeOrdering::Greater => {
					// Move to next "other range"
					let _ = rhs.next();
				}
				_ => {
					// Current "self range" can no longer be covered
					return false;
				}
			}
		}
		lhs.peek().is_none()
	}

	/// Returns whether `self` is a superset of `other`
	fn superset<O: IntervalIterator<E> + ?Sized>(&self, other: &O) -> bool {
		other.subset(self)
	}

	/// Return the set union of two interval iterators.
	fn union<O, R>(&self, other: &O) -> R
	where
		E: Clone,
		O: IntervalIterator<E>,
		R: FromIterator<RangeInclusive<E>>,
	{
		UnionIter::from_iters(self.intervals(), other.intervals()).collect()
	}
}

/// A sorted collection of inclusive ranges that can be used to represent
/// non-continuous sets of values.
///
/// # Warning
///
/// Although [`RangeList`] can be constructed for elements that do not implement
/// [`std::cmp::Ord`], but do implement [`std::cmp::PartialOrd`], constructor
/// methods, such as the [`FromIterator`] implementation, will panic if the used
/// boundary values cannot be sorted. This requirement allows the usage of types
/// like [`f64`], as long as the user can guarantee that values that cannot be
/// ordered, like `NaN`, will not appear.
#[derive(Default, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct RangeList<E: PartialOrd> {
	/// Memory representation of the ranges
	ranges: Vec<(E, E)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RangeOrdering {
	/// A compared range is strictly less than another.
	Less = -1,
	/// A compared range overlaps with another.
	Overlap = 0,
	/// A compared range is strictly greater than another.
	Greater = 1,
}

/// An iterator combinator that given two iterators yielding ordered ranges,
/// yields the ordered ranges that are in the union of the ranges yielded by the
/// iterators.
#[derive(Debug)]
pub struct UnionIter<
	E: PartialOrd,
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
> {
	/// Iterator yielding the ranges of the left-hand side of the union
	lhs: Peekable<I>,
	/// Iterator yielding the ranges of the right-hand side of the union
	rhs: Peekable<J>,
}

/// Returns the maximum of two values that implement PartialOrd
fn max<E: PartialOrd>(a: E, b: E) -> E {
	if a > b {
		a
	} else {
		b
	}
}

/// Returns the minimum of two values that implement PartialOrd
fn min<E: PartialOrd>(a: E, b: E) -> E {
	if a < b {
		a
	} else {
		b
	}
}

/// Returns whether two Ranges overlap
fn overlap<E: PartialOrd>(r1: &RangeInclusive<E>, r2: &RangeInclusive<E>) -> RangeOrdering {
	if r1.end() < r2.start() {
		RangeOrdering::Less
	} else if r2.end() < r1.start() {
		RangeOrdering::Greater
	} else {
		RangeOrdering::Overlap
	}
}

impl<E: Clone + Ord> IntervalIterator<E> for BTreeSet<E> {
	type IntervalIter = Map<<BTreeSet<E> as IntoIterator>::IntoIter, fn(E) -> RangeInclusive<E>>;

	fn intervals(&self) -> Self::IntervalIter {
		self.clone().into_iter().map(|e| e.clone()..=e)
	}
}

impl<E: Clone + Integer, I, J> DiffIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	/// Create a new [`DiffIter`] from two iterators yielding ordered ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		let mut lhs = lhs.peekable();
		Self {
			max: lhs
				.peek()
				.map(|r| Self::next_lower(r.start()))
				.unwrap_or_else(E::zero),
			lhs,
			rhs: rhs.peekable(),
		}
	}

	/// Create a new [`DiffIter`] from two set types that implement the [`IntervalIterator`] trait.
	pub fn new<A, B>(lhs: &A, rhs: &B) -> Self
	where
		A: IntervalIterator<E, IntervalIter = I>,
		B: IntervalIterator<E, IntervalIter = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}

	/// Returns the next higher value of `x`
	fn next_higher(x: &E) -> E {
		let mut v = x.clone();
		v.inc();
		v
	}

	/// Returns the next lower value of `x`
	fn next_lower(x: &E) -> E {
		let mut v = x.clone();
		v.dec();
		v
	}
}

impl<E: Clone + Integer, I, J> Iterator for DiffIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	type Item = RangeInclusive<E>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			// If LHS is empty, return None
			let _ = self.lhs.peek()?;

			let mut min = Self::next_higher(&self.max);
			self.max = self.lhs.peek().unwrap().end().clone();
			if min > *self.lhs.peek().unwrap().end() {
				let _ = self.lhs.next();
				if let Some(r) = self.lhs.peek() {
					min = r.start().clone();
					self.max = r.end().clone();
				} else {
					return None;
				}
			}
			while let Some(r) = self.rhs.peek() {
				if r.end() < self.lhs.peek().unwrap().start() {
					let _ = self.rhs.next();
				} else {
					break;
				}
			}
			if let Some(r) = self.rhs.peek() {
				if *r.start() <= self.max {
					// Interval min..max must be shurk
					if min >= *r.start() && self.max <= *r.end() {
						// Interval min..max is completely covered by r
						continue;
					}
					if *r.start() <= min {
						// Interval min..max overlaps on the left
						min = Self::next_higher(r.end());
						// Search for max
						let _ = self.rhs.next();
						if let Some(r) = self.rhs.peek() {
							if *r.start() <= self.max {
								self.max = Self::next_lower(r.start());
							}
						}
					} else {
						// Interval overlaps on the right
						self.max = Self::next_lower(r.start());
					}
				}
			}
			return Some(min..=self.max.clone());
		}
	}
}

impl<E: Clone + Ord> IntervalIterator<E> for HashSet<E> {
	type IntervalIter = Map<<Vec<E> as IntoIterator>::IntoIter, fn(E) -> RangeInclusive<E>>;

	fn intervals(&self) -> Self::IntervalIter {
		let mut v: Vec<_> = self.iter().cloned().collect();
		v.sort_unstable();
		v.into_iter().map(|e| e.clone()..=e)
	}
}

impl<E: Clone + PartialOrd, I, J> IntersectIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	/// Create a new [`IntersectIter`] from two iterators yielding ordered ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		Self {
			lhs: lhs.peekable(),
			rhs: rhs.peekable(),
		}
	}

	/// Create a new [`IntersectIter`] from two set types that implement the [`IntervalIterator`] trait.
	pub fn new<A, B>(lhs: &A, rhs: &B) -> Self
	where
		A: IntervalIterator<E, IntervalIter = I>,
		B: IntervalIterator<E, IntervalIter = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}
}

impl<E: PartialOrd + Clone, I, J> Iterator for IntersectIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	type Item = RangeInclusive<E>;

	fn next(&mut self) -> Option<Self::Item> {
		while let (Some(l), Some(r)) = (self.lhs.peek(), self.rhs.peek()) {
			match overlap(l, r) {
				RangeOrdering::Less => {
					let _ = self.lhs.next();
				}
				RangeOrdering::Greater => {
					let _ = self.rhs.next();
				}
				RangeOrdering::Overlap => {
					let v = max(l.start(), r.start()).clone()..=min(l.end(), r.end()).clone();
					if l.end() <= r.end() {
						let _ = self.lhs.next();
					} else {
						let _ = self.rhs.next();
					}
					return Some(v);
				}
			}
		}
		None
	}
}

impl<E: PartialOrd> RangeList<E> {
	/// Returns `true` if the range list contains no items.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// assert!(!RangeList::from_iter([3..=4]).is_empty());
	/// assert!(RangeList::<i64>::default().is_empty());
	/// assert!(RangeList::from_iter([3..=2]).is_empty());
	/// ```
	pub fn is_empty(&self) -> bool {
		self.ranges.is_empty()
	}

	/// Returns `true` if `item` is contained in the range list.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// assert!(RangeList::from_iter([1..=4]).contains(&4));
	/// assert!(!RangeList::from_iter([1..=4]).contains(&0));
	///
	/// assert!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).contains(&7));
	/// assert!(!RangeList::from_iter([1..=4, 6..=7, -5..=-3]).contains(&0));
	/// ```
	pub fn contains(&self, item: &E) -> bool {
		for r in self {
			if r.contains(&item) {
				return true;
			}
		}
		false
	}

	/// Returns the lower bound of the range list, or `None` if the range list is
	/// empty.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// assert_eq!(RangeList::from_iter([1..=4]).lower_bound(), Some(&1));
	/// assert_eq!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).lower_bound(), Some(&-5));
	///
	/// assert_eq!(RangeList::<i64>::default().lower_bound(), None);
	/// ```
	pub fn lower_bound(&self) -> Option<&E> {
		self.ranges.first().map(|(start, _)| start)
	}

	/// Returns the upper bound of the range list, or `None` if the range list is
	/// empty
	///
	/// # Examples
	///
	/// ```
	/// # use std::ops::RangeInclusive;
	/// # use rangelist::RangeList;
	/// assert_eq!(RangeList::from_iter([1..=4]).upper_bound(), Some(&4));
	/// assert_eq!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).upper_bound(), Some(&7));
	///
	/// assert_eq!(RangeList::<i64>::default().upper_bound(), None);
	/// ```
	pub fn upper_bound(&self) -> Option<&E> {
		self.ranges.last().map(|(_, end)| end)
	}
}

impl<E: PartialOrd + Copy> RangeList<E> {
	/// Returns an Copying iterator for the ranges in the set.
	#[allow(clippy::type_complexity)]
	pub fn iter<'a>(
		&'a self,
	) -> Map<
		<&RangeList<E> as IntoIterator>::IntoIter,
		fn(RangeInclusive<&'a E>) -> RangeInclusive<E>,
	> {
		self.into_iter().map(|r| **r.start()..=**r.end())
	}
}

impl<E: PartialOrd + Debug> Debug for RangeList<E> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		if self.ranges.is_empty() {
			return write!(f, "RangeList::default()");
		}
		if self.ranges.len() == 1 {
			return write!(
				f,
				"RangeList::from({:?}..={:?})",
				self.ranges[0].0, self.ranges[0].1
			);
		}
		write!(f, "RangeList::from_iter([")?;
		let mut first = true;
		for r in self {
			if !first {
				write!(f, ", ")?
			}
			write!(f, "{:?}", r)?;
			first = false;
		}
		write!(f, "])")
	}
}

impl<E: PartialOrd + Debug> Display for RangeList<E> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let mut first = true;
		for r in &self.ranges {
			if !first {
				write!(f, " union ")?;
			}
			write!(f, "{:?}..{:?}", r.0, r.1)?;
			first = false;
		}
		if first {
			write!(f, "1..0")?;
		}
		Ok(())
	}
}

impl<E: PartialOrd + Clone> From<&RangeInclusive<E>> for RangeList<E> {
	fn from(value: &RangeInclusive<E>) -> Self {
		if value.is_empty() {
			RangeList { ranges: Vec::new() }
		} else {
			Self {
				ranges: vec![(value.start().clone(), value.end().clone())],
			}
		}
	}
}

impl<E: PartialOrd + Clone> From<RangeInclusive<E>> for RangeList<E> {
	fn from(value: RangeInclusive<E>) -> Self {
		(&value).into()
	}
}

impl<E, R> FromIterator<R> for RangeList<E>
where
	E: PartialOrd + Clone,
	R: Into<RangeInclusive<E>>,
{
	fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
		let mut non_empty: Vec<(E, E)> = iter
			.into_iter()
			.filter_map(|r| {
				let r = r.into();
				if r.is_empty() {
					None
				} else {
					Some((r.start().clone(), r.end().clone()))
				}
			})
			.collect();
		if non_empty.is_empty() {
			return RangeList { ranges: Vec::new() };
		}
		non_empty.sort_by(|a, b| {
			a.0.partial_cmp(&b.0)
				.expect("the order of the bounds in the RangeList cannot be partial")
		});
		let mut it = non_empty.into_iter();
		let mut ranges = Vec::new();
		let mut cur = it.next().unwrap();
		for next in it {
			// Determine distance between the two ranges if integral
			let dist = match_type!((cur.1.clone(),next.0.clone()),
			{
				(i128,i128) as (ub, lb) => (lb - ub) as usize,
				(i64,i64) as (ub, lb) => (lb - ub) as usize,
				(i32,i32) as (ub, lb) => (lb - ub) as usize,
				(i16,i16) as (ub, lb) => (lb - ub) as usize,
				(i8,i8) as (ub, lb) => (lb - ub) as usize,
				(u128,u128) as (ub, lb) => (lb - ub) as usize,
				(u64,u64) as (ub, lb) => (lb - ub) as usize,
				(u32,u32) as (ub, lb) => (lb - ub) as usize,
				(u16,u16) as (ub, lb) => (lb - ub) as usize,
				(u8,u8) as (ub, lb) => (lb - ub) as usize,
				_ => usize::MAX,
			});

			if cur.1 >= next.0 || dist <= 1 {
				cur.1 = next.1
			} else {
				ranges.push(cur);
				cur = next;
			}
		}
		ranges.push(cur);
		Self { ranges }
	}
}

impl<E: PartialOrd + Clone> IntervalIterator<E> for RangeList<E> {
	type IntervalIter = <RangeList<E> as IntoIterator>::IntoIter;

	fn intervals(&self) -> Self::IntervalIter {
		self.clone().into_iter()
	}
}

impl<E: PartialOrd + Clone> IntoIterator for RangeList<E> {
	type IntoIter = Map<std::vec::IntoIter<(E, E)>, fn((E, E)) -> RangeInclusive<E>>;
	type Item = RangeInclusive<E>;

	fn into_iter(self) -> Self::IntoIter {
		self.ranges
			.into_iter()
			.map(|(start, end)| RangeInclusive::new(start, end))
	}
}

impl<'a, E: PartialOrd> IntoIterator for &'a RangeList<E> {
	type IntoIter = Map<std::slice::Iter<'a, (E, E)>, fn(&'a (E, E)) -> RangeInclusive<&'a E>>;
	type Item = RangeInclusive<&'a E>;

	fn into_iter(self) -> Self::IntoIter {
		self.ranges
			.iter()
			.map(|(start, end)| RangeInclusive::new(start, end))
	}
}

impl<E: Clone + PartialOrd, I, J> UnionIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	/// Create a new [`UnionIter`] from two iterators yielding ordered ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		Self {
			lhs: lhs.peekable(),
			rhs: rhs.peekable(),
		}
	}

	/// Create a new [`UnionIter`] from two set types that implement the [`IntervalIterator`] trait.
	pub fn new<A, B>(lhs: &A, rhs: &B) -> Self
	where
		A: IntervalIterator<E, IntervalIter = I>,
		B: IntervalIterator<E, IntervalIter = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}
}

impl<E: PartialOrd + Clone, I, J> Iterator for UnionIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	type Item = RangeInclusive<E>;

	fn next(&mut self) -> Option<Self::Item> {
		match (self.lhs.peek(), self.rhs.peek()) {
			(Some(l), None) => {
				let v = l.clone();
				let _ = self.lhs.next();
				Some(v)
			}
			(None, Some(r)) => {
				let v = r.clone();
				let _ = self.rhs.next();
				Some(v)
			}
			(Some(l), Some(r)) => match overlap(l, r) {
				RangeOrdering::Less => {
					let v = l.clone();
					let _ = self.lhs.next();
					Some(v)
				}
				RangeOrdering::Greater => {
					let v = r.clone();
					let _ = self.rhs.next();
					Some(v)
				}
				RangeOrdering::Overlap => {
					let mut ext = min(l.start(), r.start()).clone()..=max(l.end(), r.end()).clone();
					let _ = self.lhs.next();
					let _ = self.rhs.next();
					loop {
						if let Some(l) = self.lhs.peek() {
							if overlap(&ext, l) == RangeOrdering::Overlap {
								ext = ext.start().clone()..=max(ext.end(), l.end()).clone();
								let _ = self.lhs.next();
								continue;
							}
						}
						if let Some(r) = self.rhs.peek() {
							if overlap(&ext, r) == RangeOrdering::Overlap {
								ext = ext.start().clone()..=max(ext.end(), r.end()).clone();
								let _ = self.rhs.next();
								continue;
							}
						}
						break;
					}
					Some(ext)
				}
			},
			(None, None) => None,
		}
	}
}

#[cfg(test)]
mod tests {
	use expect_test::expect;

	use super::*;

	#[test]
	fn test_rangelist() {
		let empty: RangeList<i64> = RangeList::default();
		expect![[r#"
    RangeList::default()
"#]]
		.assert_debug_eq(&empty);
		assert!(empty.is_empty());

		let single_range = RangeList::from_iter([1..=4]);
		expect![[r#"
    RangeList::from(1..=4)
"#]]
		.assert_debug_eq(&single_range);
		assert!(!single_range.is_empty());
		assert!(single_range.contains(&1));
		assert!(single_range.contains(&2));
		assert!(single_range.contains(&4));
		assert!(!single_range.contains(&0));
		assert!(!single_range.contains(&5));

		let multi_range = RangeList::from_iter([1..=4, 6..=7, -5..=-3]);
		expect![[r#"
    RangeList::from_iter([-5..=-3, 1..=4, 6..=7])
"#]]
		.assert_debug_eq(&multi_range);
		assert!(multi_range.contains(&-5));
		assert!(multi_range.contains(&-3));
		assert!(multi_range.contains(&1));
		assert!(multi_range.contains(&4));
		assert!(multi_range.contains(&6));
		assert!(multi_range.contains(&7));
		assert!(!multi_range.contains(&0));
		assert!(!multi_range.contains(&5));
		assert!(!multi_range.contains(&-6));
		assert!(!multi_range.contains(&8));

		let collapse_range = RangeList::from_iter([1..=2, 2..=3, 10..=12, 11..=15]);
		expect![[r#"
    RangeList::from_iter([1..=3, 10..=15])
"#]]
		.assert_debug_eq(&collapse_range);

		let float_range = RangeList::from_iter([0.1..=3.2, 8.1..=11.2, 10.0..=50.0]);
		expect![[r#"
    RangeList::from_iter([0.1..=3.2, 8.1..=50.0])
"#]]
		.assert_debug_eq(&float_range);
	}

	#[test]
	fn test_display_rangelist() {
		let empty: RangeList<i64> = RangeList::default();
		assert_eq!(empty.to_string(), "1..0");

		let single_range = RangeList::from_iter([1..=4]);
		assert_eq!(single_range.to_string(), "1..4");

		let multi_range = RangeList::from_iter([1..=4, 6..=7, -5..=-3]);
		assert_eq!(multi_range.to_string(), "-5..-3 union 1..4 union 6..7");

		let float_range = RangeList::from_iter([0.1..=3.2, 8.1..=50.0]);
		assert_eq!(float_range.to_string(), "0.1..3.2 union 8.1..50.0");
	}

	#[test]
	fn test_set_diff() {
		let empty: RangeList<i64> = RangeList::default();
		// TODO: Can we change the diff implementation so we don't need the additional space.
		let inf: RangeList<i64> = RangeList::from_iter([i64::MIN + 1..=i64::MAX - 1]);
		let res: RangeList<_> = empty.diff(&empty);
		assert_eq!(res, empty);
		let res: RangeList<_> = inf.diff(&inf);
		assert_eq!(res, empty);
		let res: RangeList<_> = empty.diff(&inf);
		assert_eq!(res, empty);
		let res: RangeList<_> = inf.diff(&empty);
		assert_eq!(res, inf);

		let x = RangeList::from(1..=5);
		let y = RangeList::from(4..=9);
		let z: RangeList<_> = x.diff(&y);
		expect!["1..3"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.diff(&x);
		expect!["6..9"].assert_eq(&z.to_string());
		let z: RangeList<_> = x.diff(&x);
		expect!["1..0"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.diff(&y);
		expect!["1..0"].assert_eq(&z.to_string());
		let z: RangeList<_> = x.diff(&RangeList::from_iter([1..=2, 5..=5]));
		expect!["3..4"].assert_eq(&z.to_string());
		let z: RangeList<_> = x.diff(&RangeList::from(2..=4));
		expect!["1..1 union 5..5"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.diff(&RangeList::from_iter([5..=5, 7..=7, 9..=9]));
		expect!["4..4 union 6..6 union 8..8"].assert_eq(&z.to_string());

		let x = RangeList::from_iter([1..=3, 5..=7, 9..=11]);
		let z: RangeList<_> = x.diff(&y);
		expect!["1..3 union 10..11"].assert_eq(&z.to_string());
		let z: RangeList<_> = x.diff(&RangeList::from(-1..=8));
		expect!["9..11"].assert_eq(&z.to_string());
		let z: RangeList<_> = x.diff(&RangeList::from_iter([4..=4, 8..=8]));
		assert_eq!(x, z);
	}

	#[test]
	fn test_set_union() {
		let empty: RangeList<i64> = RangeList::default();
		let inf: RangeList<i64> = RangeList::from_iter([i64::MIN..=i64::MAX]);
		let res: RangeList<_> = empty.union(&empty);
		assert_eq!(res, empty);
		let res: RangeList<_> = inf.union(&inf);
		assert_eq!(res, inf);
		let res: RangeList<_> = empty.union(&inf);
		assert_eq!(res, inf);
		let res: RangeList<_> = inf.union(&empty);
		assert_eq!(res, inf);

		let x = RangeList::from(1..=5);
		let y = RangeList::from(4..=9);
		let z: RangeList<_> = x.union(&y);
		expect!["1..9"].assert_eq(&z.to_string());

		let y = RangeList::from_iter([1..=2, 4..=4]);
		let z: RangeList<_> = x.union(&y);
		expect!["1..5"].assert_eq(&z.to_string());

		let y = RangeList::from_iter([-5..=-1, 6..=9]);
		let z: RangeList<_> = x.union(&y);
		expect!["-5..-1 union 1..9"].assert_eq(&z.to_string());

		let z: RangeList<_> = y.union(&x);
		expect!["-5..-1 union 1..9"].assert_eq(&z.to_string());

		let x = RangeList::from(1..=9);
		let y = RangeList::from_iter([1..=2, 4..=5, 7..=8]);
		let z: RangeList<_> = x.union(&y);
		expect!["1..9"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.union(&x);
		expect!["1..9"].assert_eq(&z.to_string());

		let x = RangeList::from(1.0..=5.0);
		let y = RangeList::from(4.0..=9.0);
		let z: RangeList<_> = x.union(&y);
		expect!["1.0..9.0"].assert_eq(&z.to_string());
	}

	#[test]
	fn test_set_intersect() {
		let empty = RangeList::default();
		let inf = RangeList::from_iter([i64::MIN..=i64::MAX]);
		let res: RangeList<_> = empty.intersect(&empty);
		assert_eq!(res, empty);
		let res: RangeList<_> = inf.intersect(&inf);
		assert_eq!(res, inf);
		let res: RangeList<_> = empty.intersect(&inf);
		assert_eq!(res, empty);
		let res: RangeList<_> = inf.intersect(&empty);
		assert_eq!(res, empty);

		let x = RangeList::from(1..=5);
		let y = RangeList::from(4..=9);
		let z: RangeList<_> = x.intersect(&y);
		expect!["4..5"].assert_eq(&z.to_string());

		let y = RangeList::from_iter([1..=2, 4..=9]);
		let z: RangeList<_> = x.intersect(&y);
		expect!["1..2 union 4..5"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.intersect(&x);
		expect!["1..2 union 4..5"].assert_eq(&z.to_string());

		let y = RangeList::from_iter([-5..=-1, 1..=3]);
		let z: RangeList<_> = x.intersect(&y);
		expect!["1..3"].assert_eq(&z.to_string());
		let z: RangeList<_> = y.intersect(&x);
		expect!["1..3"].assert_eq(&z.to_string());

		let x = RangeList::from(1.0..=5.0);
		let y = RangeList::from(4.0..=9.0);
		let z: RangeList<_> = x.intersect(&y);
		expect!["4.0..5.0"].assert_eq(&z.to_string());
	}

	#[test]
	fn test_set_subset() {
		let empty = RangeList::default();
		let inf = RangeList::from(i64::MIN..=i64::MAX);
		assert!(empty.subset(&inf));
		assert!(!inf.subset(&empty));

		let x = RangeList::from(1..=5);
		let y = RangeList::from(1..=9);
		assert!(x.subset(&x));
		assert!(x.subset(&y));
		assert!(!y.subset(&x));
		assert!(y.subset(&y));

		let x = RangeList::from_iter([1..=2, 4..=9]);
		assert!(x.subset(&x));
		assert!(x.subset(&y));

		let x = RangeList::from(1.0..=5.0);
		let y = RangeList::from(1.0..=9.0);
		assert!(x.subset(&x));
		assert!(x.subset(&y));
		assert!(!y.subset(&x));
		assert!(y.subset(&y));
	}

	#[test]
	fn test_set_disjoint() {
		let empty = RangeList::default();
		let inf = RangeList::from(i64::MIN..=i64::MAX);

		assert!(empty.disjoint(&empty));
		assert!(empty.disjoint(&inf));
		assert!(inf.disjoint(&empty));
		assert!(!inf.disjoint(&inf));

		let x = RangeList::from_iter([1..=2, 4..=6, 8..=9]);
		assert!(empty.disjoint(&x));
		assert!(x.disjoint(&empty));
		assert!(!x.disjoint(&x));
		assert!(!inf.disjoint(&x));
		assert!(!x.disjoint(&inf));

		let x = RangeList::from_iter([1.0..=2.0, 5.0..=6.0]);
		let y = RangeList::from_iter([3.0..=4.0, 7.0..=8.0]);
		assert!(x.disjoint(&y));
		assert!(y.disjoint(&x));
	}

	#[test]
	fn test_set_card() {
		let empty = RangeList::<i64>::default();
		assert_eq!(empty.card(), 0);

		let x = RangeList::<i8>::from(1..=5);
		assert_eq!(x.card(), 5);

		let y = RangeList::<u32>::from_iter([1..=2, 4..=6, 8..=9]);
		assert_eq!(y.card(), 7);
	}
}
