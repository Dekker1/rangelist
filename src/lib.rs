//! A library for working with set of values represented as inclusive ranges.
//!
//! This library provides a [`RangeList`] struct that can be used to represent
//! sets of values as a collection of inclusive ranges. The ranges are stored
//! in a deduplicated sorted order.
//!
//! Additionally, the library defines [`IntervalIterator`] trait to be
//! implemented by types can provide an iterator of sorted inclusive
//! ranges. Any combination of types that implement this trait can be used
//! to perform standard set operations such as union and intersection.
//!
//! Finally, the library provides [`DiffIter`], [`IntersectIter`], and
//! [`UnionIter`], which are lazy iterator combinators that can be used to
//! perform set operations on two iterators of ordered ranges.

mod num_traits;

use std::{
	collections::{BTreeSet, HashSet},
	fmt::{Debug, Display},
	hash::Hash,
	iter::{Fuse, Map, Peekable},
	ops::{Bound, RangeInclusive},
};

pub use num_traits::{Adjacent, Step};

/// An iterator combinator that given two iterators yielding ordered ranges,
/// yields the ordered ranges of elements that are in the ranges yielded by
/// `lhs` iterator, but does not include elements that are in the ranges yielded
/// by the `rhs` iterator.
#[derive(Debug)]
pub struct DiffIter<
	E: Clone + Adjacent + PartialOrd,
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
> {
	/// Iterator yielding the ranges of the elements that we want to include.
	lhs: Peekable<I>,
	/// Iterator yielding the ranges of the elements that must be excluded.
	rhs: Peekable<J>,
	/// Value to use as the start of the next LHS range, because it was already
	/// partially yielded.
	next_min: Option<E>,
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

/// A trait that provides operations on iterators of ordered intervals.
pub trait IntervalIterator<E: PartialOrd> {
	/// The type of the interval iterator.
	type IntervalIter<'a>: Iterator<Item = RangeInclusive<E>>
	where
		Self: 'a;

	/// Returns the number of elements contained within the RangeList.
	///
	/// Returns `None` if the number of elements would overflow `usize`.
	fn card(&self) -> Option<usize>
	where
		E: Step,
	{
		let mut card: usize = 0;
		for r in self.intervals() {
			let c = Step::steps_between(r.start(), r.end())?;
			card = card.checked_add(c)?.checked_add(1)?
		}
		Some(card)
	}

	/// Returns `true` if `elem` is contained in the range list.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::{RangeList, IntervalIterator};
	/// assert!(RangeList::from_iter([1..=4]).contains(&4));
	/// assert!(!RangeList::from_iter([1..=4]).contains(&0));
	///
	/// assert!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).contains(&7));
	/// assert!(!RangeList::from_iter([1..=4, 6..=7, -5..=-3]).contains(&0));
	/// ```
	fn contains(&self, elem: &E) -> bool {
		self.intervals().any(|r| r.contains(elem))
	}

	/// Compute RangeList without any of the elements in the ranges of `other`.
	///
	/// # Warning
	///
	/// To cut a range of `self`, the implementation takes the predecessor of
	/// the start, or the successor of the end, of a range of `other`. This
	/// panics if that value does not exist in `E`. For example, for floating
	/// point types `f64::MAX` has no successor, although `f64::INFINITY` is a
	/// larger value.
	fn diff<O, R>(&self, other: &O) -> R
	where
		E: Clone + Adjacent,
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
	/// Returns an iterator over the ordered intervals.
	fn intervals(&self) -> Self::IntervalIter<'_>;

	/// Returns whether `self` is a subset of `other`
	fn subset<O: IntervalIterator<E> + ?Sized>(&self, other: &O) -> bool {
		let mut lhs = self.intervals().peekable();
		let mut rhs = other.intervals().peekable();
		while let (Some(l), Some(r)) = (lhs.peek(), rhs.peek()) {
			match overlap(l, r) {
				RangeOrdering::Overlap if r.start() <= l.start() && l.end() <= r.end() => {
					// Current "self range" is included in the current other
					// range Move to next "self range" that
					// needs to be covered
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
		O: IntervalIterator<E>,
		R: FromIterator<RangeInclusive<E>>,
	{
		UnionIter::from_iters(self.intervals(), other.intervals()).collect()
	}
}

/// An iterator over clones of the ranges of a [`RangeList`], as returned by
/// [`IntervalIterator::intervals`].
#[derive(Debug, Clone)]
pub struct Intervals<'a, E> {
	/// Iterator over the stored ranges
	iter: std::slice::Iter<'a, (E, E)>,
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
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct RangeList<E: PartialOrd> {
	/// Memory representation of the ranges
	ranges: Vec<(E, E)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// An `RangeOrdering` is the result of a comparison between two ranges of
/// values.
enum RangeOrdering {
	/// A left range is strictly less than the right range.
	Less,
	/// A compared ranges overlap with each other.
	Overlap,
	/// A left range is strictly greater than the right range.
	Greater,
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
	lhs: Peekable<Fuse<I>>,
	/// Iterator yielding the ranges of the right-hand side of the union
	rhs: Peekable<Fuse<J>>,
}

/// Extends `cur` to cover `next` if `next`, which must not start before `cur`,
/// overlaps with or is adjacent to it. Returns whether `next` was absorbed.
fn absorb<E: Adjacent + PartialOrd>(cur: &mut (E, E), next: &mut (E, E)) -> bool {
	let merge = cur.1 >= next.0 || cur.1.successor().is_some_and(|succ| next.0 <= succ);
	// `next` may be fully contained in `cur`, so keep the larger end.
	if merge && next.1 > cur.1 {
		std::mem::swap(&mut cur.1, &mut next.1);
	}
	merge
}

/// Returns the maximum of two values that implement PartialOrd
fn max<E: PartialOrd>(a: E, b: E) -> E {
	if a > b { a } else { b }
}

/// Returns the minimum of two values that implement PartialOrd
fn min<E: PartialOrd>(a: E, b: E) -> E {
	if a < b { a } else { b }
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
	type IntervalIter<'a>
		= Map<std::collections::btree_set::Iter<'a, E>, fn(&'a E) -> RangeInclusive<E>>
	where
		Self: 'a;

	fn card(&self) -> Option<usize>
	where
		E: Step,
	{
		Some(self.len())
	}

	fn contains(&self, elem: &E) -> bool {
		BTreeSet::contains(self, elem)
	}

	fn intervals(&self) -> Self::IntervalIter<'_> {
		self.iter().map(|e| e.clone()..=e.clone())
	}
}

impl<E: Clone + Adjacent + PartialOrd, I, J> DiffIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	/// Create a new [`DiffIter`] from two iterators yielding ordered ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		Self {
			next_min: None,
			lhs: lhs.peekable(),
			rhs: rhs.peekable(),
		}
	}

	/// Create a new [`DiffIter`] from two set types that implement the
	/// [`IntervalIterator`] trait.
	pub fn new<'a, A, B>(lhs: &'a A, rhs: &'a B) -> Self
	where
		A: IntervalIterator<E, IntervalIter<'a> = I>,
		B: IntervalIterator<E, IntervalIter<'a> = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}
}

impl<E: Clone + Adjacent + PartialOrd, I, J> Iterator for DiffIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	type Item = RangeInclusive<E>;

	fn next(&mut self) -> Option<Self::Item> {
		let mut lhs = self.lhs.peek()?.clone();
		if let Some(min) = self.next_min.take() {
			lhs = min..=lhs.end().clone();
		}
		loop {
			let Some(rhs) = self.rhs.peek() else {
				let _ = self.lhs.next().unwrap();
				return Some(lhs);
			};
			match overlap(&lhs, rhs) {
				// LHS range is strictly smaller than RHS range. Keep RHS range and
				// yield the LHS range.
				RangeOrdering::Less => {
					let _ = self.lhs.next().unwrap();
					return Some(lhs);
				}
				RangeOrdering::Overlap => {
					match (rhs.start() <= lhs.start(), rhs.end() >= lhs.end()) {
						// RHS fully removes the LHS range, proceed to the next LHS range
						(true, true) => {
							let _ = self.lhs.next().unwrap();
							lhs = self.lhs.peek()?.clone();
						}
						// RHS removes the beginning of the LHS range, cut LHS and proceed
						// to the next RHS range.
						(true, false) => {
							lhs = rhs.end().successor().unwrap()..=lhs.end().clone();
							let _ = self.rhs.next();
						}
						// RHS removes the end of the LHS range, emit cut LHS (and keep the
						// RHS range).
						(false, true) => {
							let _ = self.lhs.next().unwrap();
							return Some(lhs.start().clone()..=rhs.start().predecessor().unwrap());
						}
						// RHS removes a middle part of the LHS, emit cut LHS, keep its
						// remainder, and proceed to the next RHS range.
						(false, false) => {
							let lhs_cut = lhs.start().clone()..=rhs.start().predecessor().unwrap();
							self.next_min = Some(rhs.end().successor().unwrap());
							let _ = self.rhs.next();
							return Some(lhs_cut);
						}
					}
				}
				// LHS range is strictly greater than the RHS range, proceed to the next
				// RHS range
				RangeOrdering::Greater => {
					let _ = self.rhs.next();
				}
			}
		}
	}
}

impl<E: Clone + Hash + Ord> IntervalIterator<E> for HashSet<E> {
	type IntervalIter<'a>
		= Map<<Vec<E> as IntoIterator>::IntoIter, fn(E) -> RangeInclusive<E>>
	where
		Self: 'a;

	fn card(&self) -> Option<usize>
	where
		E: Step,
	{
		Some(self.len())
	}

	fn contains(&self, elem: &E) -> bool {
		HashSet::contains(self, elem)
	}

	fn intervals(&self) -> Self::IntervalIter<'_> {
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
	/// Create a new [`IntersectIter`] from two iterators yielding ordered
	/// ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		Self {
			lhs: lhs.peekable(),
			rhs: rhs.peekable(),
		}
	}

	/// Create a new [`IntersectIter`] from two set types that implement the
	/// [`IntervalIterator`] trait.
	pub fn new<'a, A, B>(lhs: &'a A, rhs: &'a B) -> Self
	where
		A: IntervalIterator<E, IntervalIter<'a> = I>,
		B: IntervalIterator<E, IntervalIter<'a> = J>,
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

impl<E: Clone> Iterator for Intervals<'_, E> {
	type Item = RangeInclusive<E>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.iter
			.next()
			.map(|(start, end)| start.clone()..=end.clone())
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.iter.size_hint()
	}
}

impl<E: PartialOrd> RangeList<E> {
	/// Returns `true` if `elem` is contained in the range list.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let rl = RangeList::from_iter([1..=4, 6..=7]);
	/// assert!(rl.contains(&6));
	/// assert!(!rl.contains(&5));
	/// ```
	pub fn contains(&self, elem: &E) -> bool {
		// Fast path for elements outside of the bounds of the range list.
		if self.min().is_none_or(|min| elem < min) || self.max().is_none_or(|max| elem > max) {
			return false;
		}
		let i = self.ranges.partition_point(|(_, end)| end < elem);
		self.ranges.get(i).is_some_and(|(start, _)| start <= elem)
	}

	/// Returns the number of elements in the range list, or `None` if it would
	/// overflow `usize`.
	fn count(&self) -> Option<usize>
	where
		E: Step,
	{
		// A dedicated loop, as it is measurably faster than `count_below`.
		self.ranges.iter().try_fold(0_usize, |count, (start, end)| {
			count
				.checked_add(Step::steps_between(start, end)?)?
				.checked_add(1)
		})
	}

	/// Returns the number of elements that are smaller than `elem`, or smaller
	/// than or equal to `elem` if `inclusive` is set.
	///
	/// Returns `None` if the count would overflow `usize`.
	fn count_below(&self, elem: &E, inclusive: bool) -> Option<usize>
	where
		E: Step,
	{
		let (count, found) = self.locate(elem)?;
		count.checked_add(usize::from(found && inclusive))
	}

	/// Returns the position of the gap before the smallest element greater than
	/// (or equal to) the given bound, following the model of
	/// [`BTreeMap::lower_bound`](std::collections::BTreeMap::lower_bound)
	/// (currently unstable).
	///
	/// The position of a gap is the number of elements before it. As such, the
	/// result is the [`Self::position`] of the smallest element that satisfies
	/// the bound, or the number of elements in the range list if no such
	/// element exists.
	///
	/// Passing `Bound::Included(x)` returns the number of elements smaller than
	/// `x`, `Bound::Excluded(x)` returns the number of elements smaller than or
	/// equal to `x`, and `Bound::Unbounded` returns `0`.
	///
	/// Returns `None` if the position would overflow `usize`.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// # use std::ops::Bound;
	/// let rl = RangeList::from_iter([1..=4, 6..=8]);
	/// assert_eq!(rl.first_position_bound(&Bound::Unbounded), Some(0));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(-1)), Some(0));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(1)), Some(0));
	/// assert_eq!(rl.first_position_bound(&Bound::Excluded(1)), Some(1));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(4)), Some(3));
	/// assert_eq!(rl.first_position_bound(&Bound::Excluded(4)), Some(4));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(5)), Some(4));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(8)), Some(6));
	/// assert_eq!(rl.first_position_bound(&Bound::Excluded(8)), Some(7));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(9)), Some(7));
	/// ```
	pub fn first_position_bound(&self, bound: &Bound<E>) -> Option<usize>
	where
		E: Step,
	{
		match bound {
			Bound::Included(x) => self.count_below(x, false),
			Bound::Excluded(x) => self.count_below(x, true),
			Bound::Unbounded => Some(0),
		}
	}

	/// Construct a [`RangeList`] from an iterator of elements in any order,
	/// collapsing them into the minimal set of inclusive ranges.
	///
	/// # Warning
	///
	/// This function will panic if any of the elements cannot be sorted (e.g.,
	/// `NaN` for floating point types).
	pub fn from_elements<T: IntoIterator<Item = E>>(iter: T) -> Self
	where
		E: Adjacent + Clone,
	{
		let mut elems: Vec<E> = iter.into_iter().collect();
		elems.sort_by(|a, b| {
			a.partial_cmp(b)
				.expect("the order of the elements in the RangeList cannot be partial")
		});
		Self::from_sorted_elements(elems)
	}

	/// Construct a [`RangeList`] from an iterator of elements that are known to
	/// be yielded in sorted (increasing) order, but where duplicates might
	/// still exist.
	///
	/// # Warning
	///
	/// This function will panic if the iterator yields a smaller element than
	/// one yielded previously.
	pub fn from_sorted_elements<T: IntoIterator<Item = E>>(iter: T) -> Self
	where
		E: Adjacent + Clone,
	{
		let mut it = iter.into_iter();
		let mut ranges = Vec::new();
		let Some(mut start) = it.next() else {
			return Self::default();
		};
		let mut end = start.clone();
		for next in it {
			if next < end {
				panic!("elements must be yielded in sorted order");
			}
			if next == end {
				continue;
			}

			if end.successor().unwrap() == next {
				end = next;
			} else {
				ranges.push((start, end));
				start = next.clone();
				end = next;
			}
		}
		ranges.push((start, end));
		Self { ranges }
	}

	/// Construct a [`RangeList`] from an iterator of inclusive ranges that are
	/// known to be yielded in sorted (increasing) order, but where ranges might
	/// still need to be merged.
	///
	/// # Warning
	///
	/// This function will panic if the iterator yields a range that starts
	/// before the (merged) range that precedes it.
	pub fn from_sorted_ranges<T: IntoIterator<Item = RangeInclusive<E>>>(iter: T) -> Self
	where
		E: Adjacent,
	{
		let mut it = iter
			.into_iter()
			.filter(|r| !r.is_empty())
			.map(RangeInclusive::into_inner);
		let mut ranges = Vec::new();
		let Some(mut cur) = it.next() else {
			return Self::default();
		};
		for mut next in it {
			if next.0 < cur.0 {
				panic!("ranges must be yielded in sorted order");
			}
			if !absorb(&mut cur, &mut next) {
				ranges.push(std::mem::replace(&mut cur, next));
			}
		}
		ranges.push(cur);
		Self { ranges }
	}

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

	/// Returns an Copying iterator for the ranges in the set.
	#[allow(
		clippy::type_complexity,
		reason = "type is less understandable if split up"
	)]
	pub fn iter<'a>(
		&'a self,
	) -> Map<
		<&'a RangeList<E> as IntoIterator>::IntoIter,
		fn(RangeInclusive<&'a E>) -> RangeInclusive<E>,
	>
	where
		E: Copy,
	{
		self.into_iter().map(|r| **r.start()..=**r.end())
	}

	/// Returns the position of the gap after the largest element smaller than
	/// (or equal to) the given bound, following the model of
	/// [`BTreeMap::upper_bound`](std::collections::BTreeMap::upper_bound)
	/// (currently unstable).
	///
	/// The position of a gap is the number of elements before it. As such, the
	/// result is one more than the [`Self::position`] of the largest element
	/// that satisfies the bound, or `0` if no such element exists. The elements
	/// between two bounds are at the positions
	/// `first_position_bound(lo)..last_position_bound(hi)`.
	///
	/// Passing `Bound::Included(x)` returns the number of elements smaller than
	/// or equal to `x`, `Bound::Excluded(x)` returns the number of elements
	/// smaller than `x`, and `Bound::Unbounded` returns the number of elements
	/// in the range list.
	///
	/// Returns `None` if the position would overflow `usize`.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// # use std::ops::Bound;
	/// let rl = RangeList::from_iter([1..=4, 6..=8]);
	/// assert_eq!(rl.last_position_bound(&Bound::Unbounded), Some(7));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(-1)), Some(0));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(1)), Some(0));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(1)), Some(1));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(4)), Some(3));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(4)), Some(4));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(5)), Some(4));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(6)), Some(4));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(9)), Some(7));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(9)), Some(7));
	/// ```
	pub fn last_position_bound(&self, bound: &Bound<E>) -> Option<usize>
	where
		E: Step,
	{
		match bound {
			Bound::Included(x) => self.count_below(x, true),
			Bound::Excluded(x) => self.count_below(x, false),
			Bound::Unbounded => self.count(),
		}
	}

	/// Returns the number of elements that are smaller than `elem`, and
	/// whether `elem` is contained in the range list.
	///
	/// Returns `None` if the count would overflow `usize`.
	fn locate(&self, elem: &E) -> Option<(usize, bool)>
	where
		E: Step,
	{
		let mut count: usize = 0;
		for (start, end) in &self.ranges {
			if elem < start {
				break;
			}
			if elem <= end {
				let count = count.checked_add(Step::steps_between(start, elem)?)?;
				return Some((count, true));
			}
			count = count
				.checked_add(Step::steps_between(start, end)?)?
				.checked_add(1)?;
		}
		Some((count, false))
	}

	/// Returns the lower bound of the range list, or `None` if the range list
	/// is empty.
	#[deprecated(since = "0.5.0", note = "use `min` instead")]
	pub fn lower_bound(&self) -> Option<&E> {
		self.min()
	}

	/// Returns the maximum element of the range list, or `None` if the range
	/// list is empty.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// assert_eq!(RangeList::from_iter([1..=4]).max(), Some(&4));
	/// assert_eq!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).max(), Some(&7));
	///
	/// assert_eq!(RangeList::<i64>::default().max(), None);
	/// ```
	pub fn max(&self) -> Option<&E> {
		self.ranges.last().map(|(_, end)| end)
	}

	/// Returns the minimum element of the range list, or `None` if the range
	/// list is empty.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// assert_eq!(RangeList::from_iter([1..=4]).min(), Some(&1));
	/// assert_eq!(RangeList::from_iter([1..=4, 6..=7, -5..=-3]).min(), Some(&-5));
	///
	/// assert_eq!(RangeList::<i64>::default().min(), None);
	/// ```
	pub fn min(&self) -> Option<&E> {
		self.ranges.first().map(|(start, _)| start)
	}

	/// Returns how many elements precede the given element in the RangeList, or
	/// `None` if the element does not occur in the RangeList.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let rl = RangeList::from_iter([1..=4, 6..=8]);
	/// assert_eq!(rl.position(&1), Some(0));
	/// assert_eq!(rl.position(&4), Some(3));
	/// assert_eq!(rl.position(&6), Some(4));
	/// assert_eq!(rl.position(&7), Some(5));
	/// assert_eq!(rl.position(&-4), None);
	/// ```
	pub fn position(&self, elem: &E) -> Option<usize>
	where
		E: Step,
	{
		let (pos, found) = self.locate(elem)?;
		found.then_some(pos)
	}

	/// Tightens the lower bound of the range list, removing any (partial)
	/// ranges that are below the new lower bound.
	#[deprecated(since = "0.5.0", note = "use `tighten_min` instead")]
	pub fn set_lower_bound(&mut self, lower_bound: E) {
		self.tighten_min(lower_bound)
	}

	/// Tightens the upper bound of the range list, removing any (partial)
	/// ranges that are above the new upper bound.
	#[deprecated(since = "0.5.0", note = "use `tighten_max` instead")]
	pub fn set_upper_bound(&mut self, upper_bound: E) {
		self.tighten_max(upper_bound)
	}

	/// Tightens the maximum of the range list, removing any (partial) ranges
	/// that are above the new maximum.
	///
	/// Note that no action is taken if `max` is greater than or equal to the
	/// current maximum.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let mut r = RangeList::from_iter([-5..=-3, 1..=4, 6..=7]);
	/// r.tighten_max(3);
	/// assert_eq!(r.max(), Some(&3));
	/// assert_eq!(r.iter().collect::<Vec<_>>(), vec![-5..=-3, 1..=3]);
	/// ```
	pub fn tighten_max(&mut self, max: E) {
		let last_kept = self.ranges.iter().rposition(|(start, _)| *start <= max);
		if let Some(end) = last_kept {
			self.ranges.truncate(end + 1);
			let last = self.ranges.last_mut().unwrap();
			if last.1 > max {
				last.1 = max;
			}
		} else {
			self.ranges = Vec::new();
		}
	}

	/// Tightens the minimum of the range list, removing any (partial) ranges
	/// that are below the new minimum.
	///
	/// Note that no action is taken if `min` is less than or equal to the
	/// current minimum.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let mut r = RangeList::from_iter([-5..=-3, 1..=4, 6..=7]);
	/// r.tighten_min(2);
	/// assert_eq!(r.min(), Some(&2));
	/// assert_eq!(r.iter().collect::<Vec<_>>(), vec![2..=4, 6..=7]);
	/// ```
	pub fn tighten_min(&mut self, min: E) {
		let first_kept = self.ranges.iter().position(|(_, end)| *end >= min);
		if let Some(start) = first_kept {
			let _ = self.ranges.drain(..start);
			if self.ranges[0].0 < min {
				self.ranges[0].0 = min;
			}
		} else {
			self.ranges = Vec::new();
		}
	}

	/// Returns the upper bound of the range list, or `None` if the range list
	/// is empty.
	#[deprecated(since = "0.5.0", note = "use `max` instead")]
	pub fn upper_bound(&self) -> Option<&E> {
		self.max()
	}
}

impl<E: Debug + PartialOrd> Debug for RangeList<E> {
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

impl<E: PartialOrd> Default for RangeList<E> {
	fn default() -> Self {
		Self {
			ranges: Default::default(),
		}
	}
}

impl<E: Debug + PartialOrd> Display for RangeList<E> {
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

impl<E: Clone + PartialOrd> From<&RangeInclusive<E>> for RangeList<E> {
	fn from(value: &RangeInclusive<E>) -> Self {
		value.clone().into()
	}
}

impl<E: PartialOrd> From<RangeInclusive<E>> for RangeList<E> {
	fn from(value: RangeInclusive<E>) -> Self {
		if value.is_empty() {
			Self::default()
		} else {
			Self {
				ranges: vec![value.into_inner()],
			}
		}
	}
}

impl<E, R> FromIterator<R> for RangeList<E>
where
	E: Adjacent + PartialOrd,
	R: Into<RangeInclusive<E>>,
{
	fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
		let iter = iter
			.into_iter()
			.map(Into::<RangeInclusive<E>>::into)
			.filter(|r| !r.is_empty())
			.map(RangeInclusive::into_inner);
		// Merge overlapping and adjacent ranges while the input is in order.
		let mut ranges: Vec<(E, E)> = Vec::new();
		let mut sorted = true;
		for mut next in iter {
			match ranges.last_mut() {
				Some(cur) if sorted && next.0 >= cur.0 => {
					if !absorb(cur, &mut next) {
						ranges.push(next);
					}
				}
				Some(_) => {
					sorted = false;
					ranges.push(next);
				}
				None => ranges.push(next),
			}
		}
		if !sorted {
			ranges.sort_unstable_by(|a, b| {
				a.0.partial_cmp(&b.0)
					.expect("the order of the bounds in the RangeList cannot be partial")
			});
			ranges.dedup_by(|next, cur| absorb(cur, next));
		}
		Self { ranges }
	}
}

impl<E: PartialOrd + Clone> IntervalIterator<E> for RangeList<E> {
	type IntervalIter<'a>
		= Intervals<'a, E>
	where
		Self: 'a;

	fn card(&self) -> Option<usize>
	where
		E: Step,
	{
		self.count()
	}

	fn contains(&self, elem: &E) -> bool {
		RangeList::contains(self, elem)
	}

	fn intervals(&self) -> Self::IntervalIter<'_> {
		Intervals {
			iter: self.ranges.iter(),
		}
	}
}

impl<E: PartialOrd> IntoIterator for RangeList<E> {
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

impl<E: PartialOrd, I, J> UnionIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	/// Create a new [`UnionIter`] from two iterators yielding ordered ranges.
	pub fn from_iters(lhs: I, rhs: J) -> Self {
		Self {
			lhs: lhs.fuse().peekable(),
			rhs: rhs.fuse().peekable(),
		}
	}

	/// Create a new [`UnionIter`] from two set types that implement the
	/// [`IntervalIterator`] trait.
	pub fn new<'a, A, B>(lhs: &'a A, rhs: &'a B) -> Self
	where
		A: IntervalIterator<E, IntervalIter<'a> = I>,
		B: IntervalIterator<E, IntervalIter<'a> = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}
}

impl<E: PartialOrd, I, J> Iterator for UnionIter<E, I, J>
where
	I: Iterator<Item = RangeInclusive<E>>,
	J: Iterator<Item = RangeInclusive<E>>,
{
	type Item = RangeInclusive<E>;

	fn next(&mut self) -> Option<Self::Item> {
		let (Some(l), Some(r)) = (self.lhs.peek(), self.rhs.peek()) else {
			// At most one side has ranges left, which can be yielded as is.
			return self.lhs.next().or_else(|| self.rhs.next());
		};
		match overlap(l, r) {
			RangeOrdering::Less => self.lhs.next(),
			RangeOrdering::Greater => self.rhs.next(),
			RangeOrdering::Overlap => {
				let (l_start, l_end) = self.lhs.next()?.into_inner();
				let (r_start, r_end) = self.rhs.next()?.into_inner();
				let start = min(l_start, r_start);
				let mut end = max(l_end, r_end);
				// The ranges are ordered, so a following range overlaps if it
				// starts at or before `end`.
				while let Some(next) = self
					.lhs
					.next_if(|r| *r.start() <= end)
					.or_else(|| self.rhs.next_if(|r| *r.start() <= end))
				{
					end = max(end, next.into_inner().1);
				}
				Some(start..=end)
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use expect_test::expect;

	use super::*;

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
	fn test_from_elements() {
		// Unsorted elements (with duplicates) should be collapsed into ranges.
		let elems = [5_u32, 1, 4, 2, 1, 6];
		let rl = RangeList::from_elements(elems);
		let expected = RangeList::from_iter([1_u32..=2, 4..=6]);
		assert_eq!(rl, expected);

		// Signed values are handled correctly.
		let rl2 = RangeList::from_elements([3_i64, -2, -1, 3, 0]);
		let expected2 = RangeList::from_iter([-2_i64..=0, 3..=3]);
		assert_eq!(rl2, expected2);

		// Empty iterator yields empty/default RangeList.
		let rl_empty: RangeList<u32> = RangeList::from_elements([]);
		assert!(rl_empty.is_empty());
	}

	#[test]
	fn test_from_sorted_elements() {
		// Sorted elements (with duplicates) should be collapsed into ranges.
		let elems = [1_u32, 1, 2, 4, 5, 6];
		let rl = RangeList::from_sorted_elements(elems);
		let expected = RangeList::from_iter([1_u32..=2, 4..=6]);
		assert_eq!(rl, expected);

		// Single element produces a single 1-length range.
		let rl2 = RangeList::from_sorted_elements([10_u8]);
		let expected2 = RangeList::from_iter([10_u8..=10]);
		assert_eq!(rl2, expected2);

		// Empty iterator yields empty/default RangeList.
		let rl_empty: RangeList<u32> = RangeList::from_sorted_elements([]);
		assert!(rl_empty.is_empty());
	}

	#[test]
	fn test_from_sorted_ranges() {
		// Sorted ranges with overlap and adjacency should be merged.
		let rl = RangeList::from_sorted_ranges([1..=2, 2..=4, 6..=7]);
		let expected = RangeList::from_iter([1..=4, 6..=7]);
		assert_eq!(rl, expected);

		// Also works for signed types and preserves order.
		let rl2 = RangeList::from_sorted_ranges([-5..=-3, -3..=-1, 0..=0]);
		let expected2 = RangeList::from_iter([-5..=-1, 0..=0]);
		assert_eq!(rl2, expected2);

		// Floating point ranges: overlapping/adjacent floats should be merged.
		let rl3 = RangeList::from_sorted_ranges([0.1..=3.2, 3.2..=4.0]);
		let expected3 = RangeList::from_iter([0.1..=4.0]);
		assert_eq!(rl3, expected3);

		// Regression test: a later range fully contained in the current one
		// must not shrink the accumulated range.
		let rl4 = RangeList::from_sorted_ranges([1..=10, 2..=3]);
		let expected4 = RangeList::from_iter([1..=10]);
		assert_eq!(rl4, expected4);
		let rl5 = RangeList::from_sorted_ranges([0.0..=10.0, 2.0..=3.0]);
		let expected5 = RangeList::from_iter([0.0..=10.0]);
		assert_eq!(rl5, expected5);

		// Empty iterator returns empty/default RangeList.
		let rl_empty: RangeList<f64> = RangeList::from_sorted_ranges([]);
		assert!(rl_empty.is_empty());

		// Empty ranges are ignored.
		#[expect(clippy::reversed_empty_ranges, reason = "testing empty ranges")]
		let rl_empty = RangeList::from_sorted_ranges([3..=2, 9..=7]);
		assert!(rl_empty.is_empty());
	}

	#[test]
	fn test_position_bounds() {
		let empty = RangeList::<i64>::default();
		assert_eq!(empty.first_position_bound(&Bound::Included(1)), Some(0));
		assert_eq!(empty.last_position_bound(&Bound::Unbounded), Some(0));

		// The bounds give the half-open range of positions of the elements
		// that they select.
		let rl = RangeList::from_iter([1..=4, 6..=8]);
		let positions =
			|lo, hi| rl.first_position_bound(&lo).unwrap()..rl.last_position_bound(&hi).unwrap();
		assert_eq!(positions(Bound::Unbounded, Bound::Unbounded), 0..7);
		assert_eq!(positions(Bound::Included(4), Bound::Included(6)), 3..5);
		assert_eq!(positions(Bound::Excluded(4), Bound::Excluded(6)), 4..4);
		assert_eq!(positions(Bound::Included(5), Bound::Included(5)), 4..4);
		assert_eq!(positions(Bound::Excluded(8), Bound::Included(9)), 7..7);

		// The bounds agree with `position` for every value.
		for x in -1..=10 {
			let before = rl.first_position_bound(&Bound::Included(x)).unwrap();
			let after = rl.last_position_bound(&Bound::Included(x)).unwrap();
			assert_eq!(rl.last_position_bound(&Bound::Excluded(x)), Some(before));
			assert_eq!(rl.first_position_bound(&Bound::Excluded(x)), Some(after));
			if rl.contains(&x) {
				assert_eq!(rl.position(&x), Some(before));
				assert_eq!(after, before + 1);
			} else {
				assert_eq!(rl.position(&x), None);
				assert_eq!(after, before);
			}
		}

		// Positions do not depend on the cardinality fitting in `usize`.
		let full = RangeList::from(0_u64..=u64::MAX);
		assert_eq!(full.first_position_bound(&Bound::Included(5)), Some(5));
		assert_eq!(
			full.last_position_bound(&Bound::Excluded(u64::MAX)),
			Some(usize::MAX)
		);
		assert_eq!(full.last_position_bound(&Bound::Included(u64::MAX)), None);
		assert_eq!(full.last_position_bound(&Bound::Unbounded), None);
	}

	#[test]
	fn test_position_overflow() {
		// A full-width range has more elements than `usize` can count, so its
		// cardinality overflows, but positions within it (which are distances)
		// must still be reported.
		let full = RangeList::from(0_u64..=u64::MAX);
		assert_eq!(full.card(), None);
		assert_eq!(full.position(&0), Some(0));
		assert_eq!(full.position(&u64::MAX), Some(usize::MAX));

		let signed = RangeList::from(i64::MIN..=i64::MAX);
		assert_eq!(signed.card(), None);
		assert_eq!(signed.position(&i64::MIN), Some(0));
		assert_eq!(signed.position(&i64::MAX), Some(usize::MAX));
	}

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

		// A range that is fully contained in another must not shrink it.
		let contained = RangeList::from_iter([2..=3, 1..=10, 1..=4, 12..=12, 5..=6]);
		assert_eq!(contained.iter().collect::<Vec<_>>(), vec![1..=10, 12..=12]);

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
	fn test_set_bounds() {
		let mut empty = RangeList::<i64>::default();
		empty.tighten_min(10);
		empty.tighten_max(20);
		assert_eq!(empty.min(), None);
		assert_eq!(empty.max(), None);

		let mut r = RangeList::<i64>::from_iter([1..=2, 4..=6, 8..=9]);
		r.tighten_min(0);
		assert_eq!(r.min(), Some(&1));
		r.tighten_min(1);
		assert_eq!(r.min(), Some(&1));
		r.tighten_min(2);
		assert_eq!(r.min(), Some(&2));
		r.tighten_min(4);
		assert_eq!(r.min(), Some(&4));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![4..=6, 8..=9]);
		r.tighten_min(9);
		assert_eq!(r.min(), Some(&9));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![9..=9]);
		r.tighten_min(10);
		assert_eq!(r.min(), None);
		assert!(r.is_empty());

		let mut r = RangeList::<i64>::from_iter([1..=2, 4..=6, 8..=9]);
		r.tighten_max(10);
		assert_eq!(r.max(), Some(&9));
		r.tighten_max(9);
		assert_eq!(r.max(), Some(&9));
		r.tighten_max(8);
		assert_eq!(r.max(), Some(&8));
		r.tighten_max(6);
		assert_eq!(r.max(), Some(&6));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![1..=2, 4..=6]);
		r.tighten_max(1);
		assert_eq!(r.max(), Some(&1));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![1..=1]);
		r.tighten_max(0);
		assert_eq!(r.max(), None);
		assert!(r.is_empty());
	}

	#[test]
	fn test_set_card() {
		let empty = RangeList::<i64>::default();
		assert_eq!(empty.card(), Some(0));

		let full: RangeList<i64> = (i64::MIN..=i64::MAX).into();
		assert_eq!(full.card(), None);

		let x = RangeList::<i8>::from(1..=5);
		assert_eq!(x.card(), Some(5));

		let y = RangeList::<u32>::from_iter([1..=2, 4..=6, 8..=9]);
		assert_eq!(y.card(), Some(7));
	}

	#[test]
	fn test_set_diff() {
		let empty: RangeList<i64> = RangeList::default();
		let inf: RangeList<i64> = RangeList::from_iter([i64::MIN..=i64::MAX]);
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

		// Regression test: z would previously contain 14.
		let x = RangeList::from_iter([3..=4, 6..=9, 11..=12, 14..=14, 16..=16]);
		let z: RangeList<_> = x.diff(&RangeList::from_iter([1..=1, 3..=3, 12..=14]));
		expect!["4..4 union 6..9 union 11..11 union 16..16"].assert_eq(&z.to_string());

		// Floats cut at the next/previous representable value.
		let x = RangeList::from(1.0..=5.0);
		let z: RangeList<f64> = x.diff(&RangeList::from(2.0..=4.0));
		assert_eq!(
			z.iter().collect::<Vec<_>>(),
			vec![1.0..=2.0_f64.next_down(), 4.0_f64.next_up()..=5.0]
		);
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
	fn test_std_sets() {
		let b = BTreeSet::from([1, 2, 3, 7]);
		let h = HashSet::from([7, 3, 2, 1]);
		let rl = RangeList::from_iter([1..=3, 7..=7]);
		assert_eq!(IntervalIterator::card(&b), Some(4));
		assert_eq!(IntervalIterator::card(&h), Some(4));
		assert!(IntervalIterator::contains(&b, &7) && !IntervalIterator::contains(&b, &4));
		assert!(IntervalIterator::contains(&h, &7) && !IntervalIterator::contains(&h, &4));
		assert_eq!(IntervalIterator::union::<_, RangeList<_>>(&b, &h), rl);
		assert_eq!(rl.intersect::<_, RangeList<_>>(&h), rl);
		assert!(rl.diff::<_, RangeList<_>>(&b).is_empty());
	}

	#[test]
	fn test_union_iter_exhausted() {
		// An exhausted side must not be polled again.
		let mut done = false;
		let lhs = std::iter::from_fn(move || {
			assert!(!done, "exhausted iterator was polled again");
			done = true;
			None
		});
		let union = UnionIter::from_iters(lhs, [1..=2, 4..=5].into_iter());
		assert_eq!(union.collect::<Vec<_>>(), vec![1..=2, 4..=5]);
	}
}
