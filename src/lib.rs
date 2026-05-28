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
	iter::{Map, Peekable},
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

/// A trait that provides operations on iterators of orderdered intervals.
pub trait IntervalIterator<E: PartialOrd> {
	/// The type of the interval iterator.
	type IntervalIter: Iterator<Item = RangeInclusive<E>>;

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
	/// The implementation decrements the lowest value of `self` and increments
	/// the largest value of `self`. This could cause a panic if this causes
	/// overflow in `E`.
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
	fn intervals(&self) -> Self::IntervalIter;

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
	lhs: Peekable<I>,
	/// Iterator yielding the ranges of the right-hand side of the union
	rhs: Peekable<J>,
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
	type IntervalIter = Map<<BTreeSet<E> as IntoIterator>::IntoIter, fn(E) -> RangeInclusive<E>>;

	fn intervals(&self) -> Self::IntervalIter {
		self.clone().into_iter().map(|e| e.clone()..=e)
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
	pub fn new<A, B>(lhs: &A, rhs: &B) -> Self
	where
		A: IntervalIterator<E, IntervalIter = I>,
		B: IntervalIterator<E, IntervalIter = J>,
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
	/// Returns the [`Self::position`] pointing at the smallest element greater
	/// than (or equal to) the given bound.
	///
	/// Passing `Bound::Included(x)` will return the position of the smallest
	/// element greater than or equal to `x`, or `None` if all elements are
	/// smaller than `x`.
	///
	/// Passing `Bound::Excluded(x)` will return the position of the smallest
	/// element greater than `x`, or `None` if all elements are smaller than or
	/// equal to `x`.
	///
	/// Passing `Bound::Unbounded` will return `None`.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// # use std::ops::Bound;
	/// let rl = RangeList::from_iter([1..=4, 6..=8]);
	/// assert_eq!(rl.first_position_bound(&Bound::Included(-1)), Some(0));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(1)), Some(0));
	/// assert_eq!(rl.first_position_bound(&Bound::Excluded(1)), Some(1));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(4)), Some(3));
	/// assert_eq!(rl.first_position_bound(&Bound::Excluded(4)), Some(4));
	/// assert_eq!(rl.first_position_bound(&Bound::Included(8)), Some(6));
	///
	/// assert_eq!(rl.first_position_bound(&Bound::Included(9)), None);
	/// ```
	pub fn first_position_bound(&self, bound: &Bound<E>) -> Option<usize>
	where
		E: Clone + Step,
	{
		let elem = match bound {
			Bound::Included(x) => x,
			Bound::Excluded(x) => x,
			Bound::Unbounded => {
				return None;
			}
		};
		let mut pos = 0;
		let card = self.card()?;
		for (start, end) in &self.ranges {
			if elem < start {
				return Some(pos);
			}
			if elem <= end {
				pos += Step::steps_between(start, elem)?;
				match bound {
					Bound::Excluded(_) if pos > card => return None,
					Bound::Excluded(_) => pos += 1,
					_ => {}
				}
				debug_assert!(pos <= card);
				return Some(pos);
			}
			pos += Step::steps_between(start, end)? + 1;
		}
		debug_assert_eq!(pos, self.card().unwrap());
		None
	}

	/// Construct a [`RangeList`] from an iterator of elements that are known to
	/// be yielded in sorted (increasing) order, but where duplicates might
	/// still exist.
	///
	/// # Warning
	///
	/// This function will panic if the iterator yields a larger element than
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
	/// This function will panic if the iterator yields a strictly larger range
	/// than the previous one.
	pub fn from_sorted_ranges<T: IntoIterator<Item = RangeInclusive<E>>>(iter: T) -> Self
	where
		E: Adjacent + Clone,
	{
		let mut it = iter.into_iter();
		let mut ranges = Vec::new();
		let Some(mut cur) = it.next().map(|r| (r.start().clone(), r.end().clone())) else {
			return Self::default();
		};
		for next in it {
			if next.start() < &cur.0 {
				panic!("ranges must be yielded in sorted order");
			}
			let next = (next.start().clone(), next.end().clone());
			// Merge the ranges if they overlap, or if they are adjacent (the
			// successor of the current end reaches the start of the next range).
			let adjacent = cur.1.successor().is_some_and(|succ| next.0 <= succ);
			if cur.1 >= next.0 || adjacent {
				// `next` may be fully contained in `cur`, so keep the larger end.
				cur.1 = max(cur.1, next.1)
			} else {
				ranges.push(cur);
				cur = next;
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

	/// Returns the [`Self::position`] pointing at the largest element smaller
	/// than (or equal to) the given bound.
	///
	/// Passing `Bound::Included(x)` will return the position of the largest
	/// element smaller than or equal to `x`, or `None` if all elements are
	/// larger `x`.
	///
	/// Passing `Bound::Excluded(x)` will return the position of the largest
	/// element smaller than `x`, or `None` if all elements are larger than or
	/// equal to `x`.
	///
	/// Passing `Bound::Unbounded` will return `None`.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// # use std::ops::Bound;
	/// let rl = RangeList::from_iter([1..=4, 6..=8]);
	/// assert_eq!(rl.last_position_bound(&Bound::Included(1)), Some(0));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(4)), Some(3));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(4)), Some(2));
	/// assert_eq!(rl.last_position_bound(&Bound::Included(9)), Some(7));
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(9)), Some(7));
	///
	/// assert_eq!(rl.last_position_bound(&Bound::Included(-1)), None);
	/// assert_eq!(rl.last_position_bound(&Bound::Excluded(1)), None);
	/// ```
	pub fn last_position_bound(&self, bound: &Bound<E>) -> Option<usize>
	where
		E: Clone + Step,
	{
		let mut pos = self.card()?;
		let lb = self.min()?;
		let elem = match bound {
			Bound::Included(x) => {
				if x < lb {
					return None;
				}
				x
			}
			Bound::Excluded(x) => {
				if x <= lb {
					return None;
				}
				x
			}
			Bound::Unbounded => {
				return None;
			}
		};
		for (start, end) in self.ranges.iter().rev() {
			if elem > end {
				return Some(pos);
			}
			if elem >= start {
				pos -= Step::steps_between(elem, end)? + 1;
				if matches!(bound, Bound::Excluded(_)) {
					pos -= 1;
				}
				return Some(pos);
			}
			pos -= Step::steps_between(start, end)? + 1;
		}
		unreachable!()
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
		let mut pos = 0;
		for (start, end) in &self.ranges {
			if elem < start {
				return None;
			}
			if elem <= end {
				let elems = Step::steps_between(start, elem)?;
				return Some(pos + elems);
			}
			pos += Step::steps_between(start, end)? + 1;
		}
		None
	}

	/// Tightens the lower bound of the range list, removing any (partial)
	/// ranges that are below the new lower bound.
	#[deprecated(since = "0.5.0", note = "use `tighten_min` instead")]
	pub fn set_lower_bound(&mut self, lower_bound: E)
	where
		E: Debug,
	{
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
		let last_kept = self
			.ranges
			.iter()
			.enumerate()
			.rfind(|(_, (start, _))| *start <= max)
			.map(|(i, _)| i);
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
	pub fn tighten_min(&mut self, min: E)
	where
		E: Debug,
	{
		let first_kept = self
			.ranges
			.iter()
			.enumerate()
			.find_map(|(i, (_, end))| (*end >= min).then_some(i));
		if let Some(start) = first_kept {
			if self.ranges[start].0 < min {
				self.ranges[start].0 = min;
			}
			if start > 0 {
				for i in start..self.ranges.len() {
					self.ranges.swap(i, i - start);
				}
				self.ranges.truncate(self.ranges.len() - start);
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
		if value.is_empty() {
			RangeList { ranges: Vec::new() }
		} else {
			Self {
				ranges: vec![(value.start().clone(), value.end().clone())],
			}
		}
	}
}

impl<E: Clone + PartialOrd> From<RangeInclusive<E>> for RangeList<E> {
	fn from(value: RangeInclusive<E>) -> Self {
		(&value).into()
	}
}

impl<E, R> FromIterator<R> for RangeList<E>
where
	E: Adjacent + Clone + PartialOrd,
	R: Into<RangeInclusive<E>>,
{
	fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
		let mut non_empty: Vec<RangeInclusive<E>> = iter
			.into_iter()
			.map(|r| r.into())
			.filter(|r| !r.is_empty())
			.collect();
		non_empty.sort_by(|a, b| {
			a.start()
				.partial_cmp(b.start())
				.expect("the order of the bounds in the RangeList cannot be partial")
		});
		Self::from_sorted_ranges(non_empty)
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

	/// Create a new [`UnionIter`] from two set types that implement the
	/// [`IntervalIterator`] trait.
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
						if let Some(l) = self.lhs.peek()
							&& overlap(&ext, l) == RangeOrdering::Overlap
						{
							ext = ext.start().clone()..=max(ext.end(), l.end()).clone();
							let _ = self.lhs.next();
							continue;
						}
						if let Some(r) = self.rhs.peek()
							&& overlap(&ext, r) == RangeOrdering::Overlap
						{
							ext = ext.start().clone()..=max(ext.end(), r.end()).clone();
							let _ = self.rhs.next();
							continue;
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

		// Regression test: a later range fully contained in the current one must
		// not shrink the accumulated range.
		let rl4 = RangeList::from_sorted_ranges([1..=10, 2..=3]);
		let expected4 = RangeList::from_iter([1..=10]);
		assert_eq!(rl4, expected4);
		let rl5 = RangeList::from_sorted_ranges([0.0..=10.0, 2.0..=3.0]);
		let expected5 = RangeList::from_iter([0.0..=10.0]);
		assert_eq!(rl5, expected5);

		// Empty iterator returns empty/default RangeList.
		let rl_empty: RangeList<f64> = RangeList::from_sorted_ranges([]);
		assert!(rl_empty.is_empty());
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
}
