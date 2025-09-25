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

/// Macro to help with the implementation of [`DiscreteElements`] for the
/// integer types in the standard library.
macro_rules! discrete_elems_impls {
	{
		narrower than or same width as usize:
			$( [ $u_narrower:ident $i_narrower:ident ] ),+;
		wider than usize:
			$( [ $u_wider:ident $i_wider:ident ] ),+;
	} => {
		$(
			impl DiscreteElement for $u_narrower {
				#[inline]
				fn elem_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						// This relies on $u_narrower <= usize
						#[allow(trivial_numeric_casts, reason = "macro is used for many integer types including usize")]
						let steps = (*end - *start) as usize;
						steps.checked_add(1)
					} else {
						None
					}
				}

				#[inline]
				fn successor(&self) -> Option<Self> {
					self.checked_add(1)
				}

				#[inline]
				fn predecessor(&self) -> Option<Self> {
					self.checked_sub(1)
				}
			}

			impl DiscreteElement for $i_narrower {
				#[inline]
				fn elem_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						#[allow(trivial_numeric_casts, reason = "macro is used for many integer types including isize")]
						let steps = (*end as isize).wrapping_sub(*start as isize) as usize;
						steps.checked_add(1)
					} else {
						None
					}
				}

				#[inline]
				fn successor(&self) -> Option<Self> {
					self.checked_add(1)
				}

				#[inline]
				fn predecessor(&self) -> Option<Self> {
					self.checked_sub(1)
				}
			}
		)+

		$(
			impl DiscreteElement for $u_wider {
				#[inline]
				fn elem_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						if let Ok(steps) = usize::try_from(*end - *start) {
							steps.checked_add(1)
						} else {
							None
						}
					} else {
						None
					}
				}

				#[inline]
				fn successor(&self) -> Option<Self> {
					self.checked_add(1)
				}

				#[inline]
				fn predecessor(&self) -> Option<Self> {
					self.checked_sub(1)
				}
			}

			impl DiscreteElement for $i_wider {
				#[inline]
				fn elem_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						if let Ok(steps) = usize::try_from(end.checked_sub(*start)?) {
							steps.checked_add(1)
						} else {
							None
						}
					} else {
						None
					}
				}

				#[inline]
				fn successor(&self) -> Option<Self> {
					self.checked_add(1)
				}

				#[inline]
				fn predecessor(&self) -> Option<Self> {
					self.checked_sub(1)
				}
			}
		)+
	};
}

use std::{
	any::Any,
	collections::{BTreeSet, HashSet},
	fmt::{Debug, Display},
	iter::{Map, Peekable},
	ops::{Bound, RangeInclusive},
};

/// An iterator combinator that given two iterators yielding ordered ranges,
/// yields the ordered ranges of elements that are in the ranges yielded by
/// `lhs` iterator, but does not include elements that are in the ranges yielded
/// by the `rhs` iterator.
#[derive(Debug)]
pub struct DiffIter<
	E: Clone + DiscreteElement + PartialOrd,
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

/// Trait implemented for type that should be considered discrete elements when
/// part of a [`RangeList`].
///
// Note that the methods in this trait are inspired by the `Step` trait and can
// be replaced when this is merged into stable Rust.
pub trait DiscreteElement: Sized {
	/// Returns the number of *elements* between `start` to `end` (inclusive).
	///
	/// Returns `None` if the number of steps would overflow `usize`, or cannot be
	/// determined.
	///
	/// # Invariants
	///
	/// For any `a`, `b`, and `n`:
	///
	/// - `elem_between(&a, &b) == Some(n)` only if `a <= b`
	/// - `elem_between(&a, &b) == Some(0)` if and only if `a == b`
	/// - `elem_between(&a, &b) == None` if `a > b`
	fn elem_between(start: &Self, end: &Self) -> Option<usize>;

	/// Returns the element that would be considered by the *successor* of `self`,
	/// or `None` if it should be considered the largest possible element.
	fn successor(&self) -> Option<Self>;

	/// Returns the element that would be considered by the *predecessor* of
	/// `self`, or `None` if it should be considered the smallest possible
	/// element.
	fn predecessor(&self) -> Option<Self>;
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
	///
	/// Returns `None` if the number of steps would overflow `usize`.
	fn card(&self) -> Option<usize>
	where
		E: DiscreteElement,
	{
		let mut card = 0;
		for r in self.intervals() {
			match DiscreteElement::elem_between(r.start(), r.end()) {
				Some(c) => card += c,
				None => return None,
			}
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
		E: Clone + DiscreteElement,
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
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd)]
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

impl<E: Clone + DiscreteElement + PartialOrd, I, J> DiffIter<E, I, J>
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

	/// Create a new [`DiffIter`] from two set types that implement the [`IntervalIterator`] trait.
	pub fn new<A, B>(lhs: &A, rhs: &B) -> Self
	where
		A: IntervalIterator<E, IntervalIter = I>,
		B: IntervalIterator<E, IntervalIter = J>,
	{
		Self::from_iters(lhs.intervals(), rhs.intervals())
	}
}

impl<E: Clone + DiscreteElement + PartialOrd, I, J> Iterator for DiffIter<E, I, J>
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
		E: Clone + DiscreteElement,
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
				pos += DiscreteElement::elem_between(start, elem)?;
				match bound {
					Bound::Included(_) => pos -= 1,
					Bound::Excluded(_) if pos > card => return None,
					_ => {}
				}
				debug_assert!(pos <= card);
				return Some(pos);
			}
			pos += DiscreteElement::elem_between(start, end)?;
		}
		debug_assert_eq!(pos, self.card().unwrap());
		None
	}

	/// Internal method used to construct a [`RangeList`] from an iterator of
	/// pairs that is known to be sorted order, but where ranges might still need
	/// to be merged.
	fn from_sorted_iter<T: IntoIterator<Item = (E, E)>>(iter: T) -> Self
	where
		E: Any + Clone,
	{
		let mut it = iter.into_iter();
		let mut ranges = Vec::new();
		let Some(mut cur) = it.next() else {
			return Self::default();
		};
		for next in it {
			// Determine distance between the two ranges if the elements are discrete.
			let inbetween: &dyn Any = &(cur.1.clone(), next.0.clone());
			let dist = if let Some((ub, lb)) = inbetween.downcast_ref::<(isize, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(i128, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(i64, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(i32, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(i16, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(i8, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(usize, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(u128, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(u64, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(u32, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(u16, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else if let Some((ub, lb)) = inbetween.downcast_ref::<(u8, _)>() {
				DiscreteElement::elem_between(ub, lb)
			} else {
				None
			};

			if cur.1 >= next.0 || dist.unwrap_or(usize::MAX) <= 2 {
				cur.1 = next.1
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
	/// element smaller than or equal to `x`, or `None` if all elements are larger
	/// `x`.
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
		E: Clone + DiscreteElement,
	{
		let mut pos = self.card()?;
		let lb = self.lower_bound()?;
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
				pos -= DiscreteElement::elem_between(elem, end)?;
				if matches!(bound, Bound::Excluded(_)) {
					pos -= 1;
				}
				return Some(pos);
			}
			pos -= DiscreteElement::elem_between(start, end)?;
		}
		unreachable!()
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
		E: DiscreteElement,
	{
		let mut pos = 0;
		for (start, end) in &self.ranges {
			if elem < start {
				return None;
			}
			if elem <= end {
				let elems = DiscreteElement::elem_between(start, elem)?;
				return Some(pos + elems - 1);
			}
			pos += DiscreteElement::elem_between(start, end)?;
		}
		None
	}

	/// Tightens the lower bound of the range list, removing any (partial) ranges
	/// that are below the new lower bound.
	///
	/// Note that no action is taken if the new lower bound is less than or equal
	/// to the current lower bound.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let mut r = RangeList::from_iter([-5..=-3, 1..=4, 6..=7]);
	/// r.set_lower_bound(2);
	/// assert_eq!(r.lower_bound(), Some(&2));
	/// assert_eq!(r.iter().collect::<Vec<_>>(), vec![2..=4, 6..=7]);
	/// ```
	pub fn set_lower_bound(&mut self, lower_bound: E)
	where
		E: Debug,
	{
		let first_kept = self
			.ranges
			.iter()
			.enumerate()
			.find_map(|(i, (_, end))| (*end >= lower_bound).then_some(i));
		if let Some(start) = first_kept {
			if self.ranges[start].0 < lower_bound {
				self.ranges[start].0 = lower_bound;
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

	/// Tightens the upper bound of the range list, removing any (partial) ranges
	/// that are above the new upper bound.
	///
	/// Note that no action is taken if the new upper bound is greater than or
	/// equal to the current upper bound.
	///
	/// # Examples
	///
	/// ```
	/// # use rangelist::RangeList;
	/// let mut r = RangeList::from_iter([-5..=-3, 1..=4, 6..=7]);
	/// r.set_upper_bound(3);
	/// assert_eq!(r.upper_bound(), Some(&3));
	/// assert_eq!(r.iter().collect::<Vec<_>>(), vec![-5..=-3, 1..=3]);
	/// ```
	pub fn set_upper_bound(&mut self, upper_bound: E) {
		let last_kept = self
			.ranges
			.iter()
			.enumerate()
			.rfind(|(_, (start, _))| *start <= upper_bound)
			.map(|(i, _)| i);
		if let Some(end) = last_kept {
			self.ranges.truncate(end + 1);
			let last = self.ranges.last_mut().unwrap();
			if last.1 > upper_bound {
				last.1 = upper_bound;
			}
		} else {
			self.ranges = Vec::new();
		}
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
	E: Any + Clone + PartialOrd,
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
		non_empty.sort_by(|a, b| {
			a.0.partial_cmp(&b.0)
				.expect("the order of the bounds in the RangeList cannot be partial")
		});
		Self::from_sorted_iter(non_empty)
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

#[cfg(target_pointer_width = "64")]
discrete_elems_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [u64 i64], [usize isize];
	wider than usize: [u128 i128];
}

#[cfg(target_pointer_width = "32")]
discrete_elems_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [usize isize];
	wider than usize: [u64 i64], [u128 i128];
}

#[cfg(target_pointer_width = "16")]
discrete_elems_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [usize isize];
	wider than usize: [u32 i32], [u64 i64], [u128 i128];
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
		empty.set_lower_bound(10);
		empty.set_upper_bound(20);
		assert_eq!(empty.lower_bound(), None);
		assert_eq!(empty.upper_bound(), None);

		let mut r = RangeList::<i64>::from_iter([1..=2, 4..=6, 8..=9]);
		r.set_lower_bound(0);
		assert_eq!(r.lower_bound(), Some(&1));
		r.set_lower_bound(1);
		assert_eq!(r.lower_bound(), Some(&1));
		r.set_lower_bound(2);
		assert_eq!(r.lower_bound(), Some(&2));
		r.set_lower_bound(4);
		assert_eq!(r.lower_bound(), Some(&4));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![4..=6, 8..=9]);
		r.set_lower_bound(9);
		assert_eq!(r.lower_bound(), Some(&9));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![9..=9]);
		r.set_lower_bound(10);
		assert_eq!(r.lower_bound(), None);
		assert!(r.is_empty());

		let mut r = RangeList::<i64>::from_iter([1..=2, 4..=6, 8..=9]);
		r.set_upper_bound(10);
		assert_eq!(r.upper_bound(), Some(&9));
		r.set_upper_bound(9);
		assert_eq!(r.upper_bound(), Some(&9));
		r.set_upper_bound(8);
		assert_eq!(r.upper_bound(), Some(&8));
		r.set_upper_bound(6);
		assert_eq!(r.upper_bound(), Some(&6));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![1..=2, 4..=6]);
		r.set_upper_bound(1);
		assert_eq!(r.upper_bound(), Some(&1));
		assert_eq!(r.iter().collect::<Vec<_>>(), vec![1..=1]);
		r.set_upper_bound(0);
		assert_eq!(r.upper_bound(), None);
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
