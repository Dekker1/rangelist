//! Numeric traits describing the element types that can be stored in a
//! [`RangeList`](crate::RangeList), along with their implementations for the
//! primitive integer and floating point types.

/// Macro implementing [`Adjacent`] for integer types using checked successor
/// and predecessor operations.
macro_rules! adjacent_int_impls {
	( $( $t:ty ),+ $(,)? ) => {
		$(
			impl Adjacent for $t {
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

/// Macro implementing [`Step`] for integer types, split by their width relative
/// to `usize` so that no more overflow checks are performed than necessary.
macro_rules! step_int_impls {
	{
		narrower than or same width as usize:
			$( [ $u_narrower:ident $i_narrower:ident ] ),+;
		wider than usize:
			$( [ $u_wider:ident $i_wider:ident ] ),+;
	} => {
		$(
			impl Step for $u_narrower {
				#[inline]
				fn steps_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						// The distance fits in `usize`, as `$u_narrower` is no
						// wider than `usize`.
						#[allow(trivial_numeric_casts, reason = "macro is used for many integer types including usize")]
						Some((*end - *start) as usize)
					} else {
						None
					}
				}
			}

			impl Step for $i_narrower {
				#[inline]
				fn steps_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						#[allow(trivial_numeric_casts, reason = "macro is used for many integer types including isize")]
						Some((*end as isize).wrapping_sub(*start as isize) as usize)
					} else {
						None
					}
				}
			}
		)+

		$(
			impl Step for $u_wider {
				#[inline]
				fn steps_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						usize::try_from(*end - *start).ok()
					} else {
						None
					}
				}
			}

			impl Step for $i_wider {
				#[inline]
				fn steps_between(start: &Self, end: &Self) -> Option<usize> {
					if *start <= *end {
						usize::try_from(end.checked_sub(*start)?).ok()
					} else {
						None
					}
				}
			}
		)+
	};
}

/// Trait implemented for types whose values have a well-defined predecessor and
/// successor, allowing the ranges of a [`RangeList`](crate::RangeList) to be
/// merged when adjacent and cut apart (for example by
/// [`IntervalIterator::diff`](crate::IntervalIterator::diff)).
///
/// This is implementable for floating point types (using the next and previous
/// representable values), where only the number of steps between two values,
/// exposed through [`Step`], cannot be determined.
pub trait Adjacent: Sized {
	/// Returns the element that would be considered by the *predecessor* of
	/// `self`, or `None` if it should be considered the smallest possible
	/// element.
	fn predecessor(&self) -> Option<Self>;

	/// Returns the element that would be considered by the *successor* of
	/// `self`, or `None` if it should be considered the largest possible
	/// element.
	fn successor(&self) -> Option<Self>;
}

/// Trait for types where the number of steps between two values can be counted.
///
/// This shadows the unstable [`std::iter::Step`] trait, and can be replaced by
/// it once it is stabilized.
pub trait Step: Sized {
	/// Returns the number of *steps* between `start` and `end`.
	///
	/// Returns `None` if the number of steps would overflow `usize`, or cannot
	/// be determined.
	///
	/// # Invariants
	///
	/// For any `a`, `b`, and `n`:
	///
	/// - `steps_between(&a, &b) == Some(n)` only if `a + n == b`
	/// - `steps_between(&a, &b) == Some(0)` if and only if `a == b`
	/// - `steps_between(&a, &b) == None` if `a > b` or `b - a > usize::MAX`
	fn steps_between(start: &Self, end: &Self) -> Option<usize>;
}

impl Adjacent for f32 {
	#[inline]
	fn predecessor(&self) -> Option<Self> {
		debug_assert!(!self.is_nan(), "NaN has no predecessor");
		let prev = self.next_down();
		(prev != Self::NEG_INFINITY).then_some(prev)
	}

	#[inline]
	fn successor(&self) -> Option<Self> {
		debug_assert!(!self.is_nan(), "NaN has no successor");
		let next = self.next_up();
		(next != Self::INFINITY).then_some(next)
	}
}

impl Adjacent for f64 {
	#[inline]
	fn predecessor(&self) -> Option<Self> {
		debug_assert!(!self.is_nan(), "NaN has no predecessor");
		let prev = self.next_down();
		(prev != Self::NEG_INFINITY).then_some(prev)
	}

	#[inline]
	fn successor(&self) -> Option<Self> {
		debug_assert!(!self.is_nan(), "NaN has no successor");
		let next = self.next_up();
		(next != Self::INFINITY).then_some(next)
	}
}

adjacent_int_impls!(
	u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize
);

#[cfg(target_pointer_width = "64")]
step_int_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [u64 i64], [usize isize];
	wider than usize: [u128 i128];
}

#[cfg(target_pointer_width = "32")]
step_int_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [usize isize];
	wider than usize: [u64 i64], [u128 i128];
}

#[cfg(target_pointer_width = "16")]
step_int_impls! {
	narrower than or same width as usize: [u8 i8], [u16 i16], [usize isize];
	wider than usize: [u32 i32], [u64 i64], [u128 i128];
}
