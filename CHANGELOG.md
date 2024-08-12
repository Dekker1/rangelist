# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2023-08-12

### Changed

- `IntervalIter` is now named `IntervalIterator`.

### Removed

- `IntervalIterator<Item = E>` is no longer automatically implemented for all types that implement `IntoIterator<Item = RangeIncluise<E>>`.

### Added

- Add `diff` operation for types that implement `IntervalIterator` over integer types.
- Implement `IntervalIterator` for `HashSet` and `BTreeSet` from `std::collections`.
- Add `DiffIter`, `IntersectIter`, and `UnionIter` to provide lazy evaluation of set operations.

## [0.1.0] - 2024-07-05

### Added

- Add initial implementation of `RangeList` type.
- Add `IntervalIter` trait for types that can output a iterator of sorted intervals and implement it for `RangeList`.
- Add set operations `card`, `disjoint`, `intersect`, `subset`, `superset`, `union` for implementers of `IntervalIter`.

[unreleased]: https://github.com/shackle-rs/shackle/releases/compare/rangelist-v0.2.0......HEAD
[0.2.0]: https://github.com/shackle-rs/shackle/releases/compare/rangelist-v0.1.0...rangelist-v0.2.0
[0.1.0]: https://github.com/shackle-rs/shackle/releases/tag/rangelist-v0.1.0
