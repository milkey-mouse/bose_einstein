//! A data structure that efficiently partitions elements into two distinct sets.
//!
//! [`Partition<T>`] maintains a collection of elements divided into "left" and
//! "right" partitions, with `O(1)` pushes, pops, and moves between partitions.
//! Internally, it uses a partition index to divide a single [`Vec<T>`] into two
//! partitions. Most operations on [`Vec<T>`] are supported by [`Partition<T>`].
//!
//! **Partitions are sets, not lists:** within each partition, order is not
//! necessarily preserved. (One could say the elements of a [`Partition<T>`]
//! obey [Bose-Einstein statistics](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics).)
//!
//! ## Examples
//!
//! ### Basic Usage
//!
//! ```rust
//! use bose_einstein::Partition;
//!
//! // create a partition and add elements to both sides
//! let mut partition = Partition::new();
//! partition.push_left("apple");
//! partition.push_left("banana");
//! partition.push_right("cherry");
//! partition.push_right("date");
//!
//! // sort elements for predictable assertions
//! // (remember: order is not preserved within partitions)
//! partition.left_mut().sort();
//! partition.right_mut().sort();
//!
//! // access elements in each partition
//! assert_eq!(partition.left(), &["apple", "banana"]);
//! assert_eq!(partition.right(), &["cherry", "date"]);
//!
//! // move elements between partitions
//! let moved = partition.move_to_right();
//! assert!(moved.is_some()); // we got some element from the left partition
//! // element should be one of the two we added to the left partition
//! let moved_val = moved.unwrap();
//! assert!(moved_val == "apple" || moved_val == "banana");
//!
//! // get both partitions at once with the convenient partitions() method
//! let (left, right) = partition.partitions();
//! assert_eq!(left.len(), 1);
//! assert_eq!(right.len(), 3);
//! ```
//!
//! ### Using Drain Operations
//!
//! ```rust
//! use bose_einstein::Partition;
//!
//! let mut partition = Partition::new();
//!
//! // add some task IDs to the pending (left) and completed (right) lists
//! partition.push_left(101);
//! partition.push_left(102);
//! partition.push_left(103);
//! partition.push_right(201);
//! partition.push_right(202);
//!
//! // process all pending tasks and move them to completed
//! println!("Processing pending tasks:");
//! for task_id in partition.drain_to_right() {
//!     println!("Processing task {}", task_id);
//!     // tasks are moved to the right partition automatically
//! }
//!
//! // all tasks are now in the completed list
//! assert_eq!(partition.left().len(), 0);
//! assert_eq!(partition.right().len(), 5);
//!
//! // we can also archive completed tasks by moving to left (in a real app)
//! println!("Archiving old tasks:");
//! for task_id in partition.drain_to_left().take(2) {
//!     println!("Archiving task {}", task_id);
//!     // even when we only process some items, all move to the destination
//! }
//!
//! // all tasks moved to the left (archived) partition
//! assert_eq!(partition.left().len(), 5);
//! assert_eq!(partition.right().len(), 0);
//! ```
//!
//! ### Using With Custom Types
//!
//! ```rust
//! use bose_einstein::Partition;
//!
//! // a simple task type for demonstration
//! #[derive(Debug, Clone, Copy, PartialEq)]
//! struct Task {
//!     id: u32,
//!     is_important: bool,
//! }
//!
//! // use partition to organize tasks by importance
//! let mut tasks = Partition::new();
//!
//! // add some important tasks to the left partition
//! tasks.push_left(Task { id: 1, is_important: true });
//! tasks.push_left(Task { id: 2, is_important: true });
//!
//! // add some regular tasks to the right partition
//! tasks.push_right(Task { id: 3, is_important: false });
//! tasks.push_right(Task { id: 4, is_important: false });
//!
//! // check that we have the correct number of tasks in each partition
//! assert_eq!(tasks.left().len(), 2);
//! assert_eq!(tasks.right().len(), 2);
//!
//! // we can move a task from important to regular
//! let moved_task = tasks.move_to_right();
//! assert!(moved_task.is_some());
//! // since we only added important tasks to the left partition,
//! // any task moved from left should be important
//! assert!(moved_task.unwrap().is_important);
//!
//! // now we have one less important task
//! assert_eq!(tasks.left().len(), 1);
//! assert_eq!(tasks.right().len(), 3);
//!
//! // we can access and modify tasks in each partition
//! // (In a real application we might use more sophisticated filtering)
//! let contains_task1 = tasks.left().iter().any(|task| task.id == 1) ||
//!                       tasks.right().iter().any(|task| task.id == 1);
//! assert!(contains_task1, "Task 1 should be in either partition");
//! ```
#![cfg_attr(not(test), no_std)]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]

extern crate alloc;

use alloc::{
    collections::TryReserveError,
    vec::{self, Vec},
};
use core::{fmt, mem};

#[cfg(feature = "allocator_api")]
use alloc::alloc::{Allocator, Global};

/// A data structure that partitions elements into left and right sets.
///
/// Order within each set is not necessarily preserved after elements are added
/// or removed. (One could say [`Partition<T>`] obeys
/// [Bose-Einstein statistics](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics).)
///
/// # Examples
///
/// ```
/// use bose_einstein::Partition;
/// let mut p = Partition::new();
/// p.push_left(1);
/// p.push_left(2);
/// p.push_right(3);
///
/// // order within partitions is not necessarily preserved, so sort before comparing
/// p.left_mut().sort();
/// p.right_mut().sort();
///
/// assert_eq!(p.left(), &[1, 2]);
/// assert_eq!(p.right(), &[3]);
/// ```
#[cfg(feature = "allocator_api")]
#[derive(Clone)]
pub struct Partition<T, A: Allocator = Global> {
    inner: Vec<T, A>,
    partition: usize,
}

/// A data structure that partitions elements into left and right sets.
///
/// Order within each set is not necessarily preserved after elements are added
/// or removed. (One could say [`Partition<T>`] obeys
/// [Bose-Einstein statistics](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics).)
///
/// # Examples
///
/// ```
/// use bose_einstein::Partition;
/// let mut p = Partition::new();
/// p.push_left(1);
/// p.push_left(2);
/// p.push_right(3);
///
/// // order within partitions is not necessarily preserved, so sort before comparing
/// p.left_mut().sort();
/// p.right_mut().sort();
///
/// assert_eq!(p.left(), &[1, 2]);
/// assert_eq!(p.right(), &[3]);
/// ```
#[cfg(not(feature = "allocator_api"))]
#[derive(Clone)]
pub struct Partition<T> {
    inner: Vec<T>,
    partition: usize,
}

impl<T: fmt::Debug> fmt::Debug for Partition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Partition")
            .field("left", &self.left())
            .field("right", &self.right())
            .finish()
    }
}

impl<T> Default for Partition<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "allocator_api")]
impl<T, A: Allocator> Partition<T, A> {
    /// Constructs a new, empty `Partition<T, A>`. with the provided allocator.
    ///
    /// The partition will not allocate until elements are pushed onto it.
    ///
    /// This is an allocator-aware version of [`Partition::new()`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// extern crate alloc;
    /// use bose_einstein::Partition;
    ///
    /// // create a new partition with the global allocator
    /// let p: Partition<usize, _> = Partition::new_in(alloc::alloc::Global);
    /// assert!(p.is_empty());
    /// ```
    #[cfg(feature = "allocator_api")]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            inner: Vec::new_in(alloc),
            partition: 0,
        }
    }

    /// Constructs a new, empty `Partition<T, A>` with at least the specified
    /// capacity with the provided allocator.
    ///
    /// The partition will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the partition will not allocate.
    ///
    /// If it is important to know the exact allocated capacity of a
    /// `Partition`, always use the [`capacity`] method after construction.
    ///
    /// For `Partition<T, A>` where `T` is a zero-sized type, there will be no
    /// allocation and the capacity will always be `usize::MAX`.
    ///
    /// This is an allocator-aware version of [`Partition::with_capacity()`].
    ///
    /// [`capacity`]: Partition::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds isize::MAX bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// extern crate alloc;
    /// use bose_einstein::Partition;
    ///
    /// // create a new partition with the global allocator and capacity
    /// let p: Partition<usize, _> = Partition::with_capacity_in(10, alloc::alloc::Global);
    /// assert!(p.capacity() >= 10);
    /// assert!(p.is_empty());
    /// ```
    #[cfg(feature = "allocator_api")]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            inner: Vec::with_capacity_in(capacity, alloc),
            partition: 0,
        }
    }

    /// Creates a `Partition<T>` from raw parts.
    ///
    /// # Panics
    ///
    /// Panics if `partition` is greater than `vec.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// // create a partition from existing data
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // first 2 elements will be in the left partition
    ///
    /// let p = Partition::from_raw_parts(vec, partition);
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    ///
    /// The function will panic if the partition index is invalid:
    ///
    /// ```should_panic
    /// use bose_einstein::Partition;
    /// let vec = vec![1, 2, 3];
    /// let partition = 4; // invalid: beyond the length of the vector
    ///
    /// // this will panic
    /// let p = Partition::from_raw_parts(vec, partition);
    /// ```
    #[cfg(feature = "allocator_api")]
    pub fn from_raw_parts(inner: Vec<T, A>, partition: usize) -> Self {
        assert!(
            partition <= inner.len(),
            "partition index {partition} is out of bounds (vector len: {})",
            inner.len()
        );
        unsafe { Self::from_raw_parts_unchecked(inner, partition) }
    }

    /// Creates a `Partition<T>` from raw parts without checking if the
    /// partition index is valid.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `partition <= vec.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// // safe usage: partition index is valid
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // first 2 elements will be in the left partition
    ///
    /// let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    #[cfg(feature = "allocator_api")]
    pub unsafe fn from_raw_parts_unchecked(inner: Vec<T, A>, partition: usize) -> Self {
        Self { inner, partition }
    }
}

impl<T> Partition<T> {
    /// Constructs a new, empty `Partition<T>`.
    ///
    /// The partition will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    ///
    /// # #[allow(unused_mut)]
    /// let mut p: Partition<i32> = Partition::new();
    /// ```
    pub const fn new() -> Self {
        Self {
            inner: Vec::new(),
            partition: 0,
        }
    }

    /// Constructs a new, empty `Partition<T>` with at least the specified capacity.
    ///
    /// The partition will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the partition will not allocate.
    ///
    /// It is important to note that although the returned partition has the
    /// minimum *capacity* specified, the partition will have a zero *length*. For
    /// an explanation of the difference between length and capacity, see
    /// *[Capacity and reallocation]*.
    ///
    /// If it is important to know the exact allocated capacity of a
    /// `Partition`, always use the [`capacity`] method after construction.
    ///
    /// For `Partition<T>` where `T` is a zero-sized type, there will be no
    /// allocation and the capacity will always be `usize::MAX`.
    ///
    /// [Capacity and reallocation]: #capacity-and-reallocation
    /// [`capacity`]: Partition::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    ///
    /// let p1: Partition<usize> = Partition::with_capacity(10);
    /// assert!(p1.capacity() <= 10);
    ///
    /// let p2: Partition<()> = Partition::with_capacity(10);
    /// assert_eq!(p2.capacity(), usize::MAX);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            partition: 0,
        }
    }

    /// Decomposes a `Partition<T>` into its raw parts.
    ///
    /// Returns the raw pointer to the underlying data, the length of the vector (in
    /// elements), and the partition index (in elements).
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Partition`. The only way to do this is to convert
    /// the raw parts back into a `Partition` with the [`from_raw_parts`] function,
    /// allowing the destructor to perform the cleanup.
    ///
    /// [`from_raw_parts`]: Partition::from_raw_parts
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    ///
    /// let (vec, partition) = p.to_raw_parts();
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(partition, 1);
    /// ```
    pub fn to_raw_parts(self) -> (Vec<T>, usize) {
        (self.inner, self.partition)
    }

    /// Creates a `Partition<T>` from raw parts.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `vec` must have been allocated using the global allocator, such as via
    ///   the [`alloc::alloc`] function.
    /// * `partition` must be less than or equal to `vec.len()`.
    ///
    /// These requirements are always upheld by any `vec` created by standard
    /// library functions like [`Vec::new`] or [`vec!`]. Other allocation sources
    /// are allowed if the invariants are upheld.
    ///
    /// The ownership of `vec` is effectively transferred to the `Partition<T>` which
    /// may then deallocate, reallocate or change the contents of memory pointed to by
    /// the vector at will. Ensure that nothing else uses the vector after calling this
    /// function.
    ///
    /// # Panics
    ///
    /// Panics if `partition` is greater than `vec.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// // create a partition from existing data
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // first 2 elements will be in the left partition
    ///
    /// let p = Partition::from_raw_parts(vec, partition);
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    ///
    /// The function will panic if the partition index is invalid:
    ///
    /// ```should_panic
    /// use bose_einstein::Partition;
    /// let vec = vec![1, 2, 3];
    /// let partition = 4; // invalid: beyond the length of the vector
    ///
    /// // this will panic
    /// let p = Partition::from_raw_parts(vec, partition);
    /// ```
    #[cfg(not(feature = "allocator_api"))]
    pub fn from_raw_parts(inner: Vec<T>, partition: usize) -> Self {
        assert!(
            partition <= inner.len(),
            "partition index {partition} is out of bounds (vector len: {})",
            inner.len()
        );
        unsafe { Self::from_raw_parts_unchecked(inner, partition) }
    }

    /// Creates a `Partition<T>` from raw parts without checking if the
    /// partition index is valid.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `vec` must have been allocated using the global allocator, such as via
    ///   the [`alloc::alloc`] function.
    /// * The caller must ensure that `partition <= vec.len()`.
    ///
    /// These requirements are always upheld by any `vec` created by standard
    /// library functions like [`Vec::new`] or [`vec!`]. Other allocation sources
    /// are allowed if the invariants are upheld.
    ///
    /// The ownership of `vec` is effectively transferred to the `Partition<T>` which
    /// may then deallocate, reallocate or change the contents of memory pointed to by
    /// the vector at will. Ensure that nothing else uses the vector after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// // safe usage: partition index is valid
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // first 2 elements will be in the left partition
    ///
    /// let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    #[cfg(not(feature = "allocator_api"))]
    pub unsafe fn from_raw_parts_unchecked(inner: Vec<T>, partition: usize) -> Self {
        Self { inner, partition }
    }

    /// Sets the partition index to the specified value.
    ///
    /// # Safety
    ///
    /// This is unsafe because:
    /// 1. The caller must ensure that `new_partition` is less than or equal to `len()`.
    /// 2. The caller must ensure any invariants they rely on regarding the partitioning
    ///    remain valid after changing the partition index.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    /// p.push_right(4);
    ///
    /// // partition index is currently 2 (two elements in left, two in right)
    /// assert_eq!(p.left().len(), 2);
    /// assert_eq!(p.right().len(), 2);
    ///
    /// // move all elements to the right partition
    /// unsafe {
    ///     p.set_partition(0);
    /// }
    ///
    /// assert_eq!(p.left().len(), 0);
    /// assert_eq!(p.right().len(), 4);
    /// ```
    pub unsafe fn set_partition(&mut self, new_partition: usize) {
        assert!(
            new_partition <= self.inner.len(),
            "new partition index ({}) is out of bounds (vector len: {})",
            new_partition,
            self.inner.len()
        );

        self.partition = new_partition;
    }

    /// Returns both partitions as a tuple of slices.
    ///
    /// This is a convenience method that returns both the left and right
    /// partitions at once as a tuple of immutable slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    ///
    /// let (left, right) = p.partitions();
    /// assert_eq!(left, &[1]);
    /// assert_eq!(right, &[2]);
    /// ```
    pub fn partitions(&self) -> (&[T], &[T]) {
        (self.left(), self.right())
    }

    /// Returns both partitions as a tuple of mutable slices.
    ///
    /// This is a convenience method that returns both the left and right
    /// partitions at once as a tuple of mutable slices, allowing mutation of both
    /// sides simultaneously.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    ///
    /// let (left, right) = p.partitions_mut();
    ///
    /// // modify all elements in both partitions
    /// // this approach works regardless of element ordering
    /// for item in left.iter_mut() {
    ///     *item = 10;
    /// }
    /// for item in right.iter_mut() {
    ///     *item = 20;
    /// }
    ///
    /// assert_eq!(p.left(), &[10]);
    /// assert_eq!(p.right(), &[20]);
    /// ```
    pub fn partitions_mut(&mut self) -> (&mut [T], &mut [T]) {
        // safe because self.partition is always <= self.inner.len()
        self.inner.split_at_mut(self.partition)
    }

    /// Returns a slice containing all elements in the left partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    ///
    /// // order not guaranteed after pushes!
    /// p.left_mut().sort();
    ///
    /// assert_eq!(p.left(), &[1, 2]);
    /// ```
    pub fn left(&self) -> &[T] {
        // SAFETY: self.partition <= self.inner.len()
        unsafe { self.inner.get_unchecked(..self.partition) }
    }

    /// Returns a mutable slice containing all elements in the left partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    ///
    /// // since there's only one element in left partition,
    /// // we can safely modify it by iterating
    /// for item in p.left_mut().iter_mut() {
    ///     *item = 2;
    /// }
    ///
    /// // we know the partition has exactly one element with value 2
    /// assert_eq!(p.pop_left(), Some(2));
    /// ```
    pub fn left_mut(&mut self) -> &mut [T] {
        // SAFETY: self.partition <= self.inner.len()
        unsafe { self.inner.get_unchecked_mut(..self.partition) }
    }

    /// Returns a slice containing all elements in the right partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    ///
    /// // order not guaranteed after pushes!
    /// p.right_mut().sort();
    ///
    /// assert_eq!(p.right(), &[3]);
    /// ```
    pub fn right(&self) -> &[T] {
        // SAFETY: self.partition <= self.inner.len()
        unsafe { self.inner.get_unchecked(self.partition..) }
    }

    /// Returns a mutable slice containing all elements in the right partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_right(1);
    ///
    /// // since there's only one element in right partition,
    /// // we can safely modify it by iterating
    /// for item in p.right_mut().iter_mut() {
    ///     *item = 2;
    /// }
    ///
    /// // we know the right partition has exactly one element with value 2
    /// assert_eq!(p.pop_right(), Some(2));
    /// ```
    pub fn right_mut(&mut self) -> &mut [T] {
        // SAFETY: self.partition <= self.inner.len()
        unsafe { self.inner.get_unchecked_mut(self.partition..) }
    }

    /// Appends an element to the left partition.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// assert_eq!(p.left(), &[]);
    ///
    /// p.push_left(1);
    ///
    /// assert_eq!(p.left(), &[1]);
    /// ```
    pub fn push_left(&mut self, mut value: T) {
        if self.partition < self.inner.len() {
            mem::swap(
                unsafe { self.inner.get_unchecked_mut(self.partition) },
                &mut value,
            );
        }
        self.inner.push(value);
        self.partition += 1;
    }

    /// Removes the last element from the left partition and returns it, or
    /// [`None`] if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    ///
    /// assert_eq!(p.pop_left(), Some(1));
    /// assert_eq!(p.pop_left(), None);
    /// ```
    pub fn pop_left(&mut self) -> Option<T> {
        if self.partition > 0 {
            self.partition -= 1;
            let ret = self.inner.swap_remove(self.partition);
            Some(ret)
        } else {
            None
        }
    }

    /// Removes an element from the left partition and returns it.
    ///
    /// The removed element is replaced by the last element of the left partition.
    /// This does not preserve ordering within the left partition, but is O(1).
    ///
    /// If you need to preserve the element order, you should copy the
    /// left partition to a separate vector, remove the element, and insert
    /// the modified vector back with `from_raw_parts`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds for the left partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_left(3);
    ///
    /// assert_eq!(p.swap_remove_left(1), 2);
    /// // the left partition now contains [1, 3] (or [3, 1])
    /// assert_eq!(p.left().len(), 2);
    /// assert!(p.left().contains(&1));
    /// assert!(p.left().contains(&3));
    /// ```
    pub fn swap_remove_left(&mut self, index: usize) -> T {
        if index >= self.partition {
            panic!(
                "swap_remove_left: index {} out of bounds for left partition (length {})",
                index, self.partition
            );
        }

        // if we're removing the last element (most common case with pop_left()),
        // we can optimize by decrementing partition and using swap_remove
        if index == self.partition - 1 {
            self.partition -= 1;
            self.inner.swap_remove(index)
        } else {
            // swap with the last element in the left partition and then remove it
            self.inner.swap(index, self.partition - 1);
            self.partition -= 1;
            self.inner.swap_remove(self.partition)
        }
    }

    /// Appends an element to the right partition.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// assert_eq!(p.right(), &[]);
    ///
    /// p.push_right(1);
    ///
    /// assert_eq!(p.right(), &[1]);
    /// ```
    pub fn push_right(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Removes the last element from the right partition and returns it, or
    /// [`None`] if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_right(1);
    ///
    /// assert_eq!(p.pop_right(), Some(1));
    /// assert_eq!(p.pop_right(), None);
    /// ```
    pub fn pop_right(&mut self) -> Option<T> {
        if self.partition < self.inner.len() {
            self.inner.pop()
        } else {
            None
        }
    }

    /// Removes an element from the right partition and returns it.
    ///
    /// The removed element is replaced by the last element of the right partition.
    /// This does not preserve ordering within the right partition, but is O(1).
    ///
    /// If you need to preserve the element order, you should copy the
    /// right partition to a separate vector, remove the element, and insert
    /// the modified vector back with `from_raw_parts`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds for the right partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_right(1);
    /// p.push_right(2);
    /// p.push_right(3);
    ///
    /// // remove the first element in the right partition (index 0)
    /// assert_eq!(p.swap_remove_right(0), 1);
    ///
    /// // the right partition now contains [3, 2] (or [2, 3])
    /// assert_eq!(p.right().len(), 2);
    /// assert!(p.right().contains(&2));
    /// assert!(p.right().contains(&3));
    /// ```
    pub fn swap_remove_right(&mut self, index: usize) -> T {
        // the index is relative to the start of the right partition
        let absolute_index = self.partition + index;

        if index >= self.right().len() {
            panic!(
                "swap_remove_right: index {} out of bounds for right partition (length {})",
                index,
                self.right().len()
            );
        }

        // for the last element, we can just pop
        if absolute_index == self.inner.len() - 1 {
            self.inner.pop().unwrap()
        } else {
            // otherwise, use swap_remove
            self.inner.swap_remove(absolute_index)
        }
    }

    /// Creates a draining iterator that removes the specified range in the left
    /// partition and yields the removed items.
    ///
    /// Removes all elements in the left partition, leaving it empty.
    /// The vector is not reallocated.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    ///
    /// let left: Vec<_> = p.drain_left().collect();
    /// assert_eq!(left.len(), 2);
    /// assert_eq!(p.left(), &[]);
    /// assert_eq!(p.right(), &[3]);
    /// ```
    pub fn drain_left(&mut self) -> vec::Drain<'_, T> {
        // update the partition index first
        let old_partition = self.partition;
        self.partition = 0;

        // then drain the elements
        self.inner.drain(0..old_partition)
    }

    /// Creates a draining iterator that removes the specified range in the right
    /// partition and yields the removed items.
    ///
    /// Removes all elements in the right partition, leaving it empty.
    /// The vector is not reallocated.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    /// p.push_right(3);
    ///
    /// let right: Vec<_> = p.drain_right().collect();
    /// assert_eq!(right.len(), 2);
    /// assert_eq!(p.left(), &[1]);
    /// assert_eq!(p.right(), &[]);
    /// ```
    pub fn drain_right(&mut self) -> vec::Drain<'_, T> {
        // simply drain from partition to end
        self.inner.drain(self.partition..)
    }

    /// Returns the number of elements in the partition, also referred to
    /// as its 'length'.
    ///
    /// This is the sum of the number of elements in the left and right partitions.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// assert_eq!(p.len(), 0);
    ///
    /// p.push_left(1);
    /// p.push_right(2);
    /// assert_eq!(p.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the partition contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// assert!(p.is_empty());
    ///
    /// p.push_left(1);
    /// assert!(!p.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of elements the partition can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let p: Partition<usize> = Partition::with_capacity(10);
    /// assert!(p.capacity() >= 10);
    /// ```
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// into the partition.
    ///
    /// The collection may reserve more space to speculatively avoid
    /// frequent reallocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.reserve(10);
    /// assert!(p.capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Reserves the minimum capacity for at least `additional` more elements to
    /// be inserted into the partition.
    ///
    /// Unlike [`reserve`](Partition::reserve), this will not deliberately over-allocate
    /// to speculatively avoid frequent allocations. After calling `reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: Partition::reserve
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.reserve_exact(10);
    /// assert!(p.capacity() >= 11);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional);
    }

    /// Shrinks the capacity of the partition as much as possible.
    ///
    /// The behavior of this method depends on the allocator, which may either shrink the
    /// vector in-place or reallocate. The resulting partition might still have some
    /// excess capacity, just as is the case for [`with_capacity`]. See
    /// [`Allocator::shrink`] for more details.
    ///
    /// [`with_capacity`]: Partition::with_capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::with_capacity(10);
    /// p.push_left(1);
    /// p.push_right(2);
    /// p.shrink_to_fit();
    /// assert!(p.capacity() >= 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Shrinks the capacity of the partition with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::with_capacity(100);
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    ///
    /// // shrink to at most 10 elements
    /// p.shrink_to(10);
    /// assert!(p.capacity() >= 3);
    /// assert!(p.capacity() <= 10);
    ///
    /// // shrink to at most 2 elements, but the partition
    /// // already contains 3 elements, so this has no effect
    /// p.shrink_to(2);
    /// assert!(p.capacity() >= 3);
    /// ```
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// into the partition.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    /// After calling `try_reserve`, capacity will be greater than or equal to
    /// `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    ///
    /// // reserve space for 10 more elements
    /// match p.try_reserve(10) {
    ///     Ok(()) => println!("Reservation successful"),
    ///     Err(e) => println!("Reservation failed: {:?}", e),
    /// }
    /// ```
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Tries to reserve the minimum capacity for at least `additional` more elements
    /// to be inserted into the partition.
    ///
    /// Unlike [`try_reserve`](Partition::try_reserve), this will not deliberately
    /// over-allocate to avoid frequent reallocations. After calling `try_reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional` if it returns
    /// `Ok(())`. Does nothing if the capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    ///
    /// // reserve exact space for 10 more elements
    /// match p.try_reserve_exact(10) {
    ///     Ok(()) => println!("Reservation successful"),
    ///     Err(e) => println!("Reservation failed: {:?}", e),
    /// }
    /// ```
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    /// Moves all the elements of `other` into this `Partition`, leaving `other` empty.
    ///
    /// Elements from `other.left()` are added to `self.left()` and elements from
    /// `other.right()` are added to `self.right()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    ///
    /// let mut partition1 = Partition::new();
    /// partition1.push_left(1);
    /// partition1.push_right(2);
    ///
    /// let mut partition2 = Partition::new();
    /// partition2.push_left(3);
    /// partition2.push_right(4);
    ///
    /// // append partition2 into partition1
    /// partition1.append(&mut partition2);
    ///
    /// // check left partition contains 1 and 3
    /// assert_eq!(partition1.left().len(), 2);
    /// assert!(partition1.left().contains(&1));
    /// assert!(partition1.left().contains(&3));
    ///
    /// // check right partition contains 2 and 4
    /// assert_eq!(partition1.right().len(), 2);
    /// assert!(partition1.right().contains(&2));
    /// assert!(partition1.right().contains(&4));
    ///
    /// // partition2 is now empty
    /// assert!(partition2.is_empty());
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        // reserve capacity for all elements at once
        self.reserve(other.len());

        // move elements from other.left() to self.left() (drain to avoid clone)
        let left_elements = other.drain_left();
        for element in left_elements {
            self.push_left(element);
        }

        // move elements from other.right() to self.right()
        let right_elements = other.drain_right();
        for element in right_elements {
            self.push_right(element);
        }

        // the other partition should now be empty
        debug_assert!(other.is_empty());
        debug_assert_eq!(other.left().len(), 0);
        debug_assert_eq!(other.right().len(), 0);
    }

    /// Clears the partition, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    ///
    /// p.clear();
    ///
    /// assert!(p.is_empty());
    /// assert_eq!(p.left(), &[]);
    /// assert_eq!(p.right(), &[]);
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
        self.partition = 0;
    }

    /// Retains only the elements in the left partition that satisfy the predicate.
    ///
    /// In other words, remove all elements `e` from the left partition such that
    /// `f(&e)` returns `false`. This method operates in place, visiting each element
    /// in the left partition exactly once and preserving the capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_left(3);
    /// p.push_left(4);
    /// p.push_right(5);
    /// p.push_right(6);
    ///
    /// // retain only even numbers in the left partition
    /// p.retain_left(|x| x % 2 == 0);
    ///
    /// // the left partition should only contain even numbers
    /// assert!(p.left().iter().all(|&x| x % 2 == 0));
    /// // right partition is unchanged
    /// assert_eq!(p.right().len(), 2);
    /// ```
    pub fn retain_left<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        if self.partition == 0 {
            return; // nothing to do
        }

        // use a two-pointer approach to track elements that should be kept
        let mut keep_idx = 0;

        // iterate through elements in the left partition
        for i in 0..self.partition {
            // keep elements that satisfy the predicate
            if f(&self.inner[i]) {
                if keep_idx != i {
                    // swap to move kept elements to the front
                    self.inner.swap(keep_idx, i);
                }
                keep_idx += 1;
            }
        }

        // if we kept fewer elements than we had, move right partition elements accordingly
        if keep_idx < self.partition {
            // determine how many elements we're removing
            let removed = self.partition - keep_idx;

            // move elements from the right partition to fill the gap
            for i in 0..self.right().len() {
                let right_idx = self.partition + i;
                let new_idx = keep_idx + i;

                if right_idx != new_idx {
                    // only swap if indices are different
                    self.inner.swap(right_idx, new_idx);
                }
            }

            // update partition index
            self.partition = keep_idx;

            // truncate the vector to the new size
            self.inner.truncate(self.inner.len() - removed);
        }
    }

    /// Retains only the elements in the right partition that satisfy the predicate.
    ///
    /// In other words, remove all elements `e` from the right partition such that
    /// `f(&e)` returns `false`. This method operates in place, visiting each element
    /// in the right partition exactly once and preserves the capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    /// p.push_right(4);
    /// p.push_right(5);
    /// p.push_right(6);
    ///
    /// // retain only odd numbers in the right partition
    /// p.retain_right(|x| x % 2 == 1);
    ///
    /// // the right partition should only contain odd numbers
    /// assert!(p.right().iter().all(|&x| x % 2 == 1));
    /// // left partition is unchanged
    /// assert_eq!(p.left().len(), 2);
    /// ```
    pub fn retain_right<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        if self.partition >= self.inner.len() {
            return; // nothing to do
        }

        // use regular Vec::retain for the right partition
        // first, create a view of the right partition
        let right_len = self.inner.len() - self.partition;
        let mut indices_to_remove = Vec::with_capacity(right_len);

        // find elements to remove
        for i in 0..right_len {
            let idx = self.partition + i;
            if !f(&self.inner[idx]) {
                indices_to_remove.push(idx);
            }
        }

        // remove elements in reverse order to avoid index shifting
        for &idx in indices_to_remove.iter().rev() {
            self.inner.swap_remove(idx);
        }
    }
}

impl<T: Copy> Partition<T> {
    /// Moves an element (if any) from the right partition to the left.
    ///
    /// Returns the moved element, or `None` if the right partition is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    ///
    /// p.push_right(1);
    /// assert_eq!(p.right(), &[1]);
    ///
    /// assert_eq!(p.move_to_left(), Some(1));
    /// assert_eq!(p.left(), &[1]);
    ///
    /// assert_eq!(p.move_to_left(), None);
    /// ```
    pub fn move_to_left(&mut self) -> Option<T> {
        if let Some(moved) = self.inner.get(self.partition) {
            self.partition += 1;
            Some(*moved)
        } else {
            None
        }
    }

    /// Moves an element (if any) from the left partition to the right.
    ///
    /// Returns the moved element, or `None` if the left partition is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    ///
    /// p.push_left(1);
    /// assert_eq!(p.left(), &[1]);
    ///
    /// assert_eq!(p.move_to_right(), Some(1));
    /// assert_eq!(p.right(), &[1]);
    ///
    /// assert_eq!(p.move_to_right(), None);
    /// ```
    pub fn move_to_right(&mut self) -> Option<T> {
        if self.partition > 0 {
            self.partition -= 1;
            Some(self.inner[self.partition])
        } else {
            None
        }
    }

    /// Returns an iterator that drains elements from the left partition to the right.
    ///
    /// Elements are moved from the left partition to the right partition as the iterator
    /// is consumed. Any elements not consumed by the time the iterator is dropped will
    /// still be moved to the right partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_left(2);
    /// p.push_right(3);
    ///
    /// let moved: Vec<_> = p.drain_to_right().collect();
    /// assert_eq!(moved.len(), 2);
    ///
    /// // elements are now on the right side
    /// assert_eq!(p.left(), &[]);
    /// assert_eq!(p.right().len(), 3);
    /// ```
    pub fn drain_to_right(&mut self) -> DrainToRight<T> {
        // first collect all left elements
        let elements = self.left().to_vec();

        // reset the left partition
        let old_partition = self.partition;
        self.partition = 0;

        // keep right elements in place
        let right_elements = self.inner.split_off(old_partition);
        self.inner = right_elements;

        // then add all the left elements to the right
        for item in &elements {
            self.push_right(*item);
        }

        DrainToRight { elements, index: 0 }
    }

    /// Returns an iterator that drains elements from the right partition to the left.
    ///
    /// Elements are moved from the right partition to the left partition as the iterator
    /// is consumed. Any elements not consumed by the time the iterator is dropped will
    /// still be moved to the left partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let mut p = Partition::new();
    /// p.push_left(1);
    /// p.push_right(2);
    /// p.push_right(3);
    ///
    /// let moved: Vec<_> = p.drain_to_left().collect();
    /// assert_eq!(moved.len(), 2);
    ///
    /// // elements are now on the left side
    /// assert_eq!(p.left().len(), 3);
    /// assert_eq!(p.right(), &[]);
    /// ```
    pub fn drain_to_left(&mut self) -> DrainToLeft<T> {
        // first collect all right elements
        let elements = self.right().to_vec();

        // reset the right partition
        self.inner.truncate(self.partition);

        // move all elements to the left
        for item in &elements {
            self.push_left(*item);
        }

        DrainToLeft { elements, index: 0 }
    }
}

/// A drain iterator that moves elements from the left to the right partition.
pub struct DrainToRight<T: Copy> {
    /// Elements from the left partition that will move to the right
    elements: Vec<T>,
    /// Current index in the elements vector
    index: usize,
}

impl<T: Copy> Iterator for DrainToRight<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.elements.len() {
            let val = self.elements[self.index];
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.elements.len() - self.index;
        (remaining, Some(remaining))
    }
}

/// A drain iterator that moves elements from the right to the left partition.
pub struct DrainToLeft<T: Copy> {
    /// Elements from the right partition that will move to the left
    elements: Vec<T>,
    /// Current index in the elements vector
    index: usize,
}

impl<T: Copy> Iterator for DrainToLeft<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.elements.len() {
            let val = self.elements[self.index];
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.elements.len() - self.index;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests;
