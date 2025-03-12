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
//! // Create a partition and add elements to both sides
//! let mut partition = Partition::new();
//! partition.push_left("apple");
//! partition.push_left("banana");
//! partition.push_right("cherry");
//! partition.push_right("date");
//!
//! // Sort elements for predictable assertions
//! // (remember: order is not preserved within partitions)
//! partition.left_mut().sort();
//! partition.right_mut().sort();
//!
//! // Access elements in each partition
//! assert_eq!(partition.left(), &["apple", "banana"]);
//! assert_eq!(partition.right(), &["cherry", "date"]);
//!
//! // Move elements between partitions
//! let moved = partition.move_to_right();
//! assert!(moved == Some("apple") || moved == Some("banana"));
//!
//! // Get both partitions at once with the convenient partitions() method
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
//! // Add some task IDs to the pending (left) and completed (right) lists
//! partition.push_left(101);
//! partition.push_left(102);
//! partition.push_left(103);
//! partition.push_right(201);
//! partition.push_right(202);
//!
//! // Process all pending tasks and move them to completed
//! println!("Processing pending tasks:");
//! for task_id in partition.drain_to_right() {
//!     println!("Processing task {}", task_id);
//!     // Tasks are moved to the right partition automatically
//! }
//!
//! // All tasks are now in the completed list
//! assert_eq!(partition.left().len(), 0);
//! assert_eq!(partition.right().len(), 5);
//!
//! // We can also archive completed tasks by moving to left (in a real app)
//! println!("Archiving old tasks:");
//! for task_id in partition.drain_to_left().take(2) {
//!     println!("Archiving task {}", task_id);
//!     // Even when we only process some items, all move to the destination
//! }
//!
//! // All tasks moved to the left (archived) partition
//! assert_eq!(partition.left().len(), 5);
//! assert_eq!(partition.right().len(), 0);
//! ```
//!
//! ### Using With Custom Types
//!
//! ```rust
//! use bose_einstein::Partition;
//!
//! // A simple task type for demonstration
//! #[derive(Debug, Clone, Copy, PartialEq)]
//! struct Task {
//!     id: u32,
//!     is_important: bool,
//! }
//!
//! // Use partition to organize tasks by importance
//! let mut tasks = Partition::new();
//!
//! // Add some important tasks to the left partition
//! tasks.push_left(Task { id: 1, is_important: true });
//! tasks.push_left(Task { id: 2, is_important: true });
//!
//! // Add some regular tasks to the right partition
//! tasks.push_right(Task { id: 3, is_important: false });
//! tasks.push_right(Task { id: 4, is_important: false });
//!
//! // Check that we have the correct number of tasks in each partition
//! assert_eq!(tasks.left().len(), 2);
//! assert_eq!(tasks.right().len(), 2);
//!
//! // We can move a task from important to regular
//! let moved_task = tasks.move_to_right();
//! assert!(moved_task.is_some());
//! assert!(moved_task.unwrap().is_important); // Was in the important partition
//!
//! // Now we have one less important task
//! assert_eq!(tasks.left().len(), 1);
//! assert_eq!(tasks.right().len(), 3);
//!
//! // We can access and modify tasks in each partition
//! // (In a real application we might use more sophisticated filtering)
//! let contains_task1 = tasks.left().iter().any(|task| task.id == 1) ||
//!                       tasks.right().iter().any(|task| task.id == 1);
//! assert!(contains_task1, "Task 1 should be in either partition");
//! ```
#![no_std]

extern crate alloc;

use alloc::vec::{self, Vec};
use core::{fmt, mem};

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
#[derive(Clone, Default)]
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

impl<T> Partition<T> {
    /// Creates a new empty partition.
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            partition: 0,
        }
    }

    /// Creates a new empty partition with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            partition: 0,
        }
    }

    /// Decomposes a `Partition<T>` into its raw parts.
    ///
    /// Returns the underlying vector and partition index.
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
    /// Returns `None` if `partition` is greater than `vec.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// // Create a partition from existing data
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // First 2 elements will be in the left partition
    ///
    /// let p = Partition::from_raw_parts(vec, partition).unwrap();
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    ///
    /// If the partition index is invalid:
    ///
    /// ```
    /// use bose_einstein::Partition;
    /// let vec = vec![1, 2, 3];
    /// let partition = 4; // Invalid: beyond the length of the vector
    ///
    /// assert!(Partition::from_raw_parts(vec, partition).is_none());
    /// ```
    pub fn from_raw_parts(inner: Vec<T>, partition: usize) -> Option<Self> {
        if partition <= inner.len() {
            Some(unsafe { Self::from_raw_parts_unchecked(inner, partition) })
        } else {
            None
        }
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
    /// // Safe usage: partition index is valid
    /// let vec = vec![1, 2, 3, 4];
    /// let partition = 2; // First 2 elements will be in the left partition
    ///
    /// let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    /// assert_eq!(p.left(), &[1, 2]);
    /// assert_eq!(p.right(), &[3, 4]);
    /// ```
    pub unsafe fn from_raw_parts_unchecked(inner: Vec<T>, partition: usize) -> Self {
        Self { inner, partition }
    }

    /// Returns both partitions as a tuple of slices.
    ///
    /// This is a convenience method that returns both the left and right
    /// partitions at once.
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
    /// partitions at once, allowing mutation of both sides simultaneously.
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
    /// if !left.is_empty() {
    ///     left[0] = 10;
    /// }
    /// if !right.is_empty() {
    ///     right[0] = 20;
    /// }
    ///
    /// assert_eq!(p.left(), &[10]);
    /// assert_eq!(p.right(), &[20]);
    /// ```
    pub fn partitions_mut(&mut self) -> (&mut [T], &mut [T]) {
        // Safe because self.partition is always <= self.inner.len()
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
    /// p.left_mut()[0] = 2;
    ///
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
    /// p.right_mut()[0] = 2;
    ///
    /// assert_eq!(p.pop_right(), Some(2));
    /// ```
    pub fn right_mut(&mut self) -> &mut [T] {
        // SAFETY: self.partition <= self.inner.len()
        unsafe { self.inner.get_unchecked_mut(self.partition..) }
    }

    /// Pushes a value into the left partition.
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

    /// Removes and returns an element (if any) from the left partition.
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

    /// Pushes a value into the right partition.
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

    /// Removes and returns an element (if any) from the right partition.
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

    /// Returns an iterator that drains all elements from the left partition.
    ///
    /// This is useful for efficiently consuming all elements on the left side
    /// without deallocating the underlying storage.
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
        // Update the partition index first
        let old_partition = self.partition;
        self.partition = 0;

        // Then drain the elements
        self.inner.drain(0..old_partition)
    }

    /// Returns an iterator that drains all elements from the right partition.
    ///
    /// This is useful for efficiently consuming all elements on the right side
    /// without deallocating the underlying storage.
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
        // Simply drain from partition to end
        self.inner.drain(self.partition..)
    }

    /// Returns the total number of elements in the partition.
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

    /// Clears the partition, removing all values.
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
    /// // Elements are now on the right side
    /// assert_eq!(p.left(), &[]);
    /// assert_eq!(p.right().len(), 3);
    /// ```
    pub fn drain_to_right(&mut self) -> DrainToRight<T> {
        // First collect all left elements
        let elements = self.left().to_vec();

        // Reset the left partition
        let old_partition = self.partition;
        self.partition = 0;

        // Keep right elements in place
        let right_elements = self.inner.split_off(old_partition);
        self.inner = right_elements;

        // Then add all the left elements to the right
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
    /// // Elements are now on the left side
    /// assert_eq!(p.left().len(), 3);
    /// assert_eq!(p.right(), &[]);
    /// ```
    pub fn drain_to_left(&mut self) -> DrainToLeft<T> {
        // First collect all right elements
        let elements = self.right().to_vec();

        // Reset the right partition
        self.inner.truncate(self.partition);

        // Move all elements to the left
        for item in &elements {
            self.push_left(*item);
        }

        DrainToLeft { elements, index: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use alloc::vec;
    use core::fmt::Debug;

    // Helper function to check if two collections have the same elements
    fn check_set_equality<T, I1, I2>(left: I1, right: I2)
    where
        T: Eq + Ord + Debug,
        I1: IntoIterator<Item = T>,
        I2: IntoIterator<Item = T>,
    {
        let left_set: BTreeSet<_> = left.into_iter().collect();
        let right_set: BTreeSet<_> = right.into_iter().collect();
        assert_eq!(left_set, right_set);
    }

    #[test]
    fn test_new() {
        let p: Partition<i32> = Partition::new();
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);
    }

    #[test]
    fn test_with_capacity() {
        let p: Partition<i32> = Partition::with_capacity(10);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);
        // Not much else we can test as capacity is an implementation detail
    }

    #[test]
    fn test_push_left() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);

        // Order-agnostic check using our helper
        check_set_equality(p.left().iter().copied(), [1, 2]);
    }

    #[test]
    fn test_push_right() {
        let mut p = Partition::new();
        p.push_right(1);
        p.push_right(2);

        // Order-agnostic check using our helper
        check_set_equality(p.right().iter().copied(), [1, 2]);
    }

    #[test]
    fn test_mixed_push() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_right(2);
        p.push_left(3);
        p.push_right(4);

        // Order-agnostic checks using our helpers
        check_set_equality(p.left().iter().copied(), [1, 3]);
        check_set_equality(p.right().iter().copied(), [2, 4]);
    }

    #[test]
    fn test_pop_left() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);

        // We can't guarantee which element will be popped first
        let val = p.pop_left();
        assert!(val == Some(1) || val == Some(2));

        // We know which element should remain
        let remaining = if val == Some(1) { 2 } else { 1 };

        // Order-agnostic check using our helper
        check_set_equality(p.left(), &[remaining]);

        // Pop the remaining element
        assert_eq!(p.pop_left(), Some(remaining));
        assert_eq!(p.left(), &[]);
        assert_eq!(p.pop_left(), None);
    }

    #[test]
    fn test_pop_right() {
        let mut p = Partition::new();
        p.push_right(1);
        p.push_right(2);

        // Pop behavior for right partition is LIFO (stack-like)
        // This is actually part of the implementation contract so we test it specifically
        assert_eq!(p.pop_right(), Some(2));

        // Order-agnostic check using our helper
        check_set_equality(p.right(), &[1]);

        assert_eq!(p.pop_right(), Some(1));
        assert_eq!(p.right(), &[]);
        assert_eq!(p.pop_right(), None);
    }

    #[test]
    fn test_left_and_right() {
        let mut p = Partition::new();
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);

        p.push_left(1);
        p.push_right(2);

        // Order-agnostic checks using our helpers
        check_set_equality(p.left(), &[1]);
        check_set_equality(p.right(), &[2]);
    }

    #[test]
    fn test_left_mut_and_right_mut() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_right(2);

        // We need to modify all elements since we don't know the order
        for item in p.left_mut() {
            *item = 3;
        }
        for item in p.right_mut() {
            *item = 4;
        }

        // Order-agnostic checks using our helpers
        check_set_equality(p.left(), &[3]);
        check_set_equality(p.right(), &[4]);
    }

    #[test]
    fn test_move_to_left() {
        let mut p = Partition::new();
        p.push_right(1);
        p.push_right(2);

        // The right partition behaves like a stack (LIFO) where push_right appends
        // and move_to_left takes from the beginning (oldest element)
        assert_eq!(p.move_to_left(), Some(1));

        // Check the partitions using our helpers
        check_set_equality(p.left(), &[1]);
        check_set_equality(p.right(), &[2]);

        // Move the next element
        assert_eq!(p.move_to_left(), Some(2));

        // Verify final state
        check_set_equality(p.left(), &[1, 2]);
        assert_eq!(p.right(), &[]);

        // No more elements to move
        assert_eq!(p.move_to_left(), None);
    }

    #[test]
    fn test_move_to_right() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);

        // We can't guarantee which element will be moved first
        let val = p.move_to_right();
        assert!(val == Some(1) || val == Some(2));

        // We can compute which element should remain in left and which moved to right
        let remaining = if val == Some(1) { 2 } else { 1 };
        let moved = if val == Some(1) { 1 } else { 2 };

        // Check both partitions using our helpers
        check_set_equality(p.left(), &[remaining]);
        check_set_equality(p.right(), &[moved]);

        // Move the remaining element
        assert_eq!(p.move_to_right(), Some(remaining));
        assert_eq!(p.left(), &[]);

        // Check right partition has both elements
        check_set_equality(p.right(), &[1, 2]);

        // No more elements to move
        assert_eq!(p.move_to_right(), None);
    }

    #[test]
    fn test_drain_left() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);

        // Collect drained elements
        let drained: Vec<_> = p.drain_left().collect();
        assert_eq!(drained.len(), 2);

        assert_eq!(p.left(), &[]);

        // Check right partition
        assert_eq!(p.right(), &[3]);
    }

    #[test]
    fn test_drain_right() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_right(2);
        p.push_right(3);

        // Collect drained elements
        let drained: Vec<_> = p.drain_right().collect();
        assert_eq!(drained.len(), 2);

        assert_eq!(p.right(), &[]);

        // Check left partition
        assert_eq!(p.left(), &[1]);
    }

    #[test]
    fn test_drain_empty() {
        let mut p: Partition<i32> = Partition::new();

        let drained: Vec<_> = p.drain_left().collect();
        assert_eq!(drained, vec![]);

        let drained: Vec<_> = p.drain_right().collect();
        assert_eq!(drained, vec![]);
    }

    #[test]
    fn test_drain_to_right() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);

        // Drain left elements to right - store elements to check them
        let moved: Vec<_> = p.drain_to_right().collect();

        // Verify we got the expected number of elements
        assert_eq!(moved.len(), 2);

        // Verify partitions
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right().len(), 3);

        // All elements should now be in the right partition
        let right_vals: BTreeSet<_> = p.right().iter().copied().collect();
        assert!(right_vals.contains(&1));
        assert!(right_vals.contains(&2));
        assert!(right_vals.contains(&3));
    }

    #[test]
    fn test_drain_to_left() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_right(2);
        p.push_right(3);

        // Drain right elements to left - store elements to check them
        let moved: Vec<_> = p.drain_to_left().collect();

        // Verify we got the expected number of elements
        assert_eq!(moved.len(), 2);

        // Verify partitions
        assert_eq!(p.right(), &[]);
        assert_eq!(p.left().len(), 3);

        // All elements should now be in the left partition
        let left_vals: BTreeSet<_> = p.left().iter().copied().collect();
        assert!(left_vals.contains(&1));
        assert!(left_vals.contains(&2));
        assert!(left_vals.contains(&3));
    }

    #[test]
    fn test_drain_iterator_size_hint() {
        // Test drain_to_right size_hint
        {
            let mut p = Partition::new();
            p.push_left(1);
            p.push_left(2);

            let iter = p.drain_to_right();
            assert_eq!(iter.size_hint(), (2, Some(2)));
        }

        // Test drain_to_left size_hint
        {
            let mut p = Partition::new();
            p.push_right(1);
            p.push_right(2);

            let iter = p.drain_to_left();
            assert_eq!(iter.size_hint(), (2, Some(2)));
        }
    }

    #[test]
    fn test_drain_partially_consumed() {
        // Test partially consumed drain_to_right
        {
            let mut p = Partition::new();
            p.push_left(1);
            p.push_left(2);
            p.push_left(3);

            // Get a count of elements before we drain
            let left_count = p.left().len();

            // Only take one element from the iterator
            {
                let mut iter = p.drain_to_right();
                let first = iter.next();
                assert!(first.is_some());
                // Let the iterator drop here - should move remaining elements
            }

            // All elements should now be in the right partition
            assert_eq!(p.left(), &[]);
            assert_eq!(p.right().len(), left_count);
        }

        // Test partially consumed drain_to_left
        {
            let mut p = Partition::new();
            p.push_right(1);
            p.push_right(2);
            p.push_right(3);

            // Get a count of elements before we drain
            let right_count = p.right().len();

            // Only take one element from the iterator
            {
                let mut iter = p.drain_to_left();
                let first = iter.next();
                assert!(first.is_some());
                // Let the iterator drop here - should move remaining elements
            }

            // All elements should now be in the left partition
            assert_eq!(p.right(), &[]);
            assert_eq!(p.left().len(), right_count);
        }
    }

    #[test]
    fn test_drain_immediate_drop() {
        // Test immediate drop of drain_to_right (same as calling drain_to_right())
        {
            let mut p = Partition::new();
            p.push_left(1);
            p.push_left(2);

            // Create and immediately drop the iterator
            drop(p.drain_to_right());

            // All elements should now be in the right partition
            assert_eq!(p.left(), &[]);
            assert_eq!(p.right().len(), 2);
        }

        // Test immediate drop of drain_to_left (same as calling drain_to_left())
        {
            let mut p = Partition::new();
            p.push_right(1);
            p.push_right(2);

            // Create and immediately drop the iterator
            drop(p.drain_to_left());

            // All elements should now be in the left partition
            assert_eq!(p.right(), &[]);
            assert_eq!(p.left().len(), 2);
        }
    }

    #[test]
    fn test_complex_operations() {
        let mut p = Partition::new();

        // Push items
        p.push_left(1);
        p.push_right(2);
        p.push_left(3);

        // Verify using our helper
        check_set_equality(p.left(), &[1, 3]);
        check_set_equality(p.right(), &[2]);

        // Move right
        let val = p.move_to_right();
        assert!(val == Some(1) || val == Some(3));

        // Move left
        let moved_left = p.move_to_left();
        assert!(moved_left.is_some());

        // Count elements in each partition
        let left_count = p.left().len();
        let right_count = p.right().len();
        assert_eq!(left_count + right_count, 3);

        // Drain left to right
        let drained_left: Vec<_> = p.drain_to_right().collect();
        assert_eq!(drained_left.len(), left_count);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right().len(), 3);

        // Drain right to left
        let drained_right: Vec<_> = p.drain_to_left().collect();
        assert_eq!(drained_right.len(), 3);
        assert_eq!(p.right(), &[]);
        assert_eq!(p.left().len(), 3);
    }

    #[test]
    fn test_large_partition_with_interleaved_operations() {
        // Create a larger partition with more complex operations
        let mut p = Partition::new();

        // Push lots of elements in different patterns
        for i in 0..50 {
            if i % 3 == 0 {
                p.push_left(i);
            } else {
                p.push_right(i);
            }
        }

        // Expected values in each partition based on our insertion pattern
        let expected_left: Vec<_> = (0..50).filter(|&i| i % 3 == 0).collect();
        let expected_right: Vec<_> = (0..50).filter(|&i| i % 3 != 0).collect();

        // Verify the partitions have correct elements
        let left_vec: Vec<_> = p.left().to_vec();
        let right_vec: Vec<_> = p.right().to_vec();

        // Check that left contains all expected values
        for &val in &expected_left {
            assert!(left_vec.contains(&val));
        }
        assert_eq!(left_vec.len(), expected_left.len());

        // Check that right contains all expected values
        for &val in &expected_right {
            assert!(right_vec.contains(&val));
        }
        assert_eq!(right_vec.len(), expected_right.len());

        // Move half of the left elements to the right
        let left_count = p.left().len();
        let mut moved_to_right = Vec::new();

        for _ in 0..(left_count / 2) {
            if let Some(val) = p.move_to_right() {
                moved_to_right.push(val);
            }
        }

        // Verify expected counts
        assert_eq!(p.left().len(), left_count - moved_to_right.len());

        // Move items from right back to left
        for _ in 0..5 {
            p.move_to_left();
        }

        // Add more elements
        for i in 50..60 {
            if i % 2 == 0 {
                p.push_left(i);
            } else {
                p.push_right(i);
            }
        }

        // Drain all elements from partitions
        let left_elements: Vec<_> = p.drain_left().collect();
        let right_elements: Vec<_> = p.drain_right().collect();

        // Verify drained elements total to expected count
        assert_eq!(left_elements.len() + right_elements.len(), 60);

        // Verify partitions are empty
        assert!(p.left().is_empty());
        assert!(p.right().is_empty());
    }

    #[test]
    fn test_random_operation_sequence() {
        let mut p = Partition::new();

        // Maintain shadow collections to verify behavior
        let mut expected_left = Vec::new();
        let mut expected_right = Vec::new();

        // Mixed Sequence 1: Add elements to both sides
        for i in 0..10 {
            if i % 2 == 0 {
                p.push_left(i);
                expected_left.push(i);
            } else {
                p.push_right(i);
                expected_right.push(i);
            }
        }

        // Verify state - direct element-by-element comparison
        let left_vec: Vec<_> = p.left().to_vec();
        let right_vec: Vec<_> = p.right().to_vec();
        assert_eq!(left_vec.len(), expected_left.len());
        assert_eq!(right_vec.len(), expected_right.len());

        for &val in &expected_left {
            assert!(left_vec.contains(&val));
        }

        for &val in &expected_right {
            assert!(right_vec.contains(&val));
        }

        // Mixed Sequence 2: Move elements between partitions
        // Move 2 elements from left to right
        for _ in 0..2 {
            if let Some(val) = p.move_to_right() {
                expected_left.retain(|&x| x != val);
                expected_right.push(val);
            }
        }

        // Move 1 element from right to left
        if let Some(val) = p.move_to_left() {
            expected_right.retain(|&x| x != val);
            expected_left.push(val);
        }

        // Verify state - direct comparison
        let left_vec: Vec<_> = p.left().to_vec();
        let right_vec: Vec<_> = p.right().to_vec();
        assert_eq!(left_vec.len(), expected_left.len());
        assert_eq!(right_vec.len(), expected_right.len());

        for &val in &expected_left {
            assert!(left_vec.contains(&val));
        }

        for &val in &expected_right {
            assert!(right_vec.contains(&val));
        }

        // Mixed Sequence 3: Pop elements
        if let Some(val) = p.pop_left() {
            expected_left.retain(|&x| x != val);
        }

        if let Some(val) = p.pop_right() {
            expected_right.retain(|&x| x != val);
        }

        // Verify state - direct comparison
        let left_vec: Vec<_> = p.left().to_vec();
        let right_vec: Vec<_> = p.right().to_vec();
        assert_eq!(left_vec.len(), expected_left.len());
        assert_eq!(right_vec.len(), expected_right.len());

        for &val in &expected_left {
            assert!(left_vec.contains(&val));
        }

        for &val in &expected_right {
            assert!(right_vec.contains(&val));
        }

        // Mixed Sequence 4: Add more elements and drain
        for i in 10..15 {
            if i % 2 == 0 {
                p.push_left(i);
                expected_left.push(i);
            } else {
                p.push_right(i);
                expected_right.push(i);
            }
        }

        // Get counts before draining
        let left_count = expected_left.len();
        let right_count = expected_right.len();

        // Drain to right and verify
        let drained_left: Vec<_> = p.drain_to_right().collect();
        assert_eq!(drained_left.len(), left_count);

        // All elements should now be in the right partition
        assert!(p.left().is_empty());
        assert_eq!(p.right().len(), left_count + right_count);
    }

    #[test]
    fn test_push_left_swap_behavior() {
        // This test verifies the internal behavior of push_left,
        // which swaps elements with the partition point
        let mut p = Partition::new();

        // First, set up a known state
        p.push_right(1);
        p.push_right(2);
        p.push_right(3);

        // The internal array should now be [1, 2, 3] with partition=0
        assert_eq!(p.left(), &[]);
        check_set_equality(p.right().iter().copied(), [1, 2, 3]);

        // Now when we push_left(4), it should:
        // 1. Swap 4 with element at partition point (which is 1)
        // 2. Push the swapped element (1) to the end
        // 3. Increment partition point
        p.push_left(4);

        // Expected state:
        // - Left contains [4]
        // - Right contains [2, 3, 1]
        check_set_equality(p.left().iter().copied(), [4]);

        // Check that we have exactly the 3 expected elements in right
        let right_set: BTreeSet<_> = p.right().iter().copied().collect();
        assert_eq!(right_set.len(), 3);
        assert!(right_set.contains(&1));
        assert!(right_set.contains(&2));
        assert!(right_set.contains(&3));

        // Now push_left(5) - should swap with element at partition (now 2)
        p.push_left(5);

        // Left should now have [4, 5]
        check_set_equality(p.left().iter().copied(), [4, 5]);

        // Right should have [3, 1, 2] - exactly these elements, no more or less
        let right_set: BTreeSet<_> = p.right().iter().copied().collect();
        assert_eq!(right_set.len(), 3);
        assert!(right_set.contains(&1));
        assert!(right_set.contains(&2));
        assert!(right_set.contains(&3));
    }

    #[test]
    fn test_partitions() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);
        p.push_right(4);

        // Get both partitions at once
        let (left, right) = p.partitions();

        // Check that they contain the expected elements
        check_set_equality(left.iter().copied(), [1, 2]);
        check_set_equality(right.iter().copied(), [3, 4]);
    }

    #[test]
    fn test_partitions_mut() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);
        p.push_right(4);

        // Get mutable access to both partitions at once
        let (left_mut, right_mut) = p.partitions_mut();

        // Modify elements in both partitions
        for item in left_mut.iter_mut() {
            *item *= 10;
        }

        for item in right_mut.iter_mut() {
            *item *= 100;
        }

        // Check that modifications were applied
        check_set_equality(p.left().iter().copied(), [10, 20]);
        check_set_equality(p.right().iter().copied(), [300, 400]);
    }

    #[test]
    fn test_len() {
        let mut p = Partition::new();
        assert_eq!(p.len(), 0);

        p.push_left(1);
        assert_eq!(p.len(), 1);

        p.push_right(2);
        assert_eq!(p.len(), 2);

        p.push_left(3);
        assert_eq!(p.len(), 3);

        // Check len after removing elements
        p.pop_left();
        assert_eq!(p.len(), 2);

        p.pop_right();
        assert_eq!(p.len(), 1);

        // Should be empty after all elements are removed
        p.pop_left();
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_is_empty() {
        let mut p = Partition::new();
        assert!(p.is_empty());

        p.push_left(1);
        assert!(!p.is_empty());

        p.pop_left();
        assert!(p.is_empty());

        // Check with right side
        p.push_right(2);
        assert!(!p.is_empty());

        p.pop_right();
        assert!(p.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);
        p.push_right(4);

        assert_eq!(p.len(), 4);
        assert_eq!(p.left().len(), 2);
        assert_eq!(p.right().len(), 2);

        // Clear the partition
        p.clear();

        // Everything should be empty
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.left().len(), 0);
        assert_eq!(p.right().len(), 0);

        // Should still work correctly after clearing
        p.push_left(5);
        p.push_right(6);

        assert_eq!(p.len(), 2);
        assert_eq!(p.left().len(), 1);
        assert_eq!(p.right().len(), 1);
        check_set_equality(p.left().iter().copied(), [5]);
        check_set_equality(p.right().iter().copied(), [6]);
    }

    #[test]
    fn test_to_raw_parts() {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_right(3);

        // Verify initial state
        assert_eq!(p.left().len(), 2);
        assert_eq!(p.right().len(), 1);

        // Decompose into raw parts
        let (vec, partition) = p.to_raw_parts();

        // Check results
        assert_eq!(vec.len(), 3);
        assert_eq!(partition, 2);

        // Verify the contents are all present
        let mut elements = vec.clone();
        elements.sort();
        assert_eq!(elements, vec![1, 2, 3]);
    }

    #[test]
    fn test_from_raw_parts() {
        // Valid case
        let vec = vec![10, 20, 30, 40];
        let partition = 2;

        let p = Partition::from_raw_parts(vec, partition).unwrap();
        assert_eq!(p.left(), &[10, 20]);
        assert_eq!(p.right(), &[30, 40]);

        // Edge case: empty vector
        let vec = Vec::<i32>::new();
        let partition = 0;

        let p = Partition::from_raw_parts(vec, partition).unwrap();
        assert!(p.left().is_empty());
        assert!(p.right().is_empty());

        // Edge case: all elements in left partition
        let vec = vec![1, 2, 3];
        let partition = 3;

        let p = Partition::from_raw_parts(vec, partition).unwrap();
        assert_eq!(p.left(), &[1, 2, 3]);
        assert!(p.right().is_empty());

        // Edge case: all elements in right partition
        let vec = vec![1, 2, 3];
        let partition = 0;

        let p = Partition::from_raw_parts(vec, partition).unwrap();
        assert!(p.left().is_empty());
        assert_eq!(p.right(), &[1, 2, 3]);

        // Invalid case
        let vec = vec![1, 2, 3];
        let partition = 4; // Beyond the vector length

        let result = Partition::from_raw_parts(vec, partition);
        assert!(result.is_none());
    }

    #[test]
    fn test_from_raw_parts_unchecked() {
        // Valid case
        let vec = vec![10, 20, 30, 40];
        let partition = 2;

        let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
        assert_eq!(p.left(), &[10, 20]);
        assert_eq!(p.right(), &[30, 40]);

        // Edge case: empty vector
        let vec = Vec::<i32>::new();
        let partition = 0;

        let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
        assert!(p.left().is_empty());
        assert!(p.right().is_empty());

        // Edge case: all elements in left partition
        let vec = vec![1, 2, 3];
        let partition = 3;

        let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
        assert_eq!(p.left(), &[1, 2, 3]);
        assert!(p.right().is_empty());

        // Edge case: all elements in right partition
        let vec = vec![1, 2, 3];
        let partition = 0;

        let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
        assert!(p.left().is_empty());
        assert_eq!(p.right(), &[1, 2, 3]);

        // We don't test the invalid case (partition > vec.len()) since that would be undefined behavior
    }

    #[test]
    fn test_raw_parts_roundtrip() {
        // Create a partition with known state
        let mut original = Partition::new();
        original.push_left(1);
        original.push_left(2);
        original.push_right(3);
        original.push_right(4);

        // Convert to raw parts
        let (vec, partition) = original.to_raw_parts();

        // Reconstruct from raw parts
        let reconstructed = Partition::from_raw_parts(vec, partition).unwrap();

        // Verify left and right partitions match
        // Use set equality since order within partitions might vary
        check_set_equality(reconstructed.left().iter().copied(), [1, 2]);
        check_set_equality(reconstructed.right().iter().copied(), [3, 4]);
    }
}
