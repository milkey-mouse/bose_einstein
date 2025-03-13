use super::*; // Import everything from the parent module
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use core::fmt::Debug;
use core::ops::{Deref, DerefMut};
use rand::prelude::*;
use std::panic::AssertUnwindSafe;

#[cfg(feature = "allocator_api")]
use alloc::alloc::Global;

#[cfg(feature = "allocator_api")]
mod allocator_tests {
    use super::*;

    // Direct tests against the parent Partition implementation to test allocator-specific features
    #[test]
    fn test_new_in() {
        let p: super::super::Partition<i32, Global> = super::super::Partition::new_in(Global);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);
    }

    #[test]
    fn test_with_capacity_in() {
        let capacity = 10;
        let p: super::super::Partition<i32, Global> =
            super::super::Partition::with_capacity_in(capacity, Global);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);
        assert!(p.capacity() >= capacity);
    }

    #[test]
    fn test_from_raw_parts_with_allocator() {
        // Create a vector with the global allocator
        let vec: Vec<i32, Global> = Vec::new_in(Global);

        // Create a vector with items using the global allocator
        let mut vec_with_items: Vec<i32, Global> = Vec::with_capacity_in(4, Global);
        vec_with_items.push(1);
        vec_with_items.push(2);
        vec_with_items.push(3);
        vec_with_items.push(4);

        // Empty vector
        let p = super::super::Partition::from_raw_parts(vec, 0);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);

        // Vector with items
        let p = super::super::Partition::from_raw_parts(vec_with_items, 2);
        assert_eq!(p.left(), &[1, 2]);
        assert_eq!(p.right(), &[3, 4]);
    }

    #[test]
    fn test_from_raw_parts_unchecked_with_allocator() {
        // Create a vector with items using the global allocator
        let mut vec_with_items: Vec<i32, Global> = Vec::with_capacity_in(4, Global);
        vec_with_items.push(1);
        vec_with_items.push(2);
        vec_with_items.push(3);
        vec_with_items.push(4);

        // Safe usage
        let p = unsafe { super::super::Partition::from_raw_parts_unchecked(vec_with_items, 2) };
        assert_eq!(p.left(), &[1, 2]);
        assert_eq!(p.right(), &[3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_from_raw_parts_panic_with_allocator() {
        // Create a vector with items using the global allocator
        let mut vec: Vec<i32, Global> = Vec::with_capacity_in(3, Global);
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // This should panic - partition index is out of bounds
        let _p = super::super::Partition::from_raw_parts(vec, 4);
    }
}

// Shadow the parent's Partition with our own that shuffles on mutation
#[derive(Clone)]
pub struct Partition<T>(super::Partition<T>);

// Implement Debug by forwarding to the inner Partition
impl<T: Debug> Debug for Partition<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

// Implement Default for our test Partition
impl<T> Default for Partition<T> {
    fn default() -> Self {
        Self(super::Partition::new())
    }
}

// Constructor methods
impl<T> Partition<T> {
    pub fn new() -> Self {
        Self(super::Partition::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(super::Partition::with_capacity(capacity))
    }

    // Static methods need to be explicitly forwarded
    pub fn from_raw_parts(inner: Vec<T>, partition: usize) -> Self {
        Self(super::Partition::from_raw_parts(inner, partition))
    }

    pub unsafe fn from_raw_parts_unchecked(inner: Vec<T>, partition: usize) -> Self {
        Self(unsafe { super::Partition::from_raw_parts_unchecked(inner, partition) })
    }

    // Forward to_raw_parts to avoid move issues
    pub fn to_raw_parts(self) -> (Vec<T>, usize) {
        let (vec, partition) = self.0.to_raw_parts();
        (vec, partition)
    }
}

// Implement DerefMut to shuffle on mutation
impl<T> DerefMut for Partition<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Shuffle both partitions unconditionally
        let mut rng = rand::thread_rng();
        self.0.left_mut().shuffle(&mut rng);
        self.0.right_mut().shuffle(&mut rng);
        &mut self.0
    }
}

// Implement Deref for immutable access
impl<T> Deref for Partition<T> {
    type Target = super::Partition<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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

// Test to demonstrate that partition order shouldn't be relied upon
#[test]
fn test_partition_order_independence() {
    // Create several partitions and add the same elements in different orders
    // These will automatically shuffle after every mutation thanks to DerefMut!
    let mut p1 = Partition::new();
    let mut p2 = Partition::new();
    let mut p3 = Partition::new();

    // Different insertion orders
    p1.push_left(1);
    p1.push_left(2);
    p1.push_left(3);

    p2.push_left(3);
    p2.push_left(1);
    p2.push_left(2);

    p3.push_left(2);
    p3.push_left(3);
    p3.push_left(1);

    // All should contain the same elements, regardless of insertion order
    check_set_equality(p1.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p2.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p3.left().iter().copied(), [1, 2, 3]);

    // No need to explicitly shuffle - it happens automatically on mutation!

    // Still contains the same elements
    check_set_equality(p1.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p2.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p3.left().iter().copied(), [1, 2, 3]);
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
    assert!(p.capacity() >= 10);
}

#[test]
fn test_capacity() {
    let mut p: Partition<i32> = Partition::new();
    let initial_capacity = p.capacity();

    // Add elements until we exceed the initial capacity
    for i in 0..initial_capacity + 10 {
        p.push_left(i as i32);
    }

    // Capacity should have increased
    assert!(p.capacity() > initial_capacity);
}

#[test]
fn test_reserve() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    let current_len = p.len();
    p.reserve(20);

    // Capacity should be at least current length + 20
    assert!(p.capacity() >= current_len + 20);

    // Should still have the same elements
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);
}

#[test]
fn test_reserve_exact() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    let current_len = p.len();
    let additional = 15;
    p.reserve_exact(additional);

    // Capacity should be at least current length + additional
    assert!(p.capacity() >= current_len + additional);

    // Should still have the same elements
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);
}

#[test]
fn test_shrink_to_fit() {
    // Create a partition with larger capacity
    let mut p: Partition<i32> = Partition::with_capacity(100);

    // Add a few elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // Verify we have excess capacity
    assert!(p.capacity() > 3);

    // Shrink the capacity
    p.shrink_to_fit();

    // Should still have at least enough for our elements
    assert!(p.capacity() >= 3);

    // Original elements should be preserved
    assert_eq!(p.len(), 3);
    check_set_equality(p.left().iter().copied(), [1, 2]);
    check_set_equality(p.right().iter().copied(), [3]);
}

#[test]
fn test_shrink_to() {
    // Create a partition with larger capacity
    let mut p: Partition<i32> = Partition::with_capacity(100);

    // Add a few elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // Verify we have excess capacity
    assert!(p.capacity() > 3);

    // Shrink to a capacity of at most 10
    p.shrink_to(10);

    // Should be between our length and the limit
    assert!(p.capacity() >= 3);
    assert!(p.capacity() <= 10);

    // Original elements should be preserved
    assert_eq!(p.len(), 3);
    check_set_equality(p.left().iter().copied(), [1, 2]);
    check_set_equality(p.right().iter().copied(), [3]);

    // Shrinking to less than length should still keep enough capacity for all elements
    p.shrink_to(2); // Our length is 3, so it should maintain capacity >= 3
    assert!(p.capacity() >= 3);
}

#[test]
fn test_try_reserve() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    // This should succeed for reasonable sizes
    let result = p.try_reserve(10);
    assert!(result.is_ok());

    // After successful reservation, capacity should be increased
    assert!(p.capacity() >= p.len() + 10);

    // Original elements should be preserved
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);

    // We don't test allocation failure cases since they're hard to reliably
    // trigger in a test environment
}

#[test]
fn test_try_reserve_exact() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    // This should succeed for reasonable sizes
    let result = p.try_reserve_exact(10);
    assert!(result.is_ok());

    // After successful reservation, capacity should be increased
    assert!(p.capacity() >= p.len() + 10);

    // Original elements should be preserved
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);
}

#[test]
fn test_append() {
    // Create two partitions
    let mut p1 = Partition::new();
    p1.push_left(1);
    p1.push_left(2);
    p1.push_right(3);

    let mut p2 = Partition::new();
    p2.push_left(4);
    p2.push_right(5);
    p2.push_right(6);

    // Save the original lengths
    let p1_left_len = p1.left().len();
    let p1_right_len = p1.right().len();
    let p2_left_len = p2.left().len();
    let p2_right_len = p2.right().len();

    // Append p2 to p1
    p1.append(&mut p2);

    // Check that p1 has all elements from both
    assert_eq!(p1.left().len(), p1_left_len + p2_left_len);
    assert_eq!(p1.right().len(), p1_right_len + p2_right_len);

    // Verify p1 contains all expected elements
    let left_contains_all = [1, 2, 4].iter().all(|&x| p1.left().contains(&x));
    let right_contains_all = [3, 5, 6].iter().all(|&x| p1.right().contains(&x));

    assert!(
        left_contains_all,
        "Left partition missing expected elements"
    );
    assert!(
        right_contains_all,
        "Right partition missing expected elements"
    );

    // Check that p2 is now empty
    assert!(p2.is_empty());
    assert_eq!(p2.left().len(), 0);
    assert_eq!(p2.right().len(), 0);
}

#[test]
fn test_append_non_copy_types() {
    // This test verifies that append works for non-Copy types like String
    let mut p1 = Partition::new();
    p1.push_left("hello".to_string());
    p1.push_right("world".to_string());

    let mut p2 = Partition::new();
    p2.push_left("foo".to_string());
    p2.push_right("bar".to_string());

    // Append p2 to p1
    p1.append(&mut p2);

    // Check lengths
    assert_eq!(p1.left().len(), 2);
    assert_eq!(p1.right().len(), 2);

    // Check p2 is empty
    assert!(p2.is_empty());

    // Check p1 contains all expected strings
    let left_contains_all =
        p1.left().iter().any(|s| s == "hello") && p1.left().iter().any(|s| s == "foo");
    let right_contains_all =
        p1.right().iter().any(|s| s == "world") && p1.right().iter().any(|s| s == "bar");

    assert!(
        left_contains_all,
        "Left partition missing expected elements"
    );
    assert!(
        right_contains_all,
        "Right partition missing expected elements"
    );
}

#[test]
fn test_set_partition() {
    let mut p = Partition::new();

    // Add some elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);
    p.push_right(4);

    // Verify initial state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // Move all elements to right partition
    unsafe {
        p.set_partition(0);
    }

    // Verify new state
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 4);
    check_set_equality(p.right().iter().copied(), [1, 2, 3, 4]);

    // Move all elements to left partition
    unsafe {
        p.set_partition(4);
    }

    // Verify new state
    assert_eq!(p.left().len(), 4);
    assert_eq!(p.right().len(), 0);
    check_set_equality(p.left().iter().copied(), [1, 2, 3, 4]);

    // Move half the elements to right partition
    unsafe {
        p.set_partition(2);
    }

    // Verify final state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // Test boundary case with empty vector
    let mut empty_p: Partition<i32> = Partition::new();
    unsafe {
        empty_p.set_partition(0);
    }
    assert_eq!(empty_p.left().len(), 0);
    assert_eq!(empty_p.right().len(), 0);

    // Test that out-of-bounds assertion works
    let mut p2 = Partition::new();
    p2.push_left(1);

    let should_panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            p2.set_partition(2); // len is 1, so this is out of bounds
        }
    }));
    assert!(
        should_panic.is_err(),
        "set_partition should panic with out-of-bounds index"
    );
}

#[test]
fn test_swap_remove_left() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);
    p.push_left(3);
    p.push_right(4);

    // Should have [1, 2, 3] in left partition
    assert_eq!(p.left().len(), 3);

    // Remove the middle element
    let removed = p.swap_remove_left(1);
    assert!(removed == 1 || removed == 2 || removed == 3);

    // Left should now have 2 elements and not contain the removed value
    assert_eq!(p.left().len(), 2);
    let left_elements: Vec<_> = p.left().to_vec();
    assert!(!left_elements.contains(&removed));

    // Right partition should be unchanged
    assert_eq!(p.right().len(), 1);
    assert!(p.right().contains(&4));

    // Try removing from an empty left partition
    let mut empty_p: Partition<i32> = Partition::new();
    let should_panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        empty_p.swap_remove_left(0);
    }));
    assert!(
        should_panic.is_err(),
        "swap_remove_left should panic with out-of-bounds index"
    );
}

#[test]
fn test_swap_remove_right() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_right(2);
    p.push_right(3);
    p.push_right(4);

    // Should have [2, 3, 4] in right partition
    assert_eq!(p.right().len(), 3);

    // Remove the first element in right partition (index 0 in right partition)
    let removed = p.swap_remove_right(0);
    assert!(removed == 2 || removed == 3 || removed == 4);

    // Right should now have 2 elements and not contain the removed value
    assert_eq!(p.right().len(), 2);
    let right_elements: Vec<_> = p.right().to_vec();
    assert!(!right_elements.contains(&removed));

    // Left partition should be unchanged
    assert_eq!(p.left().len(), 1);
    assert!(p.left().contains(&1));

    // Try removing from an empty right partition
    let mut empty_p: Partition<i32> = Partition::new();
    let should_panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        empty_p.swap_remove_right(0);
    }));
    assert!(
        should_panic.is_err(),
        "swap_remove_right should panic with out-of-bounds index"
    );
}

#[test]
fn test_retain_left() {
    let mut p = Partition::new();

    // Add a mix of even and odd numbers to both partitions
    for i in 1..=10 {
        if i <= 6 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // Retain only even numbers in the left partition
    p.retain_left(|&x| x % 2 == 0);

    // Left partition should only have even numbers and be smaller
    assert!(
        p.left().iter().all(|&x| x % 2 == 0),
        "Left partition should only contain even numbers"
    );
    assert_eq!(
        p.left().len(),
        3,
        "Left should have exactly 3 elements (2,4,6)"
    );

    // Check that all expected values are there
    assert!(p.left().contains(&2));
    assert!(p.left().contains(&4));
    assert!(p.left().contains(&6));

    // Right partition should be unchanged
    assert_eq!(p.right().len(), 4);
    for i in 7..=10 {
        assert!(
            p.right().contains(&i),
            "Right partition missing element {}",
            i
        );
    }

    // Test with empty left partition
    let mut empty_left = Partition::new();
    empty_left.push_right(1);
    empty_left.retain_left(|_| false);
    assert_eq!(empty_left.left().len(), 0);
    assert_eq!(empty_left.right().len(), 1);
}

#[test]
fn test_retain_right() {
    let mut p = Partition::new();

    // Add a mix of even and odd numbers to both partitions
    for i in 1..=10 {
        if i <= 5 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // Retain only odd numbers in the right partition
    p.retain_right(|&x| x % 2 == 1);

    // Right partition should only have odd numbers
    assert!(
        p.right().iter().all(|&x| x % 2 == 1),
        "Right partition should only contain odd numbers"
    );

    // Right should have exactly 2 elements (7, 9)
    assert_eq!(p.right().len(), 2);
    assert!(p.right().contains(&7));
    assert!(p.right().contains(&9));

    // Left partition should be unchanged
    assert_eq!(p.left().len(), 5);
    for i in 1..=5 {
        assert!(
            p.left().contains(&i),
            "Left partition missing element {}",
            i
        );
    }

    // Test with empty right partition
    let mut empty_right = Partition::new();
    empty_right.push_left(1);
    empty_right.retain_right(|_| false);
    assert_eq!(empty_right.left().len(), 1);
    assert_eq!(empty_right.right().len(), 0);
}

#[test]
fn test_retain_edge_cases() {
    // Test retaining nothing
    let mut p1 = Partition::new();
    p1.push_left(1);
    p1.push_left(2);
    p1.push_right(3);

    p1.retain_left(|_| false);
    assert_eq!(p1.left().len(), 0);
    assert_eq!(p1.right().len(), 1);

    // Test retaining everything
    let mut p2 = Partition::new();
    p2.push_left(1);
    p2.push_right(2);
    p2.push_right(3);

    p2.retain_right(|_| true);
    assert_eq!(p2.left().len(), 1);
    assert_eq!(p2.right().len(), 2);

    // Test complex retain that requires one of our more complex edge cases
    let mut p3 = Partition::new();
    for i in 1..=10 {
        p3.push_left(i);
    }
    for i in 11..=20 {
        p3.push_right(i);
    }

    // Retain only prime numbers in the left partition
    let is_prime = |&n: &i32| {
        if n <= 1 {
            return false;
        }
        if n <= 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }
        let mut i = 5;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    };

    p3.retain_left(is_prime);

    // Left should have primes: 2, 3, 5, 7
    assert_eq!(p3.left().len(), 4);
    assert!(p3.left().contains(&2));
    assert!(p3.left().contains(&3));
    assert!(p3.left().contains(&5));
    assert!(p3.left().contains(&7));

    // Right should still have all original elements
    assert_eq!(p3.right().len(), 10);
    for i in 11..=20 {
        assert!(p3.right().contains(&i));
    }
}

#[test]
fn test_push_left() {
    // Using our wrapped Partition - it will automatically shuffle on mutation!
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    // No need for explicit shuffling - it happens on mutation via DerefMut

    // Order-agnostic check using our helper
    check_set_equality(p.left().iter().copied(), [1, 2]);
}

#[test]
fn test_push_right() {
    // Using our wrapped Partition - it will automatically shuffle on mutation!
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    // No need for explicit shuffling - it happens on mutation via DerefMut

    // Order-agnostic check using our helper
    check_set_equality(p.right().iter().copied(), [1, 2]);
}

#[test]
fn test_automatic_shuffling() {
    // This test demonstrates that our wrapper automatically shuffles on mutation

    // Use two separate instances of our Partition
    let mut first = Partition::new();
    let mut second = Partition::new();

    // Add the same elements to both
    for i in 0..10 {
        first.push_left(i);
        second.push_left(i);
    }

    // Make a modification to second that should trigger a shuffle
    second.push_right(100);

    // Both should still have the same elements in the left partition
    check_set_equality(first.left().iter().copied(), second.left().iter().copied());

    // But there's a very high probability (1 - 1/10!) they'll be in a different order
    // We won't assert this since it would make the test flaky, but in practice
    // shuffling makes it vanishingly unlikely that the order would be preserved
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

    // For Partition implementations, pop_right can return either value
    // since the order isn't specified
    let first_popped = p.pop_right();
    assert!(first_popped == Some(1) || first_popped == Some(2));

    // We should have one element left
    assert_eq!(p.right().len(), 1);

    // The remaining element should be the other value
    let first_val = first_popped.unwrap();
    let second_val = if first_val == 1 { 2 } else { 1 };
    check_set_equality(p.right(), &[second_val]);

    // Pop the second element
    let second_popped = p.pop_right();
    assert_eq!(second_popped, Some(second_val));

    // Now the right partition should be empty
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

    // First move_to_left will move one of the elements
    let first_moved = p.move_to_left();
    assert!(first_moved == Some(1) || first_moved == Some(2));

    // We now have one element in each partition
    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 1);

    // The left partition should contain the value we just moved
    let left_val = first_moved.unwrap();
    check_set_equality(p.left(), &[left_val]);

    // The right partition should contain the other value
    let right_val = if left_val == 1 { 2 } else { 1 };
    check_set_equality(p.right(), &[right_val]);

    // Move the next element
    let second_moved = p.move_to_left();
    assert_eq!(second_moved, Some(right_val));

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

    let p = Partition::from_raw_parts(vec, partition);
    assert_eq!(p.left(), &[10, 20]);
    assert_eq!(p.right(), &[30, 40]);

    // Edge case: empty vector
    let vec = Vec::<i32>::new();
    let partition = 0;

    let p = Partition::from_raw_parts(vec, partition);
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());

    // Edge case: all elements in left partition
    let vec = vec![1, 2, 3];
    let partition = 3;

    let p = Partition::from_raw_parts(vec, partition);
    assert_eq!(p.left(), &[1, 2, 3]);
    assert!(p.right().is_empty());

    // Edge case: all elements in right partition
    let vec = vec![1, 2, 3];
    let partition = 0;

    let p = Partition::from_raw_parts(vec, partition);
    assert!(p.left().is_empty());
    assert_eq!(p.right(), &[1, 2, 3]);
}

#[test]
#[should_panic(expected = "partition index 4 is out of bounds")]
fn test_from_raw_parts_panic() {
    // Invalid case - should panic
    let vec = vec![1, 2, 3];
    let partition = 4; // Beyond the vector length

    // This should panic
    let _p = Partition::from_raw_parts(vec, partition);
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
    let reconstructed = Partition::from_raw_parts(vec, partition);

    // Verify left and right partitions match
    // Use set equality since order within partitions might vary
    check_set_equality(reconstructed.left().iter().copied(), [1, 2]);
    check_set_equality(reconstructed.right().iter().copied(), [3, 4]);
}

//
// Additional tests to strengthen test coverage
//

#[test]
fn test_with_non_copy_types() {
    // Test Partition with String (non-Copy type)
    let mut p = Partition::new();
    p.push_left("hello".to_string());
    p.push_left("world".to_string());
    p.push_right("foo".to_string());
    p.push_right("bar".to_string());

    // Test basic operations
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // Check accessing without cloning works
    let left_has_hello = p.left().iter().any(|s| s == "hello");
    let left_has_world = p.left().iter().any(|s| s == "world");
    assert!(left_has_hello);
    assert!(left_has_world);

    // Test pops (which move values)
    let popped = p.pop_left();
    assert!(popped.is_some());

    // Check the popped value is what we expect
    let popped_val = popped.unwrap();
    assert!(popped_val == "hello" || popped_val == "world");

    // Test cloning the partition
    let p_clone = p.clone();
    assert_eq!(p.left().len(), p_clone.left().len());
    assert_eq!(p.right().len(), p_clone.right().len());

    // Verify modifying one doesn't affect the other
    p.push_left("another".to_string());
    assert_ne!(p.left().len(), p_clone.left().len());
}

#[test]
fn test_clone_behavior() {
    let mut original = Partition::new();
    original.push_left(1);
    original.push_left(2);
    original.push_right(3);

    // Clone the partition
    let cloned = original.clone();

    // Verify cloned partition has same content
    check_set_equality(
        original.left().iter().copied(),
        cloned.left().iter().copied(),
    );
    check_set_equality(
        original.right().iter().copied(),
        cloned.right().iter().copied(),
    );

    // Modify original
    original.push_left(4);
    original.push_right(5);

    // Verify cloned partition remains unchanged
    assert_eq!(cloned.left().len(), 2);
    assert_eq!(cloned.right().len(), 1);

    check_set_equality(cloned.left().iter().copied(), [1, 2]);
    check_set_equality(cloned.right().iter().copied(), [3]);
}

#[test]
fn test_default_behavior() {
    // Test the Default implementation
    let p: Partition<i32> = Default::default();

    assert!(p.is_empty());
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_partial_iterator_consumption() {
    // Test drain_to_right with partial consumption
    let mut p = Partition::new();
    for i in 0..10 {
        p.push_left(i);
    }

    let mut iter = p.drain_to_right();

    // Only consume half the elements
    for _ in 0..5 {
        let _ = iter.next();
    }

    // Drop the iterator - remaining elements should still move to right
    drop(iter);

    // Check all elements moved to the right
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 10);

    // Test drain_to_left with partial consumption
    let mut p = Partition::new();
    for i in 0..10 {
        p.push_right(i);
    }

    let mut iter = p.drain_to_left();

    // Only consume some elements
    assert!(iter.next().is_some());
    assert!(iter.next().is_some());

    // Drop iterator
    drop(iter);

    // Check all elements moved
    assert_eq!(p.left().len(), 10);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_iterator_size_hint_accuracy() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);
    p.push_left(3);

    // Test size_hint for drain_to_right
    let iter = p.drain_to_right();
    assert_eq!(iter.size_hint(), (3, Some(3)));

    // Test size_hint for drain_left
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    let iter = p.drain_left();
    assert_eq!(iter.size_hint(), (2, Some(2)));

    // Test size_hint for drain_right
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    let iter = p.drain_right();
    assert_eq!(iter.size_hint(), (2, Some(2)));
}

#[test]
fn test_partition_invariants() {
    // Test that partition index always stays <= inner.len()
    let mut p = Partition::new();

    for i in 0..100 {
        if i % 2 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }

        // Check the invariant after each operation
        assert!(p.partition <= p.len());
    }

    // Remove elements and check invariant
    for _ in 0..50 {
        if p.left().len() > 0 {
            p.pop_left();
        }
        if p.right().len() > 0 {
            p.pop_right();
        }

        // Check the invariant after each operation
        assert!(p.partition <= p.len());
    }
}

#[test]
fn test_boundary_conditions() {
    // Create a large partition
    let mut p = Partition::new();

    // Add many elements
    for i in 0..1000 {
        if i % 2 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // Check counts
    assert_eq!(p.left().len(), 500);
    assert_eq!(p.right().len(), 500);

    // Move all from left to right
    while p.left().len() > 0 {
        p.move_to_right();
    }

    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 1000);

    // Move all from right to left
    while p.right().len() > 0 {
        p.move_to_left();
    }

    assert_eq!(p.left().len(), 1000);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_debug_contains_expected_elements() {
    // Test that the Debug implementation contains all elements
    let mut p = Partition::new();
    p.push_left(42);
    p.push_left(24);
    p.push_right(99);

    // Convert debug output to string
    let debug_string = format!("{:?}", p);

    // Debug should mention it's a Partition
    assert!(debug_string.contains("Partition"));

    // Debug should contain left and right sections
    assert!(debug_string.contains("left"));
    assert!(debug_string.contains("right"));

    // Debug should contain all the values (order agnostic)
    assert!(debug_string.contains("42"));
    assert!(debug_string.contains("24"));
    assert!(debug_string.contains("99"));
}

//
// Additional tests for uncovered paths
//

#[test]
fn test_with_capacity_reservations() {
    // Test that with_capacity actually reserves the capacity
    let capacity = 100;
    let p: Partition<i32> = Partition::with_capacity(capacity);

    // Capacity should be at least what we asked for
    // Access internal Partition fields through Deref
    assert!(p.0.inner.capacity() >= capacity);
}

#[test]
fn test_to_raw_parts_empty() {
    // Test to_raw_parts on an empty partition
    let p = Partition::<i32>::new();

    let (vec, partition) = p.to_raw_parts();

    assert!(vec.is_empty());
    assert_eq!(partition, 0);

    // Recreate partition from parts
    let p2 = Partition::from_raw_parts(vec, partition);
    assert!(p2.is_empty());
    assert_eq!(p2.left().len(), 0);
    assert_eq!(p2.right().len(), 0);
}

#[test]
fn test_left_mut_right_mut_empty() {
    // Test left_mut and right_mut with empty partitions
    let mut p = Partition::<i32>::new();

    // Empty left partition
    assert!(p.left_mut().is_empty());

    // Empty right partition
    assert!(p.right_mut().is_empty());
}

#[test]
fn test_partitions_mut_empty() {
    // Test partitions_mut with various empty configurations

    // Both partitions empty
    let mut p = Partition::<i32>::new();
    let (left, right) = p.partitions_mut();
    assert!(left.is_empty());
    assert!(right.is_empty());

    // Left empty, right has elements
    let mut p = Partition::new();
    p.push_right(1);
    let (left, right) = p.partitions_mut();
    assert!(left.is_empty());
    assert!(!right.is_empty());

    // Right empty, left has elements
    let mut p = Partition::new();
    p.push_left(1);
    let (left, right) = p.partitions_mut();
    assert!(!left.is_empty());
    assert!(right.is_empty());
}

#[test]
fn test_iterator_behavior_after_emptying() {
    // Test calling next() on iterators after they're empty

    // Test drain_to_right
    let mut p = Partition::new();
    p.push_left(1);

    let mut iter = p.drain_to_right();
    assert_eq!(iter.next(), Some(1)); // Consume the only element
    assert_eq!(iter.next(), None); // Should return None
    assert_eq!(iter.next(), None); // Should still return None

    // Test drain_left
    let mut p = Partition::new();
    p.push_left(1);

    let mut iter = p.drain_left();
    assert_eq!(iter.next(), Some(1)); // Consume the only element
    assert_eq!(iter.next(), None); // Should return None
    assert_eq!(iter.next(), None); // Should still return None

    // Test drain_right
    let mut p = Partition::new();
    p.push_right(1);

    let mut iter = p.drain_right();
    assert_eq!(iter.next(), Some(1)); // Consume the only element
    assert_eq!(iter.next(), None); // Should return None
    assert_eq!(iter.next(), None); // Should still return None
}

#[test]
fn test_zero_sized_types() {
    // Test with zero-sized type ()
    let mut p = Partition::<()>::new();

    // Push some values
    p.push_left(());
    p.push_left(());
    p.push_right(());

    // Check counts
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 1);

    // Basic operations
    let moved = p.move_to_right();
    assert_eq!(moved, Some(()));

    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 2);

    // Pop values
    assert_eq!(p.pop_left(), Some(()));
    assert_eq!(p.pop_right(), Some(()));

    // Should still have one element in right
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 1);
}

#[test]
fn test_complex_method_interactions() {
    // Test complex interactions between different methods
    let mut p = Partition::new();

    // Mix of operations in specific sequence
    p.push_left(1);
    p.push_right(2);
    p.push_left(3);
    p.push_right(4);

    // Move an element right, then pop it
    let moved = p.move_to_right();
    assert!(moved.is_some());
    let popped = p.pop_right();
    assert!(popped.is_some());

    // Push to left, then drain right to left
    p.push_left(5);
    let drained: Vec<_> = p.drain_to_left().collect();
    assert!(!drained.is_empty());

    // Now everything should be in left
    assert!(p.right().is_empty());
    assert!(!p.left().is_empty());

    // Clear and verify empty
    p.clear();
    assert!(p.is_empty());

    // Specific sequence that might trigger edge cases
    p.push_left(1);
    p.push_right(2);
    p.pop_left(); // Left now empty
    p.push_left(3);
    p.move_to_right(); // Left empty again
    p.move_to_left(); // Move from right to left

    // Verify state
    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 1);
}

#[test]
fn test_unchecked_from_raw_parts_preserves_invariants() {
    // Test that from_raw_parts_unchecked preserves invariants
    // SAFETY: We're only calling this with valid data

    let vec = vec![1, 2, 3, 4];
    let partition = 2;

    // Valid use of from_raw_parts_unchecked
    let p = unsafe { Partition::from_raw_parts_unchecked(vec.clone(), partition) };

    // Verify partition index is preserved
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // Verify we can perform operations as expected
    let left_items: Vec<_> = p.left().to_vec();
    let right_items: Vec<_> = p.right().to_vec();

    // Check that vectors have the right elements
    check_set_equality(left_items, [1, 2]);
    check_set_equality(right_items, [3, 4]);

    // Test with edge cases (empty vector, partition=0)
    let empty_vec = Vec::<i32>::new();
    let p = unsafe { Partition::from_raw_parts_unchecked(empty_vec, 0) };
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());
}
