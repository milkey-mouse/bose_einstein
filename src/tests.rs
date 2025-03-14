use super::*; // import everything from the parent module
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

    // direct tests against the parent Partition implementation to test allocator-specific features
    // TODO: fix Partition to impl everything for Partition<T, A> if feature(allocator_api) is enabled
    // so we can wrap it as struct Partition<T, A: Allocator = Global>(super::Partition<T, A>) and use
    // the shuffling wrapped partition for all tests including these

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
        // create a vector with the global allocator
        let vec: Vec<i32, Global> = Vec::new_in(Global);

        // create a vector with items using the global allocator
        let mut vec_with_items: Vec<i32, Global> = Vec::with_capacity_in(4, Global);
        vec_with_items.push(1);
        vec_with_items.push(2);
        vec_with_items.push(3);
        vec_with_items.push(4);

        // empty vector
        let p = super::super::Partition::from_raw_parts(vec, 0);
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right(), &[]);

        // vector with items
        let p = super::super::Partition::from_raw_parts(vec_with_items, 2);
        assert_eq!(p.left(), &[1, 2]);
        assert_eq!(p.right(), &[3, 4]);
    }

    #[test]
    fn test_from_raw_parts_unchecked_with_allocator() {
        // create a vector with items using the global allocator
        let mut vec_with_items: Vec<i32, Global> = Vec::with_capacity_in(4, Global);
        vec_with_items.push(1);
        vec_with_items.push(2);
        vec_with_items.push(3);
        vec_with_items.push(4);

        // safe usage
        let p = unsafe { super::super::Partition::from_raw_parts_unchecked(vec_with_items, 2) };
        assert_eq!(p.left(), &[1, 2]);
        assert_eq!(p.right(), &[3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_from_raw_parts_panic_with_allocator() {
        // create a vector with items using the global allocator
        let mut vec: Vec<i32, Global> = Vec::with_capacity_in(3, Global);
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // this should panic - partition index is out of bounds
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

    // static methods need to be explicitly forwarded
    pub fn from_raw_parts(inner: Vec<T>, partition: usize) -> Self {
        Self(super::Partition::from_raw_parts(inner, partition))
    }

    pub unsafe fn from_raw_parts_unchecked(inner: Vec<T>, partition: usize) -> Self {
        Self(unsafe { super::Partition::from_raw_parts_unchecked(inner, partition) })
    }

    // forward to_raw_parts to avoid move issues
    pub fn to_raw_parts(self) -> (Vec<T>, usize) {
        let (vec, partition) = self.0.to_raw_parts();
        (vec, partition)
    }
}

// Implement DerefMut to shuffle on mutation
impl<T> DerefMut for Partition<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // shuffle both partitions unconditionally
        let mut rng = rand::rng();
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
    // create several partitions and add the same elements in different orders
    // these will automatically shuffle after every mutation thanks to DerefMut!
    let mut p1 = Partition::new();
    let mut p2 = Partition::new();
    let mut p3 = Partition::new();

    // different insertion orders
    p1.push_left(1);
    p1.push_left(2);
    p1.push_left(3);

    p2.push_left(3);
    p2.push_left(1);
    p2.push_left(2);

    p3.push_left(2);
    p3.push_left(3);
    p3.push_left(1);

    // all should contain the same elements, regardless of insertion order
    check_set_equality(p1.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p2.left().iter().copied(), [1, 2, 3]);
    check_set_equality(p3.left().iter().copied(), [1, 2, 3]);

    // no need to explicitly shuffle - it happens automatically on mutation!

    // still contains the same elements
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

    // add elements until we exceed the initial capacity
    for i in 0..initial_capacity + 10 {
        p.push_left(i as i32);
    }

    // capacity should have increased
    assert!(p.capacity() > initial_capacity);
}

#[test]
fn test_reserve() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    let current_len = p.len();
    p.reserve(20);

    // capacity should be at least current length + 20
    assert!(p.capacity() >= current_len + 20);

    // should still have the same elements
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

    // capacity should be at least current length + additional
    assert!(p.capacity() >= current_len + additional);

    // should still have the same elements
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);
}

#[test]
fn test_shrink_to_fit() {
    // create a partition with larger capacity
    let mut p: Partition<i32> = Partition::with_capacity(100);

    // add a few elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // verify we have excess capacity
    assert!(p.capacity() > 3);

    // shrink the capacity
    p.shrink_to_fit();

    // should still have at least enough for our elements
    assert!(p.capacity() >= 3);

    // original elements should be preserved
    assert_eq!(p.len(), 3);
    check_set_equality(p.left().iter().copied(), [1, 2]);
    check_set_equality(p.right().iter().copied(), [3]);
}

#[test]
fn test_shrink_to() {
    // create a partition with larger capacity
    let mut p: Partition<i32> = Partition::with_capacity(100);

    // add a few elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // verify we have excess capacity
    assert!(p.capacity() > 3);

    // shrink to a capacity of at most 10
    p.shrink_to(10);

    // should be between our length and the limit
    assert!(p.capacity() >= 3);
    assert!(p.capacity() <= 10);

    // original elements should be preserved
    assert_eq!(p.len(), 3);
    check_set_equality(p.left().iter().copied(), [1, 2]);
    check_set_equality(p.right().iter().copied(), [3]);

    // shrinking to less than length should still keep enough capacity for all elements
    p.shrink_to(2); // our length is 3, so it should maintain capacity >= 3
    assert!(p.capacity() >= 3);
}

#[test]
fn test_try_reserve() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    // this should succeed for reasonable sizes
    let result = p.try_reserve(10);
    assert!(result.is_ok());

    // after successful reservation, capacity should be increased
    assert!(p.capacity() >= p.len() + 10);

    // original elements should be preserved
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);

    // we don't test allocation failure cases since they're hard to reliably
    // trigger in a test environment
}

#[test]
fn test_try_reserve_exact() {
    let mut p: Partition<i32> = Partition::new();
    p.push_left(1);
    p.push_right(2);

    // this should succeed for reasonable sizes
    let result = p.try_reserve_exact(10);
    assert!(result.is_ok());

    // after successful reservation, capacity should be increased
    assert!(p.capacity() >= p.len() + 10);

    // original elements should be preserved
    check_set_equality(p.left().iter().copied(), [1]);
    check_set_equality(p.right().iter().copied(), [2]);
}

#[test]
fn test_append() {
    // create two partitions
    let mut p1 = Partition::new();
    p1.push_left(1);
    p1.push_left(2);
    p1.push_right(3);

    let mut p2 = Partition::new();
    p2.push_left(4);
    p2.push_right(5);
    p2.push_right(6);

    // save the original lengths
    let p1_left_len = p1.left().len();
    let p1_right_len = p1.right().len();
    let p2_left_len = p2.left().len();
    let p2_right_len = p2.right().len();

    // append p2 to p1
    p1.append(&mut p2);

    // check that p1 has all elements from both
    assert_eq!(p1.left().len(), p1_left_len + p2_left_len);
    assert_eq!(p1.right().len(), p1_right_len + p2_right_len);

    // verify p1 contains all expected elements
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

    // check that p2 is now empty
    assert!(p2.is_empty());
    assert_eq!(p2.left().len(), 0);
    assert_eq!(p2.right().len(), 0);
}

#[test]
fn test_append_non_copy_types() {
    // this test verifies that append works for non-Copy types like String
    let mut p1 = Partition::new();
    p1.push_left("hello".to_string());
    p1.push_right("world".to_string());

    let mut p2 = Partition::new();
    p2.push_left("foo".to_string());
    p2.push_right("bar".to_string());

    // append p2 to p1
    p1.append(&mut p2);

    // check lengths
    assert_eq!(p1.left().len(), 2);
    assert_eq!(p1.right().len(), 2);

    // check p2 is empty
    assert!(p2.is_empty());

    // check p1 contains all expected strings
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

    // add some elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);
    p.push_right(4);

    // verify initial state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // move all elements to right partition
    unsafe {
        p.set_partition(0);
    }

    // verify new state
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 4);
    check_set_equality(p.right().iter().copied(), [1, 2, 3, 4]);

    // move all elements to left partition
    unsafe {
        p.set_partition(4);
    }

    // verify new state
    assert_eq!(p.left().len(), 4);
    assert_eq!(p.right().len(), 0);
    check_set_equality(p.left().iter().copied(), [1, 2, 3, 4]);

    // move half the elements to right partition
    unsafe {
        p.set_partition(2);
    }

    // verify final state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // test boundary case with empty vector
    let mut empty_p: Partition<i32> = Partition::new();
    unsafe {
        empty_p.set_partition(0);
    }
    assert_eq!(empty_p.left().len(), 0);
    assert_eq!(empty_p.right().len(), 0);

    // test that out-of-bounds assertion works
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

    // should have [1, 2, 3] in left partition
    assert_eq!(p.left().len(), 3);

    // remove the middle element
    let removed = p.swap_remove_left(1);
    assert!(removed == 1 || removed == 2 || removed == 3);

    // left should now have 2 elements and not contain the removed value
    assert_eq!(p.left().len(), 2);
    let left_elements: Vec<_> = p.left().to_vec();
    assert!(!left_elements.contains(&removed));

    // right partition should be unchanged
    assert_eq!(p.right().len(), 1);
    assert!(p.right().contains(&4));

    // try removing from an empty left partition
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

    // should have [2, 3, 4] in right partition
    assert_eq!(p.right().len(), 3);

    // remove the first element in right partition (index 0 in right partition)
    let removed = p.swap_remove_right(0);
    assert!(removed == 2 || removed == 3 || removed == 4);

    // right should now have 2 elements and not contain the removed value
    assert_eq!(p.right().len(), 2);
    let right_elements: Vec<_> = p.right().to_vec();
    assert!(!right_elements.contains(&removed));

    // left partition should be unchanged
    assert_eq!(p.left().len(), 1);
    assert!(p.left().contains(&1));

    // try removing from an empty right partition
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

    // add a mix of even and odd numbers to both partitions
    for i in 1..=10 {
        if i <= 6 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // retain only even numbers in the left partition
    p.retain_left(|&x| x % 2 == 0);

    // left partition should only have even numbers and be smaller
    assert!(
        p.left().iter().all(|&x| x % 2 == 0),
        "Left partition should only contain even numbers"
    );
    assert_eq!(
        p.left().len(),
        3,
        "Left should have exactly 3 elements (2,4,6)"
    );

    // check that all expected values are there
    assert!(p.left().contains(&2));
    assert!(p.left().contains(&4));
    assert!(p.left().contains(&6));

    // right partition should be unchanged
    assert_eq!(p.right().len(), 4);
    for i in 7..=10 {
        assert!(
            p.right().contains(&i),
            "Right partition missing element {}",
            i
        );
    }

    // test with empty left partition
    let mut empty_left = Partition::new();
    empty_left.push_right(1);
    empty_left.retain_left(|_| false);
    assert_eq!(empty_left.left().len(), 0);
    assert_eq!(empty_left.right().len(), 1);
}

#[test]
fn test_retain_right() {
    let mut p = Partition::new();

    // add a mix of even and odd numbers to both partitions
    for i in 1..=10 {
        if i <= 5 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // retain only odd numbers in the right partition
    p.retain_right(|&x| x % 2 == 1);

    // right partition should only have odd numbers
    assert!(
        p.right().iter().all(|&x| x % 2 == 1),
        "Right partition should only contain odd numbers"
    );

    // right should have exactly 2 elements (7, 9)
    assert_eq!(p.right().len(), 2);
    assert!(p.right().contains(&7));
    assert!(p.right().contains(&9));

    // left partition should be unchanged
    assert_eq!(p.left().len(), 5);
    for i in 1..=5 {
        assert!(
            p.left().contains(&i),
            "Left partition missing element {}",
            i
        );
    }

    // test with empty right partition
    let mut empty_right = Partition::new();
    empty_right.push_left(1);
    empty_right.retain_right(|_| false);
    assert_eq!(empty_right.left().len(), 1);
    assert_eq!(empty_right.right().len(), 0);
}

#[test]
fn test_retain_edge_cases() {
    // test retaining nothing
    let mut p1 = Partition::new();
    p1.push_left(1);
    p1.push_left(2);
    p1.push_right(3);

    p1.retain_left(|_| false);
    assert_eq!(p1.left().len(), 0);
    assert_eq!(p1.right().len(), 1);

    // test retaining everything
    let mut p2 = Partition::new();
    p2.push_left(1);
    p2.push_right(2);
    p2.push_right(3);

    p2.retain_right(|_| true);
    assert_eq!(p2.left().len(), 1);
    assert_eq!(p2.right().len(), 2);

    // test complex retain that requires one of our more complex edge cases
    let mut p3 = Partition::new();
    for i in 1..=10 {
        p3.push_left(i);
    }
    for i in 11..=20 {
        p3.push_right(i);
    }

    // retain only prime numbers in the left partition
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

    // left should have primes: 2, 3, 5, 7
    assert_eq!(p3.left().len(), 4);
    assert!(p3.left().contains(&2));
    assert!(p3.left().contains(&3));
    assert!(p3.left().contains(&5));
    assert!(p3.left().contains(&7));

    // right should still have all original elements
    assert_eq!(p3.right().len(), 10);
    for i in 11..=20 {
        assert!(p3.right().contains(&i));
    }
}

#[test]
fn test_push_left() {
    // using our wrapped Partition - it will automatically shuffle on mutation!
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    // no need for explicit shuffling - it happens on mutation via DerefMut

    // order-agnostic check using our helper
    check_set_equality(p.left().iter().copied(), [1, 2]);
}

#[test]
fn test_push_right() {
    // using our wrapped Partition - it will automatically shuffle on mutation!
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    // no need for explicit shuffling - it happens on mutation via DerefMut

    // order-agnostic check using our helper
    check_set_equality(p.right().iter().copied(), [1, 2]);
}

#[test]
fn test_automatic_shuffling() {
    // this test demonstrates that our wrapper automatically shuffles on mutation

    // use two separate instances of our Partition
    let mut first = Partition::new();
    let mut second = Partition::new();

    // add the same elements to both
    for i in 0..10 {
        first.push_left(i);
        second.push_left(i);
    }

    // make a modification to second that should trigger a shuffle
    second.push_right(100);

    // both should still have the same elements in the left partition
    check_set_equality(first.left().iter().copied(), second.left().iter().copied());

    // but there's a very high probability (1 - 1/10!) they'll be in a different order
    // we won't assert this since it would make the test flaky, but in practice
    // shuffling makes it vanishingly unlikely that the order would be preserved
}

#[test]
fn test_mixed_push() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_right(2);
    p.push_left(3);
    p.push_right(4);

    // order-agnostic checks using our helpers
    check_set_equality(p.left().iter().copied(), [1, 3]);
    check_set_equality(p.right().iter().copied(), [2, 4]);
}

#[test]
fn test_pop_left() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    // we can't guarantee which element will be popped first
    let val = p.pop_left();
    assert!(val == Some(1) || val == Some(2));

    // we know which element should remain
    let remaining = if val == Some(1) { 2 } else { 1 };

    // order-agnostic check using our helper
    check_set_equality(p.left(), &[remaining]);

    // pop the remaining element
    assert_eq!(p.pop_left(), Some(remaining));
    assert_eq!(p.left(), &[]);
    assert_eq!(p.pop_left(), None);
}

#[test]
fn test_pop_right() {
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    // for Partition implementations, pop_right can return either value
    // since the order isn't specified
    let first_popped = p.pop_right();
    assert!(first_popped == Some(1) || first_popped == Some(2));

    // we should have one element left
    assert_eq!(p.right().len(), 1);

    // the remaining element should be the other value
    let first_val = first_popped.unwrap();
    let second_val = if first_val == 1 { 2 } else { 1 };
    check_set_equality(p.right(), &[second_val]);

    // pop the second element
    let second_popped = p.pop_right();
    assert_eq!(second_popped, Some(second_val));

    // now the right partition should be empty
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

    // order-agnostic checks using our helpers
    check_set_equality(p.left(), &[1]);
    check_set_equality(p.right(), &[2]);
}

#[test]
fn test_left_mut_and_right_mut() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_right(2);

    // we need to modify all elements since we don't know the order
    for item in p.left_mut() {
        *item = 3;
    }
    for item in p.right_mut() {
        *item = 4;
    }

    // order-agnostic checks using our helpers
    check_set_equality(p.left(), &[3]);
    check_set_equality(p.right(), &[4]);
}

#[test]
fn test_move_to_left() {
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    // first move_to_left will move one of the elements
    let first_moved = p.move_to_left();
    assert!(first_moved == Some(1) || first_moved == Some(2));

    // we now have one element in each partition
    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 1);

    // the left partition should contain the value we just moved
    let left_val = first_moved.unwrap();
    check_set_equality(p.left(), &[left_val]);

    // the right partition should contain the other value
    let right_val = if left_val == 1 { 2 } else { 1 };
    check_set_equality(p.right(), &[right_val]);

    // move the next element
    let second_moved = p.move_to_left();
    assert_eq!(second_moved, Some(right_val));

    // verify final state
    check_set_equality(p.left(), &[1, 2]);
    assert_eq!(p.right(), &[]);

    // no more elements to move
    assert_eq!(p.move_to_left(), None);
}

#[test]
fn test_move_to_right() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    // we can't guarantee which element will be moved first
    let val = p.move_to_right();
    assert!(val == Some(1) || val == Some(2));

    // we can compute which element should remain in left and which moved to right
    let remaining = if val == Some(1) { 2 } else { 1 };
    let moved = if val == Some(1) { 1 } else { 2 };

    // check both partitions using our helpers
    check_set_equality(p.left(), &[remaining]);
    check_set_equality(p.right(), &[moved]);

    // move the remaining element
    assert_eq!(p.move_to_right(), Some(remaining));
    assert_eq!(p.left(), &[]);

    // check right partition has both elements
    check_set_equality(p.right(), &[1, 2]);

    // no more elements to move
    assert_eq!(p.move_to_right(), None);
}

#[test]
fn test_drain_left() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // collect drained elements
    let drained: Vec<_> = p.drain_left().collect();
    assert_eq!(drained.len(), 2);

    assert_eq!(p.left(), &[]);

    // check right partition
    assert_eq!(p.right(), &[3]);
}

#[test]
fn test_drain_right() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_right(2);
    p.push_right(3);

    // collect drained elements
    let drained: Vec<_> = p.drain_right().collect();
    assert_eq!(drained.len(), 2);

    assert_eq!(p.right(), &[]);

    // check left partition
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

    // drain left elements to right - store elements to check them
    let moved: Vec<_> = p.drain_to_right().collect();

    // verify we got the expected number of elements
    assert_eq!(moved.len(), 2);

    // verify partitions
    assert_eq!(p.left(), &[]);
    assert_eq!(p.right().len(), 3);

    // all elements should now be in the right partition
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

    // drain right elements to left - store elements to check them
    let moved: Vec<_> = p.drain_to_left().collect();

    // verify we got the expected number of elements
    assert_eq!(moved.len(), 2);

    // verify partitions
    assert_eq!(p.right(), &[]);
    assert_eq!(p.left().len(), 3);

    // all elements should now be in the left partition
    let left_vals: BTreeSet<_> = p.left().iter().copied().collect();
    assert!(left_vals.contains(&1));
    assert!(left_vals.contains(&2));
    assert!(left_vals.contains(&3));
}

#[test]
fn test_drain_iterator_size_hint() {
    // test drain_to_right size_hint
    {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);

        let iter = p.drain_to_right();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    // test drain_to_left size_hint
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
    // test partially consumed drain_to_right
    {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);
        p.push_left(3);

        // get a count of elements before we drain
        let left_count = p.left().len();

        // only take one element from the iterator
        {
            let mut iter = p.drain_to_right();
            let first = iter.next();
            assert!(first.is_some());
            // let the iterator drop here - should move remaining elements
        }

        // all elements should now be in the right partition
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right().len(), left_count);
    }

    // test partially consumed drain_to_left
    {
        let mut p = Partition::new();
        p.push_right(1);
        p.push_right(2);
        p.push_right(3);

        // get a count of elements before we drain
        let right_count = p.right().len();

        // only take one element from the iterator
        {
            let mut iter = p.drain_to_left();
            let first = iter.next();
            assert!(first.is_some());
            // let the iterator drop here - should move remaining elements
        }

        // all elements should now be in the left partition
        assert_eq!(p.right(), &[]);
        assert_eq!(p.left().len(), right_count);
    }
}

#[test]
fn test_drain_immediate_drop() {
    // test immediate drop of drain_to_right (same as calling drain_to_right())
    {
        let mut p = Partition::new();
        p.push_left(1);
        p.push_left(2);

        // create and immediately drop the iterator
        drop(p.drain_to_right());

        // all elements should now be in the right partition
        assert_eq!(p.left(), &[]);
        assert_eq!(p.right().len(), 2);
    }

    // test immediate drop of drain_to_left (same as calling drain_to_left())
    {
        let mut p = Partition::new();
        p.push_right(1);
        p.push_right(2);

        // create and immediately drop the iterator
        drop(p.drain_to_left());

        // all elements should now be in the left partition
        assert_eq!(p.right(), &[]);
        assert_eq!(p.left().len(), 2);
    }
}

#[test]
fn test_complex_operations() {
    let mut p = Partition::new();

    // push items
    p.push_left(1);
    p.push_right(2);
    p.push_left(3);

    // verify using our helper
    check_set_equality(p.left(), &[1, 3]);
    check_set_equality(p.right(), &[2]);

    // move right
    let val = p.move_to_right();
    assert!(val == Some(1) || val == Some(3));

    // move left
    let moved_left = p.move_to_left();
    assert!(moved_left.is_some());

    // count elements in each partition
    let left_count = p.left().len();
    let right_count = p.right().len();
    assert_eq!(left_count + right_count, 3);

    // drain left to right
    let drained_left: Vec<_> = p.drain_to_right().collect();
    assert_eq!(drained_left.len(), left_count);
    assert_eq!(p.left(), &[]);
    assert_eq!(p.right().len(), 3);

    // drain right to left
    let drained_right: Vec<_> = p.drain_to_left().collect();
    assert_eq!(drained_right.len(), 3);
    assert_eq!(p.right(), &[]);
    assert_eq!(p.left().len(), 3);
}

#[test]
fn test_large_partition_with_interleaved_operations() {
    // create a larger partition with more complex operations
    let mut p = Partition::new();

    // push lots of elements in different patterns
    for i in 0..50 {
        if i % 3 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // expected values in each partition based on our insertion pattern
    let expected_left: Vec<_> = (0..50).filter(|&i| i % 3 == 0).collect();
    let expected_right: Vec<_> = (0..50).filter(|&i| i % 3 != 0).collect();

    // verify the partitions have correct elements
    let left_vec: Vec<_> = p.left().to_vec();
    let right_vec: Vec<_> = p.right().to_vec();

    // check that left contains all expected values
    for &val in &expected_left {
        assert!(left_vec.contains(&val));
    }
    assert_eq!(left_vec.len(), expected_left.len());

    // check that right contains all expected values
    for &val in &expected_right {
        assert!(right_vec.contains(&val));
    }
    assert_eq!(right_vec.len(), expected_right.len());

    // move half of the left elements to the right
    let left_count = p.left().len();
    let mut moved_to_right = Vec::new();

    for _ in 0..(left_count / 2) {
        if let Some(val) = p.move_to_right() {
            moved_to_right.push(val);
        }
    }

    // verify expected counts
    assert_eq!(p.left().len(), left_count - moved_to_right.len());

    // move items from right back to left
    for _ in 0..5 {
        p.move_to_left();
    }

    // add more elements
    for i in 50..60 {
        if i % 2 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // drain all elements from partitions
    let left_elements: Vec<_> = p.drain_left().collect();
    let right_elements: Vec<_> = p.drain_right().collect();

    // verify drained elements total to expected count
    assert_eq!(left_elements.len() + right_elements.len(), 60);

    // verify partitions are empty
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());
}

#[test]
fn test_random_operation_sequence() {
    let mut p = Partition::new();

    // maintain shadow collections to verify behavior
    let mut expected_left = Vec::new();
    let mut expected_right = Vec::new();

    // mixed Sequence 1: Add elements to both sides
    for i in 0..10 {
        if i % 2 == 0 {
            p.push_left(i);
            expected_left.push(i);
        } else {
            p.push_right(i);
            expected_right.push(i);
        }
    }

    // verify state - direct element-by-element comparison
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

    // mixed Sequence 2: Move elements between partitions
    // move 2 elements from left to right
    for _ in 0..2 {
        if let Some(val) = p.move_to_right() {
            expected_left.retain(|&x| x != val);
            expected_right.push(val);
        }
    }

    // move 1 element from right to left
    if let Some(val) = p.move_to_left() {
        expected_right.retain(|&x| x != val);
        expected_left.push(val);
    }

    // verify state - direct comparison
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

    // mixed Sequence 3: Pop elements
    if let Some(val) = p.pop_left() {
        expected_left.retain(|&x| x != val);
    }

    if let Some(val) = p.pop_right() {
        expected_right.retain(|&x| x != val);
    }

    // verify state - direct comparison
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

    // mixed Sequence 4: Add more elements and drain
    for i in 10..15 {
        if i % 2 == 0 {
            p.push_left(i);
            expected_left.push(i);
        } else {
            p.push_right(i);
            expected_right.push(i);
        }
    }

    // get counts before draining
    let left_count = expected_left.len();
    let right_count = expected_right.len();

    // drain to right and verify
    let drained_left: Vec<_> = p.drain_to_right().collect();
    assert_eq!(drained_left.len(), left_count);

    // all elements should now be in the right partition
    assert!(p.left().is_empty());
    assert_eq!(p.right().len(), left_count + right_count);
}

#[test]
fn test_push_left_swap_behavior() {
    // this test verifies the internal behavior of push_left,
    // which swaps elements with the partition point
    let mut p = Partition::new();

    // first, set up a known state
    p.push_right(1);
    p.push_right(2);
    p.push_right(3);

    // the internal array should now be [1, 2, 3] with partition=0
    assert_eq!(p.left(), &[]);
    check_set_equality(p.right().iter().copied(), [1, 2, 3]);

    // now when we push_left(4), it should:
    // 1. Swap 4 with element at partition point (which is 1)
    // 2. Push the swapped element (1) to the end
    // 3. Increment partition point
    p.push_left(4);

    // expected state:
    // - Left contains [4]
    // - Right contains [2, 3, 1]
    check_set_equality(p.left().iter().copied(), [4]);

    // check that we have exactly the 3 expected elements in right
    let right_set: BTreeSet<_> = p.right().iter().copied().collect();
    assert_eq!(right_set.len(), 3);
    assert!(right_set.contains(&1));
    assert!(right_set.contains(&2));
    assert!(right_set.contains(&3));

    // now push_left(5) - should swap with element at partition (now 2)
    p.push_left(5);

    // left should now have [4, 5]
    check_set_equality(p.left().iter().copied(), [4, 5]);

    // right should have [3, 1, 2] - exactly these elements, no more or less
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

    // get both partitions at once
    let (left, right) = p.partitions();

    // check that they contain the expected elements
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

    // get mutable access to both partitions at once
    let (left_mut, right_mut) = p.partitions_mut();

    // modify elements in both partitions
    for item in left_mut.iter_mut() {
        *item *= 10;
    }

    for item in right_mut.iter_mut() {
        *item *= 100;
    }

    // check that modifications were applied
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

    // check len after removing elements
    p.pop_left();
    assert_eq!(p.len(), 2);

    p.pop_right();
    assert_eq!(p.len(), 1);

    // should be empty after all elements are removed
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

    // check with right side
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

    // clear the partition
    p.clear();

    // everything should be empty
    assert!(p.is_empty());
    assert_eq!(p.len(), 0);
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 0);

    // should still work correctly after clearing
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

    // verify initial state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 1);

    // decompose into raw parts
    let (vec, partition) = p.to_raw_parts();

    // check results
    assert_eq!(vec.len(), 3);
    assert_eq!(partition, 2);

    // verify the contents are all present
    let mut elements = vec.clone();
    elements.sort();
    assert_eq!(elements, vec![1, 2, 3]);
}

#[test]
fn test_from_raw_parts() {
    // valid case
    let vec = vec![10, 20, 30, 40];
    let partition = 2;

    let p = Partition::from_raw_parts(vec, partition);
    assert_eq!(p.left(), &[10, 20]);
    assert_eq!(p.right(), &[30, 40]);

    // edge case: empty vector
    let vec = Vec::<i32>::new();
    let partition = 0;

    let p = Partition::from_raw_parts(vec, partition);
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());

    // edge case: all elements in left partition
    let vec = vec![1, 2, 3];
    let partition = 3;

    let p = Partition::from_raw_parts(vec, partition);
    assert_eq!(p.left(), &[1, 2, 3]);
    assert!(p.right().is_empty());

    // edge case: all elements in right partition
    let vec = vec![1, 2, 3];
    let partition = 0;

    let p = Partition::from_raw_parts(vec, partition);
    assert!(p.left().is_empty());
    assert_eq!(p.right(), &[1, 2, 3]);
}

#[test]
#[should_panic(expected = "partition index 4 is out of bounds")]
fn test_from_raw_parts_panic() {
    // invalid case - should panic
    let vec = vec![1, 2, 3];
    let partition = 4; // beyond the vector length

    // this should panic
    let _p = Partition::from_raw_parts(vec, partition);
}

#[test]
fn test_from_raw_parts_unchecked() {
    // valid case
    let vec = vec![10, 20, 30, 40];
    let partition = 2;

    let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    assert_eq!(p.left(), &[10, 20]);
    assert_eq!(p.right(), &[30, 40]);

    // edge case: empty vector
    let vec = Vec::<i32>::new();
    let partition = 0;

    let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());

    // edge case: all elements in left partition
    let vec = vec![1, 2, 3];
    let partition = 3;

    let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    assert_eq!(p.left(), &[1, 2, 3]);
    assert!(p.right().is_empty());

    // edge case: all elements in right partition
    let vec = vec![1, 2, 3];
    let partition = 0;

    let p = unsafe { Partition::from_raw_parts_unchecked(vec, partition) };
    assert!(p.left().is_empty());
    assert_eq!(p.right(), &[1, 2, 3]);

    // we don't test the invalid case (partition > vec.len()) since that would be undefined behavior
}

#[test]
fn test_raw_parts_roundtrip() {
    // create a partition with known state
    let mut original = Partition::new();
    original.push_left(1);
    original.push_left(2);
    original.push_right(3);
    original.push_right(4);

    // convert to raw parts
    let (vec, partition) = original.to_raw_parts();

    // reconstruct from raw parts
    let reconstructed = Partition::from_raw_parts(vec, partition);

    // verify left and right partitions match
    // use set equality since order within partitions might vary
    check_set_equality(reconstructed.left().iter().copied(), [1, 2]);
    check_set_equality(reconstructed.right().iter().copied(), [3, 4]);
}

//
// Additional tests to strengthen test coverage
//

#[test]
fn test_with_non_copy_types() {
    // test Partition with String (non-Copy type)
    let mut p = Partition::new();
    p.push_left("hello".to_string());
    p.push_left("world".to_string());
    p.push_right("foo".to_string());
    p.push_right("bar".to_string());

    // test basic operations
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // check accessing without cloning works
    let left_has_hello = p.left().iter().any(|s| s == "hello");
    let left_has_world = p.left().iter().any(|s| s == "world");
    assert!(left_has_hello);
    assert!(left_has_world);

    // test pops (which move values)
    let popped = p.pop_left();
    assert!(popped.is_some());

    // check the popped value is what we expect
    let popped_val = popped.unwrap();
    assert!(popped_val == "hello" || popped_val == "world");

    // test cloning the partition
    let p_clone = p.clone();
    assert_eq!(p.left().len(), p_clone.left().len());
    assert_eq!(p.right().len(), p_clone.right().len());

    // verify modifying one doesn't affect the other
    p.push_left("another".to_string());
    assert_ne!(p.left().len(), p_clone.left().len());
}

#[test]
fn test_clone_behavior() {
    let mut original = Partition::new();
    original.push_left(1);
    original.push_left(2);
    original.push_right(3);

    // clone the partition
    let cloned = original.clone();

    // verify cloned partition has same content
    check_set_equality(
        original.left().iter().copied(),
        cloned.left().iter().copied(),
    );
    check_set_equality(
        original.right().iter().copied(),
        cloned.right().iter().copied(),
    );

    // modify original
    original.push_left(4);
    original.push_right(5);

    // verify cloned partition remains unchanged
    assert_eq!(cloned.left().len(), 2);
    assert_eq!(cloned.right().len(), 1);

    check_set_equality(cloned.left().iter().copied(), [1, 2]);
    check_set_equality(cloned.right().iter().copied(), [3]);
}

#[test]
fn test_default_behavior() {
    // test the Default implementation
    let p: Partition<i32> = Default::default();

    assert!(p.is_empty());
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_partial_iterator_consumption() {
    // test drain_to_right with partial consumption
    let mut p = Partition::new();
    for i in 0..10 {
        p.push_left(i);
    }

    let mut iter = p.drain_to_right();

    // only consume half the elements
    for _ in 0..5 {
        let _ = iter.next();
    }

    // drop the iterator - remaining elements should still move to right
    drop(iter);

    // check all elements moved to the right
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 10);

    // test drain_to_left with partial consumption
    let mut p = Partition::new();
    for i in 0..10 {
        p.push_right(i);
    }

    let mut iter = p.drain_to_left();

    // only consume some elements
    assert!(iter.next().is_some());
    assert!(iter.next().is_some());

    // drop iterator
    drop(iter);

    // check all elements moved
    assert_eq!(p.left().len(), 10);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_iterator_size_hint_accuracy() {
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);
    p.push_left(3);

    // test size_hint for drain_to_right
    let iter = p.drain_to_right();
    assert_eq!(iter.size_hint(), (3, Some(3)));

    // test size_hint for drain_left
    let mut p = Partition::new();
    p.push_left(1);
    p.push_left(2);

    let iter = p.drain_left();
    assert_eq!(iter.size_hint(), (2, Some(2)));

    // test size_hint for drain_right
    let mut p = Partition::new();
    p.push_right(1);
    p.push_right(2);

    let iter = p.drain_right();
    assert_eq!(iter.size_hint(), (2, Some(2)));
}

#[test]
fn test_partition_invariants() {
    // test that partition index always stays <= inner.len()
    let mut p = Partition::new();

    for i in 0..100 {
        if i % 2 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }

        // check the invariant after each operation
        assert!(p.partition <= p.len());
    }

    // remove elements and check invariant
    for _ in 0..50 {
        if p.left().len() > 0 {
            p.pop_left();
        }
        if p.right().len() > 0 {
            p.pop_right();
        }

        // check the invariant after each operation
        assert!(p.partition <= p.len());
    }
}

#[test]
fn test_boundary_conditions() {
    // create a large partition
    let mut p = Partition::new();

    // add many elements
    for i in 0..1000 {
        if i % 2 == 0 {
            p.push_left(i);
        } else {
            p.push_right(i);
        }
    }

    // check counts
    assert_eq!(p.left().len(), 500);
    assert_eq!(p.right().len(), 500);

    // move all from left to right
    while p.left().len() > 0 {
        p.move_to_right();
    }

    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 1000);

    // move all from right to left
    while p.right().len() > 0 {
        p.move_to_left();
    }

    assert_eq!(p.left().len(), 1000);
    assert_eq!(p.right().len(), 0);
}

#[test]
fn test_debug_contains_expected_elements() {
    // test that the Debug implementation contains all elements
    let mut p = Partition::new();
    p.push_left(42);
    p.push_left(24);
    p.push_right(99);

    // convert debug output to string
    let debug_string = format!("{:?}", p);

    // debug should mention it's a Partition
    assert!(debug_string.contains("Partition"));

    // debug should contain left and right sections
    assert!(debug_string.contains("left"));
    assert!(debug_string.contains("right"));

    // debug should contain all the values (order agnostic)
    assert!(debug_string.contains("42"));
    assert!(debug_string.contains("24"));
    assert!(debug_string.contains("99"));
}

//
// Additional tests for uncovered paths
//

#[test]
fn test_with_capacity_reservations() {
    // test that with_capacity actually reserves the capacity
    let capacity = 100;
    let p: Partition<i32> = Partition::with_capacity(capacity);

    // capacity should be at least what we asked for
    // access internal Partition fields through Deref
    assert!(p.0.inner.capacity() >= capacity);
}

#[test]
fn test_to_raw_parts_empty() {
    // test to_raw_parts on an empty partition
    let p = Partition::<i32>::new();

    let (vec, partition) = p.to_raw_parts();

    assert!(vec.is_empty());
    assert_eq!(partition, 0);

    // recreate partition from parts
    let p2 = Partition::from_raw_parts(vec, partition);
    assert!(p2.is_empty());
    assert_eq!(p2.left().len(), 0);
    assert_eq!(p2.right().len(), 0);
}

#[test]
fn test_left_mut_right_mut_empty() {
    // test left_mut and right_mut with empty partitions
    let mut p = Partition::<i32>::new();

    // empty left partition
    assert!(p.left_mut().is_empty());

    // empty right partition
    assert!(p.right_mut().is_empty());
}

#[test]
fn test_partitions_mut_empty() {
    // test partitions_mut with various empty configurations

    // both partitions empty
    let mut p = Partition::<i32>::new();
    let (left, right) = p.partitions_mut();
    assert!(left.is_empty());
    assert!(right.is_empty());

    // left empty, right has elements
    let mut p = Partition::new();
    p.push_right(1);
    let (left, right) = p.partitions_mut();
    assert!(left.is_empty());
    assert!(!right.is_empty());

    // right empty, left has elements
    let mut p = Partition::new();
    p.push_left(1);
    let (left, right) = p.partitions_mut();
    assert!(!left.is_empty());
    assert!(right.is_empty());
}

#[test]
fn test_iterator_behavior_after_emptying() {
    // test calling next() on iterators after they're empty

    // test drain_to_right
    let mut p = Partition::new();
    p.push_left(1);

    let mut iter = p.drain_to_right();
    assert_eq!(iter.next(), Some(1)); // consume the only element
    assert_eq!(iter.next(), None); // should return None
    assert_eq!(iter.next(), None); // should still return None

    // test drain_left
    let mut p = Partition::new();
    p.push_left(1);

    let mut iter = p.drain_left();
    assert_eq!(iter.next(), Some(1)); // consume the only element
    assert_eq!(iter.next(), None); // should return None
    assert_eq!(iter.next(), None); // should still return None

    // test drain_right
    let mut p = Partition::new();
    p.push_right(1);

    let mut iter = p.drain_right();
    assert_eq!(iter.next(), Some(1)); // consume the only element
    assert_eq!(iter.next(), None); // should return None
    assert_eq!(iter.next(), None); // should still return None
}

#[test]
fn test_zero_sized_types() {
    // test with zero-sized type ()
    let mut p = Partition::<()>::new();

    // push some values
    p.push_left(());
    p.push_left(());
    p.push_right(());

    // check counts
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 1);

    // basic operations
    let moved = p.move_to_right();
    assert_eq!(moved, Some(()));

    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 2);

    // pop values
    assert_eq!(p.pop_left(), Some(()));
    assert_eq!(p.pop_right(), Some(()));

    // should still have one element in right
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 1);
}

#[test]
fn test_complex_method_interactions() {
    // test complex interactions between different methods
    let mut p = Partition::new();

    // mix of operations in specific sequence
    p.push_left(1);
    p.push_right(2);
    p.push_left(3);
    p.push_right(4);

    // move an element right, then pop it
    let moved = p.move_to_right();
    assert!(moved.is_some());
    let popped = p.pop_right();
    assert!(popped.is_some());

    // push to left, then drain right to left
    p.push_left(5);
    let drained: Vec<_> = p.drain_to_left().collect();
    assert!(!drained.is_empty());

    // now everything should be in left
    assert!(p.right().is_empty());
    assert!(!p.left().is_empty());

    // clear and verify empty
    p.clear();
    assert!(p.is_empty());

    // specific sequence that might trigger edge cases
    p.push_left(1);
    p.push_right(2);
    p.pop_left(); // left now empty
    p.push_left(3);
    p.move_to_right(); // left empty again
    p.move_to_left(); // move from right to left

    // verify state
    assert_eq!(p.left().len(), 1);
    assert_eq!(p.right().len(), 1);
}

#[test]
fn test_unchecked_from_raw_parts_preserves_invariants() {
    // test that from_raw_parts_unchecked preserves invariants
    // SAFETY: We're only calling this with valid data

    let vec = vec![1, 2, 3, 4];
    let partition = 2;

    // valid use of from_raw_parts_unchecked
    let p = unsafe { Partition::from_raw_parts_unchecked(vec.clone(), partition) };

    // verify partition index is preserved
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);

    // verify we can perform operations as expected
    let left_items: Vec<_> = p.left().to_vec();
    let right_items: Vec<_> = p.right().to_vec();

    // check that vectors have the right elements
    check_set_equality(left_items, [1, 2]);
    check_set_equality(right_items, [3, 4]);

    // test with edge cases (empty vector, partition=0)
    let empty_vec = Vec::<i32>::new();
    let p = unsafe { Partition::from_raw_parts_unchecked(empty_vec, 0) };
    assert!(p.left().is_empty());
    assert!(p.right().is_empty());
}

#[test]
fn test_spare_capacity_mut() {
    // create a partition with specific capacity
    let mut p: Partition<usize> = Partition::with_capacity(10);

    // add some elements
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // capture current state
    let len_before = p.len();
    let cap_before = p.capacity();
    let spare_len = cap_before - len_before;

    {
        // get spare capacity in its own scope
        let spare = p.spare_capacity_mut();

        // verify size of spare capacity
        assert_eq!(spare.len(), spare_len);
        assert!(spare.len() >= 7); // should have at least 7 spare slots

        // write to the spare capacity
        spare[0].write(10);
        spare[1].write(20);
        spare[2].write(30);

        // spare is dropped at the end of this scope
    }

    // initialize those elements
    unsafe {
        p.set_len(len_before + 3);
    }

    // elements should now be in the right partition
    assert_eq!(p.left().len(), 2);
    check_set_equality(p.left().iter().copied(), [1, 2]);

    assert_eq!(p.right().len(), 4);

    // check that the right partition contains both the original and new elements
    let mut right_elements = p.right().to_vec();
    right_elements.sort(); // sort for predictable comparison
    assert_eq!(right_elements, vec![3, 10, 20, 30]);

    // test with empty partition
    let mut empty_p: Partition<usize> = Partition::with_capacity(5);
    let capacity = empty_p.capacity();

    {
        // get spare capacity for empty partition in its own scope
        let empty_spare = empty_p.spare_capacity_mut();
        assert_eq!(empty_spare.len(), capacity);

        // write to spare capacity
        empty_spare[0].write(100);

        // empty_spare is dropped at the end of this scope
    }

    unsafe {
        empty_p.set_len(1);
    }

    assert_eq!(empty_p.left().len(), 0);
    assert_eq!(empty_p.right().len(), 1);
    assert_eq!(empty_p.right()[0], 100);
}

#[test]
fn test_set_len() {
    // create a partition with capacity
    let mut p: Partition<usize> = Partition::with_capacity(10);

    // ensure the capacity is at least 3
    assert!(p.capacity() >= 3);

    // test extending empty partition
    let ptr = p.inner.as_mut_ptr();
    unsafe {
        // initialize first 3 elements directly
        ptr.write(10);
        ptr.add(1).write(20);
        ptr.add(2).write(30);

        // set the length to include these initialized elements
        p.set_len(3);
    }

    // verify results
    assert_eq!(p.len(), 3);
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 3);
    check_set_equality(p.right().iter().copied(), [10, 20, 30]);

    // test extending a partition with existing elements
    let mut p2: Partition<usize> = Partition::with_capacity(10);
    p2.push_left(1);
    p2.push_left(2);
    p2.push_right(3);
    let len_before = p2.len();

    // verify we have enough capacity
    assert!(p2.capacity() >= len_before + 2);

    // get ptr to end of current data
    let ptr = unsafe { p2.inner.as_mut_ptr().add(len_before) };
    unsafe {
        // initialize two more elements directly
        ptr.write(40);
        ptr.add(1).write(50);

        // set the new length
        p2.set_len(len_before + 2);
    }

    // verify results
    assert_eq!(p2.len(), len_before + 2);
    assert_eq!(p2.left().len(), 2);
    assert_eq!(p2.right().len(), 3);

    // right partition should contain both original and new elements
    let right_elements: Vec<_> = p2.right().to_vec();
    assert_eq!(right_elements.len(), 3);
    assert!(right_elements.contains(&3));
    assert!(right_elements.contains(&40));
    assert!(right_elements.contains(&50));
}

#[test]
fn test_reserve_and_reserve_exact() {
    // test reserve
    let mut p1: Partition<usize> = Partition::new();
    assert_eq!(p1.capacity(), 0);

    p1.reserve(10);
    assert!(p1.capacity() >= 10);

    // reserving more should increase capacity
    let cap_before = p1.capacity();
    p1.reserve(20);
    // the actual implementation might not add exactly 20 more,
    // just ensure the capacity increased
    assert!(p1.capacity() >= cap_before);

    // test reserve_exact
    let mut p2: Partition<usize> = Partition::new();
    p2.reserve_exact(10);
    assert!(p2.capacity() >= 10);

    // fill the partition partially
    for i in 0..5 {
        p2.push_left(i);
    }

    // reserve more exactly
    p2.reserve_exact(10);
    assert!(p2.capacity() >= p2.len() + 10);

    // test with different distributions of elements
    let mut p3: Partition<usize> = Partition::new();
    p3.push_left(1);
    p3.push_left(2);
    p3.push_right(3);
    p3.push_right(4);

    let len_before = p3.len();
    let cap_before = p3.capacity();

    // reserve more space
    p3.reserve(20);

    // capacity should increase
    assert!(p3.capacity() >= cap_before + 20);

    // length and partition distribution should remain unchanged
    assert_eq!(p3.len(), len_before);
    assert_eq!(p3.left().len(), 2);
    assert_eq!(p3.right().len(), 2);

    // elements should remain in their original partitions
    check_set_equality(p3.left().iter().copied(), [1, 2]);
    check_set_equality(p3.right().iter().copied(), [3, 4]);
}

#[test]
fn test_spare_capacity_mut_with_set_len_interaction() {
    // this test ensures that spare_capacity_mut and set_len work well together
    let mut p: Partition<usize> = Partition::with_capacity(20);

    // add some elements to both partitions
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);
    p.push_right(4);

    // initial state
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 2);
    assert_eq!(p.len(), 4);

    // store current length
    let current_len = p.len();

    {
        // get spare capacity in its own scope
        let spare = p.spare_capacity_mut();

        // initialize 5 elements in spare capacity
        for i in 0..5 {
            spare[i].write(100 + i);
        }

        // spare is dropped at the end of this scope
    }

    // extend the length to include these 5 new elements
    unsafe {
        p.set_len(current_len + 5);
    }

    // verify state after extension
    assert_eq!(p.len(), 9);
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 7);

    // verify original elements are still in their partitions
    assert!(p.left().contains(&1));
    assert!(p.left().contains(&2));

    // verify new elements are in right partition
    for i in 0..5 {
        assert!(p.right().contains(&(100 + i)));
    }

    // verify that set_len doesn't change the partition index
    let partition_idx_before = p.partition;
    let current_len = p.len();

    unsafe {
        // pretend to initialize more but don't actually do it
        // just testing the partition index behavior
        p.set_len(current_len + 1);
    }
    assert_eq!(
        p.partition, partition_idx_before,
        "Partition index shouldn't change with set_len"
    );
}

#[test]
#[should_panic]
fn test_set_len_debug_assertion() {
    // this test verifies the debug_assert in set_len
    let mut p: Partition<usize> = Partition::with_capacity(5);

    // this should trigger the debug assertion because new_len > capacity
    unsafe {
        p.set_len(10);
    }
}
