# Vec Methods Analysis for Partition

## Introduction and Context

This document provides a comprehensive analysis of all methods and traits defined on `Vec<T>` in the Rust standard library and evaluates their applicability to the `Partition<T>` data structure in the bose_einstein crate.

### Purpose of This Analysis

The bose_einstein crate implements a `Partition<T>` data structure that efficiently divides elements into two distinct sets: a "left" partition and a "right" partition. It provides O(1) operations for pushing, popping, and moving elements between these partitions. The current documentation states that "most operations on Vec<T> are supported by Partition<T>", but this is not entirely accurate as there are many Vec methods that do not yet have Partition equivalents.

This analysis aims to:

1. Identify all methods and traits from Vec that are not yet implemented for Partition
2. Evaluate whether each method would make sense for Partition's two-sided design
3. Prioritize which methods should be added to make Partition more complete
4. Provide detailed reasoning about how each method should be adapted for Partition's design

### About the Partition Data Structure

`Partition<T>` is designed to efficiently maintain a collection of elements divided into "left" and "right" partitions. Internally, it uses a single `Vec<T>` with a partition index that divides the elements:

```rust
pub struct Partition<T> {
    inner: Vec<T>,
    partition: usize,
}
```

Elements from index 0 to `partition-1` belong to the left partition, while elements from index `partition` to the end belong to the right partition. This design provides several benefits:

1. O(1) operations for moving elements between partitions (simply adjusting the partition index)
2. Space efficiency (only one allocation for both partitions)
3. Cache locality (elements are stored contiguously in memory)

However, this design also presents challenges when adapting some Vec methods, as operations may affect how elements are distributed between partitions.

### Key Design Considerations

When adapting Vec methods to Partition, several design decisions must be made:

1. **Naming Convention**: Most methods are split into left/right variants (e.g., `push_left`, `push_right`) to operate on the respective partitions.

2. **Order Preservation**: A key characteristic of Partition is that order within each partition is not guaranteed to be preserved - they are essentially sets, not lists. This influences how methods like `insert`, `remove`, etc. should be implemented.

3. **Trait Implementation Semantics**: For traits like `IntoIterator`, `FromIterator`, `Extend`, etc., decisions must be made about how elements are distributed between the left and right partitions.

4. **Low-Level Access**: For methods that provide raw pointer access or manipulate memory directly, considerations must be made for Partition's two-sided structure.

### How to Use This Document

For each Vec method or trait, this document provides:

- **STABLE?** - Whether the method is part of Vec's stable API
- **EXISTS?** - Whether Partition already has an equivalent implementation
- **ADD?** - The priority level for adding this method to Partition (High, Medium, Low, or N/A)
- **Notes** - Detailed explanation about the method, its applicability to Partition, and implementation considerations

At the end of the document, there is a prioritized list of methods to implement, categorized by importance.

This analysis is meant to serve as a comprehensive guide for enhancing the Partition data structure to more fully support the interface provided by Vec, while respecting Partition's unique two-sided design.

## Constructors and Initialization

### `new()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

The `new()` method creates an empty Vec without allocating memory. Partition already implements this method with identical semantics - it creates an empty Partition with no elements in either the left or right partitions. This is a core constructor and is already appropriately implemented.

### `with_capacity(capacity)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

The `with_capacity(capacity)` method creates an empty Vec with preallocated space for at least `capacity` elements. Partition already implements this method with identical semantics, allocating the same amount of space. This preallocated space can be used for either partition, which matches the intended behavior.

### `new_in(alloc)`

**STABLE?** ❌  
**EXISTS?** ✅  
**ADD?** N/A  

This allocator-aware constructor is unstable in Vec but is already implemented in Partition under the `allocator_api` feature flag. It allows creating a Partition with a custom allocator, which is consistent with Vec's behavior.

### `with_capacity_in(capacity, alloc)`

**STABLE?** ❌  
**EXISTS?** ✅  
**ADD?** N/A  

This allocator-aware constructor with capacity is unstable in Vec but is already implemented in Partition under the `allocator_api` feature flag. It allows creating a Partition with preallocated space using a custom allocator.

### `try_with_capacity(capacity)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable fallible constructor for Vec that returns a Result instead of panicking when capacity limits are reached. Partition doesn't implement this, and it's a low priority since it's an unstable API and fallible constructors are rarely needed in practice.

### `try_with_capacity_in(capacity, alloc)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable allocator-aware fallible constructor. For the same reasons as `try_with_capacity`, this is a low priority addition for Partition.

### `from_raw_parts(ptr, length, capacity)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

While both Vec and Partition have methods with this name, they serve different purposes. Vec's version takes raw pointers and creates a vector directly from memory, while Partition's takes a Vec and a partition index. This difference is appropriate since Partition is built on top of Vec and offers a higher-level abstraction.

### `from_parts(ptr, length, capacity)` (NonNull version)

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable NonNull pointer version of `from_raw_parts` in Vec. Since Partition doesn't expose low-level pointer operations in its API design, adding this method is a low priority.

### `from_raw_parts_in(ptr, length, capacity, alloc)`

**STABLE?** ❌  
**EXISTS?** ✅  
**ADD?** N/A  

This is an unstable allocator-aware version of `from_raw_parts`. Partition implements a version under its allocator_api feature, though with a different signature appropriate to its abstraction level.

### `from_parts_in(ptr, length, capacity, alloc)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable allocator-aware version of `from_parts` using NonNull pointers. Adding this to Partition is low priority due to its unstable status and low-level nature.

### `from_raw_parts_unchecked(inner, partition)`

**STABLE?** N/A  
**EXISTS?** ✅  
**ADD?** N/A  

This is a Partition-specific unsafe constructor that skips bounds checking on the partition index. It doesn't exist in Vec, but makes sense for Partition's design as it's consistent with Rust's pattern of providing checked and unchecked variants.

### `from_elem(elem, n)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This function (hidden but used by the `vec!` macro) creates a Vec with `n` copies of `elem`. Partition could benefit from this, but would need to decide whether the elements go into the left partition, right partition, or offer both variants. Medium priority since it would improve consistency with Vec.

### `from_elem_in(elem, n, alloc)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable allocator-aware version of `from_elem`. Low priority due to its unstable status and specialized use case.

## Raw Parts and Conversions

### `into_raw_parts()`

**STABLE?** ❌  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `into_raw_parts()` is unstable and decomposes a Vec into a raw pointer, length, and capacity. Partition has a stable analog called `to_raw_parts()` which returns the underlying Vec and partition index. This difference in naming and signature is appropriate for Partition's abstraction level.

### `into_parts()` (NonNull version)

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable NonNull version of `into_raw_parts`. Adding this to Partition is low priority due to its unstable status and low-level pointer manipulation.

### `into_raw_parts_with_alloc()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable allocator-aware version of `into_raw_parts`. Low priority for the same reasons as other unstable allocator methods.

### `into_parts_with_alloc()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable allocator-aware version of `into_parts` using NonNull pointers. Low priority for the same reasons as other unstable allocator methods.

### `into_boxed_slice()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method converts a Vec into a Box<[T]>, shrinking the allocation to match the length. For Partition, this would conceptually need to return two boxed slices (one for each partition). This is a low priority addition as the use case is specialized and the semantics aren't straightforward for a partitioned structure.

### `into_flattened()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This specialized method converts a Vec<[T; N]> into a Vec<T>. This is very specialized and not clearly applicable to Partition's use cases, so it's a low priority addition.

## Capacity and Size Management

### `capacity()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method returns the number of elements the vector can hold without reallocating. Partition implements this method with identical semantics, returning the capacity of the underlying Vec.

### `reserve(additional)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method ensures that the vector can hold at least `additional` more elements. Partition implements this with identical semantics, reserving space that can be used by either partition.

### `reserve_exact(additional)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Similar to `reserve()` but more precise about the exact amount of space needed. Partition implements this with identical semantics.

### `try_reserve(additional)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This is a fallible version of `reserve()` that returns a Result. Partition implements this with identical semantics.

### `try_reserve_exact(additional)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This is a fallible version of `reserve_exact()` that returns a Result. Partition implements this with identical semantics.

### `shrink_to_fit()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method shrinks the capacity to match the length. Partition implements this with identical semantics, operating on the underlying Vec.

### `shrink_to(min_capacity)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method shrinks the capacity with a minimum bound. Partition implements this with identical semantics.

### `len()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method returns the number of elements in the vector. Partition implements this with identical semantics, returning the total number of elements across both partitions.

### `is_empty()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method checks if the vector contains no elements. Partition implements this with identical semantics, checking if both partitions are empty.

## Element Access

### `as_slice()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `as_slice()` returns a slice to the entire vector. Partition has analogous methods `left()` and `right()` that return slices to each partition, which is more appropriate for Partition's two-sided design.

### `as_mut_slice()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `as_mut_slice()` returns a mutable slice to the entire vector. Partition has analogous methods `left_mut()` and `right_mut()` that return mutable slices to each partition, which is more appropriate for Partition's design.

### `as_ptr()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method returns a raw pointer to the vector's buffer. Partition could add `left_as_ptr()` and `right_as_ptr()` to provide similar low-level access to each partition. Medium priority as it would be useful for advanced use cases but isn't commonly needed.

### `as_mut_ptr()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method returns a mutable raw pointer to the vector's buffer. Partition could add `left_as_mut_ptr()` and `right_as_mut_ptr()` to provide similar functionality. Medium priority for the same reasons as `as_ptr()`.

### `as_non_null()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable method that returns a NonNull pointer. Low priority due to its unstable status and specialized use case.

### `allocator()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable method that returns a reference to the allocator. Low priority due to its unstable status and specialized use case.

### `set_len(new_len)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `set_len()` unsafely changes the vector's length. Partition has an analogous method `set_partition()` which unsafely changes the partition index. This difference is appropriate for Partition's design as changing the partition index effectively changes how many elements are in each partition.

## Element Insertion and Removal

### `swap_remove(index)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `swap_remove(index)` removes an element at the specified index and replaces it with the last element. Partition provides `swap_remove_left(index)` and `swap_remove_right(index)` which do the same for each partition. This split implementation is appropriate for Partition's design.

### `insert(index, element)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This method inserts an element at the specified position, shifting all elements after it to the right. Partition should implement `insert_left(index, element)` and `insert_right(index, element)` to provide similar functionality for each partition. High priority as these are fundamental operations for collections.

### `remove(index)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This method removes and returns the element at the specified position, shifting all elements after it to the left. Partition should implement `remove_left(index)` and `remove_right(index)` to provide similar functionality for each partition. High priority as these are fundamental operations for collections.

### `push(value)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `push(value)` adds an element to the end of the vector. Partition provides `push_left(value)` and `push_right(value)` which add elements to each partition. This split implementation is appropriate for Partition's design.

### `push_within_capacity(value)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable method that tries to push without reallocating. Low priority due to its unstable status and specialized use case.

### `pop()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Vec's `pop()` removes and returns the last element. Partition provides `pop_left()` and `pop_right()` which do the same for each partition. This split implementation is appropriate for Partition's design.

### `pop_if(predicate)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method (added in Rust 1.86) pops an element if it satisfies a predicate. Partition should implement `pop_left_if(predicate)` and `pop_right_if(predicate)` to provide similar functionality. Medium priority as it's a newer convenience method rather than core functionality.

### `append(other)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method moves all elements from another vector into this one. Partition implements this by preserving the partitioning - elements from the left partition of `other` go into the left partition of `self`, and similarly for the right partition. This behavior is appropriate for Partition's design.

### `clear()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method removes all elements from the vector. Partition implements this with identical semantics, clearing both partitions.

### `truncate(len)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method shortens the vector to the specified length. Partition could implement `truncate_left(len)` and `truncate_right(len)` to provide similar functionality for each partition. Medium priority as it's useful but not as fundamental as insert/remove.

### `split_off(at)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method splits the vector at the given index, returning the second half. Partition could implement a similar method that splits a partition into two separate Partition instances. Medium priority as it's useful for some operations but not as commonly needed as other methods.

### `resize(new_len, value)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method resizes the vector to the specified length, either truncating or filling with copies of a value. The semantics for a partitioned structure aren't clear - which partition should grow or shrink? Low priority due to this conceptual mismatch.

### `resize_with(new_len, f)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

Similar to `resize()` but fills with values from a closure. Low priority for the same reasons as `resize()`.

### `extend_with(n, value)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper method in Vec. Low priority as it's not part of the public API.

### `append_elements(other)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper method in Vec. Low priority as it's not part of the public API.

## Filtering and Transformation

### `retain(f)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method retains only the elements specified by a predicate. Partition provides `retain_left(f)` and `retain_right(f)` which do the same for each partition. This split implementation is appropriate for Partition's design.

### `retain_mut(f)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This is the mutable version of `retain()` that passes a mutable reference to the predicate. Partition should implement `retain_left_mut(f)` and `retain_right_mut(f)` to provide the same functionality. High priority as it's a standard collection operation that's missing from the current implementation.

### `dedup()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method removes consecutive duplicate elements. Partition should implement `dedup_left()` and `dedup_right()` to provide this functionality for each partition. Medium priority as it's useful but not as fundamental as other operations.

### `dedup_by_key(key)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method removes consecutive elements that resolve to the same key. Partition should implement versions for each partition. Medium priority, similar to `dedup()`.

### `dedup_by(same_bucket)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method removes consecutive elements that satisfy an equality relation. Partition should implement versions for each partition. Medium priority, similar to `dedup()`.

### `extract_if(range, filter)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method creates an iterator which uses a closure to determine if an element should be removed. Partition should implement `extract_if_left(filter)` and `extract_if_right(filter)`. Medium priority as it's useful but not as fundamental as other operations.

## Iteration and Collection Operations

### `drain(range)`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method creates a draining iterator that removes and yields elements. Partition provides `drain_left()` and `drain_right()` which drain entire partitions. This implementation is appropriate, though Partition could potentially add range-based versions in the future.

### `splice(range, replace_with)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method replaces a range of elements with elements from an iterator. Implementing this for Partition would be complex due to the two-sided structure, but could be valuable. Medium priority due to this complexity.

### `leak()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method consumes and leaks the Vec, returning a mutable reference to the contents. For Partition, this would need to return two slices. Low priority due to its specialized use case.

### `spare_capacity_mut()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method returns a mutable slice of uninitialized memory. Low priority for Partition as it's a specialized low-level operation.

### `split_at_spare_mut()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an unstable method that returns content as a slice and spare capacity. Low priority due to its unstable status and specialized use case.

### `split_at_spare_mut_with_len()`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper method in Vec. Low priority as it's not part of the public API.

### `extend_from_slice(other)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This method extends a vector with the contents of a slice. Partition should implement `extend_left_from_slice(slice)` and `extend_right_from_slice(slice)` to provide similar functionality. High priority as it's an efficient way to add multiple elements, which is a common operation.

### `extend_from_within(src)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method extends a vector with elements copied from a range within itself. Partition could implement versions for each partition, potentially with options to extend one partition from another. Medium priority as it's useful but has some complexity in the partitioned context.

### `extend_desugared(iterator)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper method in Vec. Low priority as it's not part of the public API.

### `extend_trusted(iterator)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper method in Vec. Low priority as it's not part of the public API.

### `spec_extend_from_within(src)`

**STABLE?** ❌  
**EXISTS?** ❌  
**ADD?** Low  

This is an internal helper trait method in Vec. Low priority as it's not part of the public API.

## Partition-Specific Methods

### `move_to_left()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method moves an element from the right partition to the left. It has no analog in Vec but is central to Partition's functionality.

### `move_to_right()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method moves an element from the left partition to the right. It has no analog in Vec but is central to Partition's functionality.

### `drain_to_left()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method drains elements from the right partition to the left. It has no analog in Vec but is a useful extension of Partition's core functionality.

### `drain_to_right()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method drains elements from the left partition to the right. It has no analog in Vec but is a useful extension of Partition's core functionality.

### `partitions()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method returns both partitions as a tuple of slices. It has no analog in Vec but is a convenient way to access both partitions at once.

### `partitions_mut()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This Partition-specific method returns both partitions as a tuple of mutable slices. It has no analog in Vec but is a convenient way to modify both partitions at once.

## Trait Implementations

### `Clone`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

The Clone trait is already implemented for Partition, with appropriate semantics.

### `Clone::clone_from`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is an optimization for cloning when the destination already exists. Implementing it for Partition would improve performance in some cases. Medium priority as it's an optimization rather than new functionality.

### `Drop`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

Partition inherits Drop behavior from its inner Vec, which is appropriate.

### `Default`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

The Default trait is already implemented for Partition, creating an empty partition.

### `Debug`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

The Debug trait is already implemented for Partition, showing both left and right partitions.

### `Hash`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

The Hash trait allows collections to be used as keys in hash maps. Partition should implement this by hashing both partitions. High priority as it's a standard trait for collections.

### `Deref/DerefMut`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

These traits allow transparent access to the underlying type. For Partition, it's unclear what they should deref to since there are two partitions. Low priority due to this conceptual mismatch.

### `Index/IndexMut`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

These traits allow using the index operator []. For Partition, it's unclear how indexing would work with two partitions. Low priority due to this conceptual mismatch.

### `PartialEq/Eq`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

These traits allow comparing collections for equality. Partition should implement them, comparing left with left and right with right. High priority as they're standard traits for collections.

### `PartialOrd/Ord`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

These traits allow comparing collections for ordering. Partition could implement them using lexicographical comparison of partitions. Medium priority as they're less commonly used than equality comparisons.

### `FromIterator`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This trait allows creating a collection from an iterator. For Partition, this would require deciding whether elements go into the left or right partition. Medium priority as it's useful but requires design decisions.

### `IntoIterator`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This trait allows using a collection in for loops. However, for Partition there's an ambiguity about which partition should be iterated over. Instead of implementing `IntoIterator` directly on `Partition<T>`, we should provide wrapper types like `Left<T>` and `Right<T>` with methods `into_iter_left()` and `into_iter_right()` that make it explicit which partition is being consumed. This approach also aligns with common use cases where one partition contains initialized data and the other contains uninitialized/freed data. High priority as these are fundamental operations for collections.

### `Extend`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This trait allows adding multiple elements to a collection. For Partition, this would require deciding whether elements go into the left or right partition. Medium priority for the same reasons as `FromIterator`.

### `Extend<&'a T>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This is a specialized version of Extend for references. Medium priority for the same reasons as `Extend`.

### `AsRef<Vec<T>>` / `AsMut<Vec<T>>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

These traits allow viewing a type as a Vec. For Partition, this doesn't align well with the two-partition design. Low priority due to this conceptual mismatch.

### `AsRef<[T]>` / `AsMut<[T]>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

These traits allow viewing a type as a slice. For Partition, this doesn't align well with the two-partition design. Low priority due to this conceptual mismatch.

## From/Into Conversions

### `From<&[T]>` / `From<&mut [T]>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

These allow creating a collection from slices. For Partition, these would require deciding which partition the elements go into (likely left by default). Medium priority as they're useful standard conversions.

### `From<&[T; N]>` / `From<&mut [T; N]>` / `From<[T; N]>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

These allow creating a collection from arrays. For Partition, these would have the same considerations as the slice conversions. Medium priority for the same reasons.

### `From<Cow<'a, [T]>>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This allows creating a collection from a clone-on-write slice. For Partition, this would have the same considerations as other slice conversions. Medium priority for the same reasons.

### `From<Box<[T]>>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This allows creating a collection from a boxed slice. For Partition, this would have the same considerations as other slice conversions. Medium priority for the same reasons.

### `From<Vec<T>>` for `Box<[T]>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This conversion is from Vec to Box<[T]>, not to Vec. For Partition, this could potentially convert to a tuple of boxed slices, but the use case is specialized. Low priority.

### `From<&str>` for `Vec<u8>`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This specialized conversion is for creating a byte vector from a string. For Partition, it's unclear which partition the bytes should go into. Low priority due to this specialized nature.

### `TryFrom<Vec<T>>` for `[T; N]`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This conversion is from Vec to a fixed-size array, not to Vec. It doesn't make conceptual sense for Partition with its two partitions. Low priority.

## Additional Collection Methods

### `extend_one(item)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is part of the Extend trait implementation for Vec. It's used to add a single item to a collection and can be optimized separately from the general extend method. For Partition, we would need to decide whether to add to the left or right partition. Medium priority as it's a specialized optimization.

### `extend_one_unchecked(item)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This is an unsafe version of `extend_one` that skips bounds checking. Low priority due to its specialized nature and the need to make design decisions about which partition to extend.

### `extend_reserve(additional)`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is called by the extend implementation to reserve space for additional elements. For Partition, implementing this could improve performance when extending. Medium priority as it's an optimization rather than new functionality.

### `into_iter()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This method is part of the `IntoIterator` trait and converts a collection into an iterator. For Partition, this would need to decide whether to iterate over the left partition first, then the right, or provide separate iterators for each. High priority as it's a fundamental trait for collections.

### `fmt()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method is part of the `Debug` and `Display` traits and formats the collection for output. Partition already implements the Debug trait which uses this method.

### `as_mut()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `AsMut` trait and allows converting the collection to a mutable reference of another type. For Partition with its two-sided structure, this isn't a natural fit. Low priority due to this conceptual mismatch.

### `as_ref()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `AsRef` trait and allows converting the collection to a reference of another type. For Partition with its two-sided structure, this isn't a natural fit. Low priority due to this conceptual mismatch.

### `cmp()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is part of the `Ord` trait and compares two collections for ordering. For Partition, this would likely compare left partitions first, then right partitions. Medium priority as it's less commonly used than equality comparisons.

### `partial_cmp()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is part of the `PartialOrd` trait and compares two collections for partial ordering. Medium priority for the same reasons as `cmp()`.

### `from_iter()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

This method is part of the `FromIterator` trait and creates a collection from an iterator. For Partition, this would require deciding which partition the elements go into. Medium priority as it's useful but requires design decisions.

### `try_from()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `TryFrom` trait in Vec and is used for fallible conversions. For Partition, the semantics aren't clear. Low priority due to this conceptual mismatch.

### `hash()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** High  

This method is part of the `Hash` trait and computes a hash value for the collection. For Partition, this would need to hash both the left and right partitions. High priority as it's necessary for using Partition in hash-based collections.

### `deref()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `Deref` trait and allows for transparent dereferencing to the underlying type. For Partition with its two partitions, it's unclear what this should deref to. Low priority due to this conceptual mismatch.

### `deref_mut()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `DerefMut` trait and allows for transparent mutable dereferencing to the underlying type. For Partition with its two partitions, it's unclear what this should deref to. Low priority due to this conceptual mismatch.

### `index()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `Index` trait and allows for indexing into the collection using the `[]` operator. For Partition with its two partitions, the semantics of indexing are unclear. Low priority due to this conceptual mismatch.

### `index_mut()`

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Low  

This method is part of the `IndexMut` trait and allows for mutable indexing into the collection using the `[]` operator. For Partition with its two partitions, the semantics of indexing are unclear. Low priority due to this conceptual mismatch.

### `clone()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method is part of the `Clone` trait and creates a copy of the collection. Partition already implements this trait appropriately.

### `default()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method is part of the `Default` trait and creates an empty collection. Partition already implements this trait appropriately.

### `drop()`

**STABLE?** ✅  
**EXISTS?** ✅  
**ADD?** N/A  

This method is part of the `Drop` trait and handles cleanup when the collection is dropped. Partition inherits this from its inner Vec, which is appropriate.

### `from()` (multiple implementations)

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

These methods are part of various `From` trait implementations for creating a Vec from different sources. For Partition, these would require deciding which partition the elements go into (likely left by default). Medium priority as they're useful standard conversions.

### `extend()` (multiple implementations)

**STABLE?** ✅  
**EXISTS?** ❌  
**ADD?** Medium  

These methods are part of various `Extend` trait implementations for adding multiple elements to a collection. For Partition, these would require deciding which partition to extend (likely left by default). Medium priority as they're useful but require design decisions.

## Implementation Priorities

### High Priority

1. `insert_left()/insert_right()` - Basic operations for inserting at index
2. `remove_left()/remove_right()` - Basic operations for removing at index
3. `retain_left_mut()/retain_right_mut()` - Mutable filtering operations
4. `extend_from_slice_left()/extend_from_slice_right()` - Efficient slice extension
5. `Hash` implementation - For use in collections
6. `PartialEq/Eq` implementation - Basic comparison
7. `IntoIterator` implementation - Standard collection trait

### Medium Priority

1. `dedup_*` methods - Useful for removing duplicates
2. `pop_left_if()/pop_right_if()` - Conditional popping
3. `truncate_left()/truncate_right()` - Shortening partitions
4. `extract_if_*` - Filtered extraction
5. `as_ptr()/as_mut_ptr()` - Low-level pointer access
6. `extend_one()/extend_reserve()` - Extend optimizations
7. Various `From<>` implementations - Conversion from other types

### Low Priority

1. Methods related to unstable features
2. Methods with unclear semantics in a partitioned context
3. Internal helper methods (like `extend_one_unchecked()`)
4. Highly specialized conversions