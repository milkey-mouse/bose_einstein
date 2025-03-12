# Bose Einstein

A data structure that efficiently partitions elements into left and right sets.

## Overview

`Partition<T>` is a data structure that maintains a collection of elements partitioned into two sets: left and right. Elements can be efficiently moved between the sets, and the relative order within each set is not guaranteed to be preserved.

## Features

- Efficient push and pop operations for both left and right sets
- Ability to move elements between partitions
- Drain iterators for consuming elements from either partition
- Specialized iterators for moving elements from one partition to another
- Raw parts access for advanced use cases

## Example

```rust
use bose_einstein::Partition;

fn main() {
    // Create a new partition
    let mut p = Partition::new();

    // Add elements to left and right partitions
    p.push_left(1);
    p.push_left(2);
    p.push_right(3);

    // Access the elements in each partition
    assert_eq!(p.left().len(), 2);
    assert_eq!(p.right().len(), 1);

    // Move elements between partitions
    let moved = p.move_to_right();
    assert!(moved.is_some());

    // Drain elements from a partition
    let left_elements: Vec<_> = p.drain_left().collect();

    // Check the state after operations
    assert_eq!(p.left().len(), 0);
    assert_eq!(p.right().len(), 2);
}
```

## License

This project is licensed under CC0-1.0.
