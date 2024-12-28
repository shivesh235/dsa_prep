Arrays are a foundational data structure used to store a collection of elements (usually of the same type). Arrays can be categorized into **static arrays** and **dynamic arrays**, each with distinct features. Below is an explanation of both types along with their implementation in Python.

---

### **1. Static Array**
A **static array** has a fixed size defined at the time of its creation. Once the size is set, it cannot be changed, and the array has a predefined amount of memory allocated.

#### Features:
- **Fixed Size**: The size must be known at compile or initialization time.
- **Efficient Memory**: Memory is allocated once, avoiding frequent reallocations.
- **Fast Access**: Random access is possible in O(1) time due to contiguous memory allocation.

#### Implementation in Python:
Python does not have native static arrays, but the `array` module or libraries like `numpy` can be used to simulate static arrays.

Using the `array` module:
```python
import array

# Create a static array of integers
static_array = array.array('i', [1, 2, 3, 4, 5])

# Accessing elements
print(static_array[2])  # Output: 3

# Modifying an element
static_array[2] = 10
print(static_array)  # Output: array('i', [1, 2, 10, 4, 5])

# Static array size is fixed; appending or resizing directly is not supported.
```

Using `numpy` (preferred for advanced use cases):
```python
import numpy as np

# Create a static array of integers
static_array = np.array([1, 2, 3, 4, 5], dtype=int)

# Accessing elements
print(static_array[2])  # Output: 3

# Modifying an element
static_array[2] = 10
print(static_array)  # Output: [ 1  2 10  4  5 ]
```

---

### **2. Dynamic Array**
A **dynamic array** can grow or shrink as needed. It allocates additional memory dynamically when more elements are added.

#### Features:
- **Resizable**: Grows or shrinks as needed, making it more flexible than static arrays.
- **Efficient Operations**: Append, insert, and delete operations are supported.
- **Amortized Time Complexity**: Resizing requires reallocation, but this happens less frequently.

#### Implementation in Python:
Python’s built-in `list` is a dynamic array implementation.

```python
# Create a dynamic array (Python list)
dynamic_array = [1, 2, 3, 4, 5]

# Adding elements (append)
dynamic_array.append(6)
print(dynamic_array)  # Output: [1, 2, 3, 4, 5, 6]

# Inserting elements at a specific index
dynamic_array.insert(2, 10)
print(dynamic_array)  # Output: [1, 2, 10, 3, 4, 5, 6]

# Removing elements
dynamic_array.pop()  # Removes the last element
print(dynamic_array)  # Output: [1, 2, 10, 3, 4, 5]

# Resizing happens automatically when the capacity is exceeded.
```

---

### **Comparison**

| Feature            | Static Array                      | Dynamic Array                 |
|--------------------|-----------------------------------|-------------------------------|
| **Size**           | Fixed at initialization           | Can grow or shrink dynamically |
| **Memory Usage**   | Efficient, as size is preallocated | Less efficient due to resizing overhead |
| **Operations**     | Limited to fixed size             | Flexible (append, insert, delete) |
| **Access Speed**   | O(1) (random access)              | O(1) (random access)           |
| **Implementation** | `array` or `numpy` in Python      | Built-in `list` in Python      |

---

### **Static Array Implementation (Custom Class)**
For educational purposes, you can implement a static array manually.

```python
class StaticArray:
    def __init__(self, size):
        self.size = size
        self.array = [None] * size  # Preallocate memory

    def __setitem__(self, index, value):
        if index >= self.size:
            raise IndexError("Index out of bounds")
        self.array[index] = value

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of bounds")
        return self.array[index]

    def __repr__(self):
        return str(self.array)

# Example usage
static_array = StaticArray(5)
static_array[0] = 10
static_array[1] = 20
print(static_array)  # Output: [10, 20, None, None, None]
```

---

### **Dynamic Array Implementation (Custom Class)**
A custom implementation for a dynamic array would mimic Python’s `list` behavior.

```python
class DynamicArray:
    def __init__(self):
        self.array = []
        self.capacity = 1  # Initial capacity
        self.size = 0

    def append(self, value):
        if self.size == self.capacity:
            self.resize(2 * self.capacity)
        self.array.append(value)
        self.size += 1

    def resize(self, new_capacity):
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of bounds")
        return self.array[index]

    def __repr__(self):
        return str(self.array[:self.size])

# Example usage
dynamic_array = DynamicArray()
dynamic_array.append(1)
dynamic_array.append(2)
print(dynamic_array)  # Output: [1, 2]
```

---

### **When to Use?**
- **Static Array**: Best when the size is known beforehand and memory efficiency is critical.
- **Dynamic Array**: Preferred for situations where the size may change frequently. Python’s built-in `list` is typically sufficient for most use cases.
