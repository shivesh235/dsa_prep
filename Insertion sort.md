### Insertion Sort
**Input:**  
An array \( A \) of size \( n \) containing \( n \) elements \( A[0], A[1], \dots, A[n-1] \).

**Output:**  
The array \( A \), sorted in ascending order.

**Goal:**  
Sort the array \( A \) in ascending order using the **Insertion Sort** algorithm, which builds the sorted portion of the array one element at a time by inserting each element into its correct position.

---

### Pseudocode

```text
InsertionSort(A, n):
    Input: Array A of size n
    Output: Array A sorted in ascending order

    for i ← 1 to n-1 do:
        key ← A[i]  // Element to be inserted into the sorted portion
        j ← i - 1   // Index of the last element in the sorted portion

        // Shift elements of the sorted portion to the right to make space for key
        while j ≥ 0 and A[j] > key do:
            A[j+1] ← A[j]
            j ← j - 1

        // Insert the key into its correct position
        A[j+1] ← key

    return A
```

---

### Explanation of the Algorithm

1. **Outer Loop (i):**
   - The outer loop iterates from the second element (\( i = 1 \)) to the last element (\( i = n-1 \)).
   - At each iteration, the element \( A[i] \) (called the "key") is the element to be inserted into the correct position in the sorted portion of the array (\( A[0 \dots i-1] \)).

2. **Shifting Elements (Inner Loop):**
   - The inner loop iterates through the sorted portion of the array (\( A[0 \dots i-1] \)) and shifts elements to the right to make space for the key.
   - The condition \( A[j] > \text{key} \) ensures that the elements larger than the key are shifted.

3. **Insertion:**
   - Once the correct position for the key is found, it is inserted at \( A[j+1] \).

4. **Output:**
   - The array \( A \) is sorted in ascending order.

---

### Example Walkthrough

#### Input:
- \( A = [12, 11, 13, 5, 6] \), \( n = 5 \).

#### Execution:

**Step 1 (i = 1):**
- Key: \( 11 \), Sorted Portion: \( [12] \).
- Compare \( 11 \) with \( 12 \): Shift \( 12 \) to the right.
- Insert \( 11 \) at position \( 0 \).
- Array after insertion: \( [11, 12, 13, 5, 6] \).

**Step 2 (i = 2):**
- Key: \( 13 \), Sorted Portion: \( [11, 12] \).
- Compare \( 13 \) with \( 12 \): No shift needed.
- Insert \( 13 \) at position \( 2 \).
- Array remains: \( [11, 12, 13, 5, 6] \).

**Step 3 (i = 3):**
- Key: \( 5 \), Sorted Portion: \( [11, 12, 13] \).
- Compare \( 5 \) with \( 13 \): Shift \( 13 \) to the right.
- Compare \( 5 \) with \( 12 \): Shift \( 12 \) to the right.
- Compare \( 5 \) with \( 11 \): Shift \( 11 \) to the right.
- Insert \( 5 \) at position \( 0 \).
- Array after insertion: \( [5, 11, 12, 13, 6] \).

**Step 4 (i = 4):**
- Key: \( 6 \), Sorted Portion: \( [5, 11, 12, 13] \).
- Compare \( 6 \) with \( 13 \): Shift \( 13 \) to the right.
- Compare \( 6 \) with \( 12 \): Shift \( 12 \) to the right.
- Compare \( 6 \) with \( 11 \): Shift \( 11 \) to the right.
- Insert \( 6 \) at position \( 1 \).
- Array after insertion: \( [5, 6, 11, 12, 13] \).

#### Output:
- Sorted array: \( [5, 6, 11, 12, 13] \).

---

### Complexity

- **Time Complexity:**  
  - **Best Case:** \( O(n) \) (when the array is already sorted).  
  - **Worst Case:** \( O(n^2) \) (when the array is sorted in reverse order).  
  - **Average Case:** \( O(n^2) \).

- **Space Complexity:**  
  - \( O(1) \) (in-place sorting).
