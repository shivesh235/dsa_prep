### Selection Sort
**Input:**  
An array \( A \) of size \( n \) containing \( n \) elements \( A[0], A[1], \dots, A[n-1] \).

**Output:**  
The array \( A \), sorted in ascending order.

**Goal:**  
Sort the array \( A \) in ascending order using the **Selection Sort** algorithm, which repeatedly selects the smallest element from the unsorted portion of the array and places it in its correct position.

---

### Pseudocode

```text
SelectionSort(A, n):
    Input: Array A of size n
    Output: Array A sorted in ascending order

    for i ← 0 to n-2 do:
        // Find the index of the minimum element in the unsorted portion
        min_index ← i
        for j ← i+1 to n-1 do:
            if A[j] < A[min_index] then:
                min_index ← j
        
        // Swap the smallest element with the element at index i
        if min_index ≠ i then:
            Swap A[i] and A[min_index]

    return A
```

---

### Explanation of the Algorithm

1. **Outer Loop (i):**
   - The outer loop runs from the first element (\( i = 0 \)) to the second-last element (\( i = n-2 \)).
   - At each iteration, \( i \) represents the index where the next smallest element will be placed.

2. **Finding the Minimum (Inner Loop):**
   - For the current \( i \), the inner loop searches for the smallest element in the unsorted portion of the array (\( A[i], A[i+1], \dots, A[n-1] \)).
   - \( min_index \) keeps track of the index of the smallest element found.

3. **Swapping:**
   - After finding the smallest element in the unsorted portion, it is swapped with the element at index \( i \).

4. **Sorted Portion:**
   - As \( i \) increases, the portion of the array before \( i \) becomes sorted.

5. **Output:**
   - The array \( A \) is sorted in ascending order.

---

### Example Walkthrough

#### Input:
- \( A = [29, 10, 14, 37, 13] \), \( n = 5 \).

#### Execution:

**Step 1 (i = 0):**
- Initial array: \( [29, 10, 14, 37, 13] \).
- Find the smallest element in \( A[0 \dots 4] \): \( 10 \) at index 1.
- Swap \( A[0] \) and \( A[1] \).
- Array after swap: \( [10, 29, 14, 37, 13] \).

**Step 2 (i = 1):**
- Find the smallest element in \( A[1 \dots 4] \): \( 13 \) at index 4.
- Swap \( A[1] \) and \( A[4] \).
- Array after swap: \( [10, 13, 14, 37, 29] \).

**Step 3 (i = 2):**
- Find the smallest element in \( A[2 \dots 4] \): \( 14 \) at index 2.
- No swap needed since \( A[2] \) is already the smallest.
- Array remains: \( [10, 13, 14, 37, 29] \).

**Step 4 (i = 3):**
- Find the smallest element in \( A[3 \dots 4] \): \( 29 \) at index 4.
- Swap \( A[3] \) and \( A[4] \).
- Array after swap: \( [10, 13, 14, 29, 37] \).

#### Output:
- Sorted array: \( [10, 13, 14, 29, 37] \).

---

### Complexity

- **Time Complexity:**  
  - Outer loop runs \( n-1 \) times.  
  - Inner loop runs \( n-1, n-2, \dots, 1 \) times.  
  - Total comparisons: \( \frac{n(n-1)}{2} = O(n^2) \).

- **Space Complexity:**  
  - \( O(1) \) (in-place sorting).
