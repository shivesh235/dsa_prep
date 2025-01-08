### Merge Sort

**Input:**  
An array \( A \) of size \( n \) containing \( n \) elements \( A[0], A[1], \dots, A[n-1] \).

**Output:**  
The array \( A \), sorted in ascending order.

**Goal:**  
Sort the array \( A \) in ascending order using the **Merge Sort** algorithm, which is a divide-and-conquer algorithm that recursively divides the array into smaller subarrays, sorts them, and then merges the sorted subarrays.

---

### Pseudocode

```text
MergeSort(A, left, right):
    Input: Array A, start index left, end index right
    Output: Sorted subarray A[left...right]

    if left < right then:
        // Find the middle point
        mid ← (left + right) // 2

        // Recursively sort the left half
        MergeSort(A, left, mid)

        // Recursively sort the right half
        MergeSort(A, mid+1, right)

        // Merge the two sorted halves
        Merge(A, left, mid, right)


Merge(A, left, mid, right):
    Input: Array A, indices left, mid, right
    Output: Merges two sorted subarrays into one sorted subarray

    // Sizes of the two subarrays
    n1 ← mid - left + 1
    n2 ← right - mid

    // Create temporary arrays for the left and right subarrays
    L ← new array of size n1
    R ← new array of size n2

    // Copy data to the temporary arrays
    for i ← 0 to n1-1 do:
        L[i] ← A[left + i]

    for j ← 0 to n2-1 do:
        R[j] ← A[mid + 1 + j]

    // Merge the two subarrays
    i ← 0, j ← 0, k ← left
    while i < n1 and j < n2 do:
        if L[i] ≤ R[j] then:
            A[k] ← L[i]
            i ← i + 1
        else:
            A[k] ← R[j]
            j ← j + 1
        k ← k + 1

    // Copy any remaining elements of L
    while i < n1 do:
        A[k] ← L[i]
        i ← i + 1
        k ← k + 1

    // Copy any remaining elements of R
    while j < n2 do:
        A[k] ← R[j]
        j ← j + 1
        k ← k + 1
```

---

### Explanation of the Algorithm

1. **Divide:**  
   - Recursively divide the array into two halves until each subarray contains only one element (base case).

2. **Conquer:**  
   - Sort each half by recursively applying the merge sort.

3. **Merge:**  
   - Combine two sorted subarrays into a single sorted subarray using the `Merge` function.

4. **Output:**  
   - The array \( A \) is sorted in ascending order.

---

### Example Walkthrough

#### Input:
- \( A = [12, 11, 13, 5, 6, 7] \), \( n = 6 \).

#### Execution:

**Step 1 (Divide):**
1. Divide \( A[0...5] \) into \( A[0...2] \) and \( A[3...5] \).
2. Divide \( A[0...2] \) into \( A[0...1] \) and \( A[2...2] \).
3. Divide \( A[0...1] \) into \( A[0...0] \) and \( A[1...1] \).

**Step 2 (Conquer):**
- \( A[0...0] = [12] \), \( A[1...1] = [11] \), \( A[2...2] = [13] \), \( A[3...3] = [5] \), \( A[4...4] = [6] \), \( A[5...5] = [7] \). These are already sorted.

**Step 3 (Merge):**
1. Merge \( A[0...0] \) and \( A[1...1] \) into \( A[0...1] = [11, 12] \).
2. Merge \( A[0...1] \) and \( A[2...2] \) into \( A[0...2] = [11, 12, 13] \).
3. Merge \( A[3...3] \) and \( A[4...4] \) into \( A[3...4] = [5, 6] \).
4. Merge \( A[3...4] \) and \( A[5...5] \) into \( A[3...5] = [5, 6, 7] \).
5. Merge \( A[0...2] \) and \( A[3...5] \) into \( A[0...5] = [5, 6, 7, 11, 12, 13] \).

#### Output:
- Sorted array: \( [5, 6, 7, 11, 12, 13] \).

---

### Complexity

- **Time Complexity:**  
  - Divide: \( O(\log n) \) (since the array is recursively divided in half).  
  - Merge: \( O(n) \) (merging two subarrays takes linear time).  
  - Total: \( O(n \log n) \).

- **Space Complexity:**  
  - \( O(n) \) (additional space for temporary arrays during merging).
