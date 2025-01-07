## Find all the duplicate elements

Here is an algorithm to find all the duplicate elements in an array without using extra space, assuming the array contains integers where values are within the range `[1, n]` and `n` is the size of the array. This approach modifies the array in-place using index marking.

### Algorithm:
1. **Traverse the array** from the first element to the last.
2. For each element `arr[i]`, **compute the index** based on the value of the current element:  
   \[
   \text{index} = |arr[i]| - 1
   \]
   This computes the correct index (since the values are from 1 to n).
3. **Mark the element** at that index as negative if it is positive, which indicates that this value has been visited.
4. If the element at the computed index is already negative, **this indicates a duplicate**.
5. After traversing the array, **restore the original values** by converting all negative numbers back to positive if needed.

### Python Code Example:
```python
def find_duplicates(arr):
    duplicates = []
    
    # Traverse the array
    for i in range(len(arr)):
        index = abs(arr[i]) - 1  # Find the correct index
        
        # If the value at that index is already negative, it's a duplicate
        if arr[index] < 0:
            duplicates.append(abs(arr[i]))
        else:
            # Mark the element as visited by negating the value at that index
            arr[index] = -arr[index]
    
    # Optional: Restore the original array by making all values positive
    for i in range(len(arr)):
        arr[i] = abs(arr[i])
    
    return duplicates

# Example usage:
arr = [4, 3, 2, 7, 8, 2, 3, 1]
duplicates = find_duplicates(arr)
print(duplicates)  # Output: [2, 3]
```

### Explanation:
- The array is modified in place by negating the elements at specific indices to mark them as visited.
- If an element at the computed index is already negative, it indicates that the element has been encountered before, hence it is a duplicate.
- The algorithm runs in **O(n)** time and does not use any extra space beyond the input array (ignoring the output list).
