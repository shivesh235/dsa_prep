## Kadane’s Algorithm: Maximum Subarray Sum

**Kadane's Algorithm** is an efficient algorithm used to find the maximum sum of a contiguous subarray in a one-dimensional numeric array. This problem is also known as the "Maximum Subarray Problem."

The key idea of the algorithm is to iterate through the array while maintaining a running sum of the current subarray and updating the maximum sum found so far.

### Problem Statement:
Given an array of integers (which can include both positive and negative numbers), find the contiguous subarray (containing at least one number) that has the largest sum, and return that sum.

### Kadane’s Algorithm Explanation:

1. **Initialize two variables**:
   - `max_current`: This will store the maximum sum of the subarray that ends at the current index.
   - `max_global`: This will store the maximum sum found so far.

2. **Start iterating** through the array from the first element:
   - For each element, calculate the maximum sum of the subarray ending at that element. The current subarray can either be:
     - The current element itself (if starting a new subarray at this element is better).
     - The sum of the current element and the previous subarray (if continuing the previous subarray gives a better result).

   Thus, at each step:
   \[
   \text{max\_current} = \max(\text{arr}[i], \text{max\_current} + \text{arr}[i])
   \]
   
3. **Update the global maximum** if the current subarray sum is greater than the global maximum sum found so far:
   \[
   \text{max\_global} = \max(\text{max\_global}, \text{max\_current})
   \]

4. After completing the iteration, the `max_global` will contain the maximum sum of a contiguous subarray.

### Algorithm Steps:

1. Initialize:
   - `max_current = arr[0]`
   - `max_global = arr[0]`
   
2. Traverse through the array starting from index 1:
   - For each element, update `max_current` as the maximum between the current element and `max_current + arr[i]`.
   - Update `max_global` if `max_current` is greater than `max_global`.

3. Return `max_global` at the end, which is the maximum subarray sum.

### Time Complexity:
- **O(n)**, where `n` is the number of elements in the array (since we only traverse the array once).

### Space Complexity:
- **O(1)**, as we only use a constant amount of extra space.

### Python Code Example:

```python
def kadane(arr):
    # Initialize current and global max
    max_current = max_global = arr[0]
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # Update max_current to be the max of the current element itself
        # or the sum of max_current and the current element
        max_current = max(arr[i], max_current + arr[i])
        
        # Update max_global if max_current is greater than max_global
        if max_current > max_global:
            max_global = max_current
    
    return max_global

# Example usage:
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(kadane(arr))  # Output: 6
```

### Example Walkthrough:
For the array `[-2, 1, -3, 4, -1, 2, 1, -5, 4]`, let's walk through Kadane’s algorithm:

1. Initialize `max_current = -2`, `max_global = -2`.
2. Traverse the array:
   - `arr[1] = 1`:  
     \[
     \text{max\_current} = \max(1, -2 + 1) = 1 \quad (\text{update } \text{max\_global} = 1)
     \]
   - `arr[2] = -3`:  
     \[
     \text{max\_current} = \max(-3, 1 - 3) = -2
     \]
   - `arr[3] = 4`:  
     \[
     \text{max\_current} = \max(4, -2 + 4) = 4 \quad (\text{update } \text{max\_global} = 4)
     \]
   - `arr[4] = -1`:  
     \[
     \text{max\_current} = \max(-1, 4 - 1) = 3
     \]
   - `arr[5] = 2`:  
     \[
     \text{max\_current} = \max(2, 3 + 2) = 5 \quad (\text{update } \text{max\_global} = 5)
     \]
   - `arr[6] = 1`:  
     \[
     \text{max\_current} = \max(1, 5 + 1) = 6 \quad (\text{update } \text{max\_global} = 6)
     \]
   - `arr[7] = -5`:  
     \[
     \text{max\_current} = \max(-5, 6 - 5) = 1
     \]
   - `arr[8] = 4`:  
     \[
     \text{max\_current} = \max(4, 1 + 4) = 5
     \]

3. The maximum subarray sum is `max_global = 6`, corresponding to the subarray `[4, -1, 2, 1]`.

### Key Points:
- Kadane’s algorithm efficiently finds the maximum sum of a contiguous subarray in linear time.
- The algorithm works well for arrays containing both positive and negative numbers.
- It can handle cases where the entire array is negative. In that case, it will return the maximum single element.
