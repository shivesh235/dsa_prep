## Product except self

Yes, it’s possible to solve this problem without using two extra spaces (`left` and `right` arrays) and only using the **output array** to store both the left and right product results. This way, you can compute the result in **two passes**, but without explicitly allocating extra arrays for left and right products.

### Optimized Algorithm:
1. **First pass**: Store the left products directly in the output array.
2. **Second pass**: Traverse the array from the right side and update the output array with the right products on the fly, using a temporary variable to keep track of the right product.

### Steps:
1. Initialize the result array with 1's to hold the product of all elements except the current one.
2. Perform the first pass to calculate and store the product of elements to the **left** of each index in the result array.
3. In the second pass, traverse from the right and update the result array with the product of elements to the **right** of each index while maintaining a running right product.

### Time Complexity:
- **O(n)** time complexity, as you traverse the array twice.

### Space Complexity:
- **O(1)** additional space complexity (besides the result array).

### Python Code Implementation:

```python
def product_except_self(arr):
    n = len(arr)
    
    # Initialize the result array with 1s
    result = [1] * n
    
    # First pass: Calculate the left product for each index
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= arr[i]  # Update left_product for the next element
    
    # Second pass: Calculate the right product and update the result array
    right_product = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_product  # Multiply with the current right product
        right_product *= arr[i]  # Update right_product for the next element
    
    return result

# Example usage:
arr = [1, 2, 3, 4]
output = product_except_self(arr)
print(output)  # Output: [24, 12, 8, 6]
```

### Explanation:
1. **First pass (left products)**:
   - We compute the cumulative product of elements to the left of each index and store it directly in the `result` array.
   - After the first pass for `[1, 2, 3, 4]`, `result` becomes `[1, 1, 2, 6]`.

2. **Second pass (right products)**:
   - We traverse from the rightmost side, maintaining a `right_product` that tracks the cumulative product of elements to the right of the current index.
   - We update each element in the `result` array by multiplying the stored left product with the `right_product`.
   - After this pass, the `result` becomes `[24, 12, 8, 6]`.

### Key Points:
- This approach eliminates the need for extra space beyond the result array and works with a constant extra space (ignoring the result).
- The algorithm runs in **O(n)** time with **O(1)** additional space.
