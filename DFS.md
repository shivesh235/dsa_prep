### **Depth-First Search (DFS)**

**Explanation**:
Depth-First Search is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It uses a stack data structure (either explicitly or through recursion) to keep track of the nodes to visit next.

**Key Characteristics**:
1. DFS starts at the root (or any arbitrary node in case of an unconnected graph).
2. It explores a node, then goes deeper into one of its unvisited neighbors.
3. If it reaches a node with no unvisited neighbors, it backtracks to the previous node.
4. This process continues until all nodes have been visited.

**Applications**:
- Detecting cycles in a graph
- Solving puzzles and mazes
- Topological sorting
- Connected components in a graph

---

### **Pseudocode for DFS**

#### Recursive Approach:
```plaintext
DFS(node, visited):
    Mark node as visited
    For each neighbor of node:
        If neighbor is not visited:
            Call DFS(neighbor, visited)
```

#### Iterative Approach:
```plaintext
DFS(start_node):
    Create an empty stack and push start_node
    While stack is not empty:
        Pop the top node from the stack
        If the node is not visited:
            Mark it as visited
            Push all unvisited neighbors of the node onto the stack
```

---

### **Python Implementation**

#### Recursive DFS:
```python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()  # Initialize visited set
    visited.add(node)
    print(node, end=" ")  # Process the node (e.g., print it)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
```

#### Iterative DFS:
```python
def dfs_iterative(graph, start_node):
    visited = set()  # To track visited nodes
    stack = [start_node]  # Initialize stack with the starting node
    
    while stack:
        node = stack.pop()  # Get the last element from the stack
        if node not in visited:
            print(node, end=" ")  # Process the node (e.g., print it)
            visited.add(node)
            # Add unvisited neighbors to the stack
            stack.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
```

---

### **Example Usage**

```python
# Example graph represented as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C']
}

print("Recursive DFS:")
dfs_recursive(graph, 'A')

print("\nIterative DFS:")
dfs_iterative(graph, 'A')
```

---

**Output**:
```
Recursive DFS:
A B D E C F 

Iterative DFS:
A C F B E D
```

Note: The output order may vary depending on how neighbors are stored in the adjacency list.
