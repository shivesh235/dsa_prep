**Breadth-First Search (BFS)**

**Explanation**:
Breadth-First Search is a graph traversal algorithm that explores all neighbors of a node before moving to the next level of nodes. It uses a queue data structure to keep track of nodes to visit in the order they were discovered.

**Key Characteristics**:
1. BFS starts at the root (or any arbitrary node in case of an unconnected graph).
2. It explores all immediate neighbors of the starting node first.
3. Then it proceeds to the neighbors of those neighbors, and so on.
4. BFS ensures that all nodes at a given depth are visited before moving deeper.

**Applications**:
- Finding the shortest path in an unweighted graph
- Solving problems like the shortest path in a maze
- Detecting connected components in a graph

---

### **Pseudocode for BFS**

```plaintext
BFS(start_node):
    Create an empty queue and enqueue start_node
    Create a set to track visited nodes
    While the queue is not empty:
        Dequeue a node from the queue
        If the node is not visited:
            Mark it as visited
            Process the node (e.g., print it)
            Enqueue all unvisited neighbors of the node
```

---

### **Python Implementation**

#### BFS Algorithm:
```python
from collections import deque

def bfs(graph, start_node):
    visited = set()  # To track visited nodes
    queue = deque([start_node])  # Initialize queue with the starting node
    
    while queue:
        node = queue.popleft()  # Dequeue the first element
        if node not in visited:
            print(node, end=" ")  # Process the node (e.g., print it)
            visited.add(node)
            # Enqueue all unvisited neighbors
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
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

print("BFS Traversal:")
bfs(graph, 'A')
```

---

### **Output**:
```
BFS Traversal:
A B C D E F
```

---

### **Comparison with DFS**

| Feature            | DFS                                 | BFS                                |
|--------------------|-------------------------------------|------------------------------------|
| Data Structure     | Stack (explicit or implicit)       | Queue                             |
| Traversal Order    | Depth-first, explores deep first   | Breadth-first, explores wide first|
| Use Case           | Solving puzzles, detecting cycles  | Shortest path in unweighted graph |
| Complexity         | O(V + E)                          | O(V + E)                          |

