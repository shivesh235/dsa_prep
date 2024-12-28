A **linked list** is a linear data structure in which elements, called **nodes**, are stored in sequential order, but they are not stored in contiguous memory locations. Each node contains:

1. **Data**: The actual value stored in the node.
2. **Pointer**: A reference (or link) to the next node in the sequence.

Linked lists come in various forms, including **singly linked lists**, **doubly linked lists**, and **circular linked lists**. Below is an explanation of each, along with their Python implementation.

---

### **1. Singly Linked List**
In a **singly linked list**, each node has a single pointer that points to the next node. The last node's pointer is `None`.

#### Features:
- Traversal is one-way (from head to tail).
- Insertion and deletion are efficient at the beginning or middle, but finding a specific node requires traversal.

#### Implementation:
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")
```

---

### **2. Doubly Linked List**
In a **doubly linked list**, each node contains two pointers:
- One points to the **next node**.
- The other points to the **previous node**.

#### Features:
- Supports two-way traversal (forward and backward).
- Requires more memory than a singly linked list due to the extra pointer.

#### Implementation:
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def display_forward(self):
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")

    def display_backward(self):
        current = self.head
        while current and current.next:
            current = current.next
        while current:
            print(current.data, end=" <-> ")
            current = current.prev
        print("None")
```

---

### **3. Circular Linked List**
In a **circular linked list**, the last node points back to the first node, forming a circular structure. This can be applied to both singly and doubly linked lists.

#### Features:
- No node in the list points to `None`.
- Traversal is continuous, requiring termination conditions.

#### Implementation (Singly Circular Linked List):
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
            return
        current = self.head
        while current.next != self.head:
            current = current.next
        current.next = new_node
        new_node.next = self.head

    def display(self):
        if not self.head:
            return
        current = self.head
        while True:
            print(current.data, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("(head)")
```

#### Implementation (Doubly Circular Linked List):
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class CircularDoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            new_node.prev = new_node
            return
        last = self.head.prev
        last.next = new_node
        new_node.prev = last
        new_node.next = self.head
        self.head.prev = new_node

    def display(self):
        if not self.head:
            return
        current = self.head
        while True:
            print(current.data, end=" <-> ")
            current = current.next
            if current == self.head:
                break
        print("(head)")
```

---

### **Comparison**

| Type                | Memory Usage | Traversal      | Insertion/Deletion |
|---------------------|--------------|----------------|---------------------|
| Singly Linked List  | Low          | One-way        | Easy                |
| Doubly Linked List  | Medium       | Two-way        | Easier (due to `prev` pointer) |
| Circular Linked List| Medium       | Continuous     | Suitable for circular buffers |

These implementations provide the foundation for using linked lists in Python, but Python's built-in `list` is often preferred in practice for its simplicity and versatility.
