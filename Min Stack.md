## Min Stack<br>

**Resource:** [leetcode155](https://leetcode.com/problems/min-stack/description/)
**Description:**
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
- push(x) -- Push element x onto stack.
- pop() -- Removes the element on top of the stack.
- top() -- Get the top element.
- getMin() -- Retrieve the minimum element in the stack.

**Example:**
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
**Solution:**
1. Use a native stack to maintain the stack and a priority queue to maintain the minimum value.
2. Create a node data structure, each node contains its value, next node and minimum value of all nodes pushed  before itself. We have a head node to maintain the FIFO order.

**Code:**

```
class MinStack {

    Node head;

    public MinStack() {
        head = new Node();
    }

    public void push(int x) {
        if (head.next == null) head.next = new Node(x, x, null);
        else head.next = new Node(x, Math.min(x, head.next.min), head.next);
    }

    public void pop() {
        head.next = head.next.next;
    }

    public int top() {
        return head.next.val;
    }

    public int getMin() {
        return head.next.min;
    }

    class Node {

        int val;
        int min;
        Node next;

        Node() {}

        Node(int val, int min, Node next) {
            this.val = val;
            this.min = min;
            this.next = next;
        }
    }
}
```
