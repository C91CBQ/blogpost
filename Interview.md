# Google Interview Questions

## 1. Multiply Strings
**Resource:** [leetcode 43](https://leetcode.com/problems/multiply-strings/description/)
**Description:** Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2.
**Example**:
**Solution:** Start from right to left, perform multiplication on every pair of digits, and add them together
time: O(n1 + n2);
space: O(n1 + n2);
**Code:**
```
```

## 2. Generate Parenthesis
**resource:** [leetcode 22](https://leetcode.com/problems/generate-parentheses/description/)
**description:** Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
**solution:** backtracking
time: O(2^n);
space: O(2^n); extra space O(n);

## 3. Permutation
**resource:** [leetcode 46](https://leetcode.com/problems/permutations/description/)
**description:** Given a collection of distinct numbers, return all possible permutations.
**solution:** backtracking
time: O(n!);
space: O(n!); extra space O(n);

## 4. Gray Code
**resource:** [leetcode 89](https://leetcode.com/problems/gray-code/description/)
**description:** The gray code is a binary numeral system where two successive values differ in only one bit. Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.
**solution:** The middle two numbers only differ at their highest bit, while the rest numbers of part two are exactly symmetric of part one.
time: O(2^n);
space: O(2^n); extra space: O(1);

## 5. Search a 2D Matrix
**resource:** [leetcode 74](https://leetcode.com/problems/search-a-2d-matrix/description/), [leetcode 240](https://leetcode.com/problems/search-a-2d-matrix-ii/description/)
**description:** Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
* Integers in each row are sorted in ascending from left to right.
* Integers in each column are sorted in ascending from top to bottom.

**solution:** Select left-bottom(or right-top) element, if the target element is larger than the selected, move RIGHT; otherwise move TOP. If the index out of the boundary, return false;
time: O(m + n);
space: O(1);

## 6. Search for a Range
**resource:** [leetcode 34](https://leetcode.com/problems/search-for-a-range/description/)
**description:** Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.
**solution:**
1. the general binary search could find the left-most(start) point of matched target, we can find the start of matched (target + 1) as the end point of the matched target.
2. modify the general binary search and make it right-prone to find the end point.

time: O(logn);
space: O(1);
**code:**
```
Class BinarySearch {

    //binary search, left-prone
    public boolean BS1(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        if (nums[left] != target) return false;
        else return true;
    }

    //binary search, right-prone
    public boolean BS1(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + ((right - left) >> 1) + 1;// make it right-prone
            if (nums[mid] <= target) left = mid;
            else right = mid - 1;
        }
        if (nums[left] != target) return false;
        else return true;
    }

}

```

## 7. Paint Fence
**Resource:** [leetcode 276](https://leetcode.com/problems/paint-fence/description/)
**Description:** There is a fence with n posts, each post can be painted with one of the k colors. You have to paint all the posts such that no more than two adjacent fence posts have the same color. Return the total number of ways you can paint the fence.
**Example:** n = 3, k = 2, return 6.
**Solution:** Dynamic Programming. The color of i-th post have to option:
1. same with its previous post when its previous two posts do not have same color;
2. different with its previous post no matter whether its previous two posts' color is same.

time: O(n);
space: O(1);
**Code:**
```
class Solution {
    public int numWays(int n, int k) {
        if (n == 0 || k == 0) return 0;
        if (n == 1) return k;
        int same = k;
        int diff = k * (k - 1);
        for (int i = 2; i < n; i++) {
            int temp = diff;
            diff = (k - 1) * (same + diff);
            same = temp;
        }
        return diff + same;
    }
}
```
## 8. Paint House
**Resource:** [leetcode 265](https://leetcode.com/problems/paint-house-ii/description/)
**Description:** There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color. The cost of painting each house with a certain color is represented by a n x k cost matrix.
**Example:**
 [R, G, B]
[[4, 2, 4],
 [1, 5, 2],
 [3, 4, 1]]
return 4 (min cost: G, R, B);
**Solution:** Dynamic Programming. We try every possible color for each house. The total number of cost is the smallest cost so-far plus i-th chosen color cost. Because no adjacent house with same color is allowed, we need to maintain two minimum cost so far.
color[i] = minimum cost so far + cost[j][i];
time: O(nk);
space: O(k);
**Code:**
```
class Solution {
    public int minCostII(int[][] costs) {
        int n = costs.length;
        if (n == 0) return 0;
        int[] colors = new int[costs[0].length];
        int min1 = 0, min2 = 0;
        for (int i = 0; i < n; i++) {
            int next_min1 = Integer.MAX_VALUE, next_min2 = Integer.MAX_VALUE;
            for (int j = 0; j < colors.length; j++) {
                if (colors[j] == min1) colors[j] = costs[i][j] + min2;
                else colors[j] = costs[i][j] + min1;
                if (colors[j] < next_min1) {
                    next_min2 = next_min1;
                    next_min1 = colors[j];
                }
                else if (colors[j] >= next_min1 && colors[j] < next_min2) next_min2 = colors[j];
            }
            min1 = next_min1;
            min2 = next_min2;
        }
        return min1;
    }
}
```
## 9. Compare Version Number
**Resource:** [leetcode 165](https://leetcode.com/problems/compare-version-numbers/description/)
**Description:** Compare two version numbers version1 and version2.
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
**Example:** 0.1 < 1.1.4 = 1.1.4.0 < 1.2 < 13.37
**Solution:** String operations. Be careful to the case with .0;
time: O(n1 + n2);
space: O(n1 + n2);
**Code:**
```
class Solution {
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null) return 0;
        String[] split1 = version1.split("[.]");
        String[] split2 = version2.split("[.]");
        for (int i = 0; i < Math.max(split1.length, split2.length); i++) {
            int num1 = i < split1.length ? Integer.valueOf(split1[i]) : 0;
            int num2 = i < split2.length ? Integer.valueOf(split2[i]) : 0;
            if (num1 < num2) return -1;
            else if (num1 > num2) return 1;
        }
        return 0;
    }
}
```
## 10. Binary Tree Path
**Resource:** [leetcode 257](https://leetcode.com/problems/binary-tree-paths/description/)
**Description:** Given a binary tree, return all root-to-leaf paths.
**Example:**   
   1
 /   \
2     3
 \
  5
return ["1->2->5", "1->3"];
**Solution:** recursion, dfs iteration with stack, bfs iteration with queue;
time: O(n);
space: O(n);// O(h) for dfs and recursion, O(n) for bfs
```
class Solution {

    //dfs iteration with stack
    public List<String> binaryTreePaths1(TreeNode root) {
        List<String> res = new LinkedList<>();
        if (root == null) return res;
        Stack<TreeNode> nodeStack = new Stack<>();
        Stack<String> strStack = new Stack<>();
        nodeStack.push(root);
        strStack.push("");
        while (!nodeStack.isEmpty()) {
            TreeNode node = nodeStack.pop();
            String cur = strStack.pop();
            if (node.left == null && node.right == null) res.add(cur + node.val);
            if (node.right != null) {
                nodeStack.push(node.right);
                strStack.push(cur + node.val + "->");
            }
            if (node.left != null) {
                nodeStack.push(node.left);
                strStack.push(cur + node.val + "->");
            }
        }
        return res;
    }

    //bfs iteration with queue
    public List<String> binaryTreePaths2(TreeNode root) {
        List<String> res = new LinkedList<>();
        if (root == null) return res;
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<String> strQueue = new LinkedList<>();
        nodeQueue.offer(root);
        strQueue.offer("");
        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.poll();
            String cur = strQueue.poll();
            if (node.left == null && node.right == null) res.add(cur + node.val);
            if (node.left != null) {
                nodeQueue.offer(node.left);
                strQueue.offer(cur + node.val + "->");
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
                strQueue.offer(cur + node.val + "->");
            }
        }
        return res;
    }

    //recursion
    public List<String> binaryTreePaths3(TreeNode root) {
        List<String> res = new LinkedList<>();
        if (root == null) return res;
        dfs(root, "", res);
        return res;
    }
    private void dfs(TreeNode node, String cur, List<String> res) {
        if (node.left == null && node.right == null) {
            res.add(cur + node.val);
            return;
        }
        String temp = cur + node.val + "->";
        if (node.left != null) dfs(node.left, temp, res);
        if (node.right != null) dfs(node.right, temp, res);
    }

}
```
## 11. Binary Search Tree Iterator
**Resource:** [leetcode 173](https://leetcode.com/problems/binary-search-tree-iterator/description/)
**Description:** mplement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST. Calling next() will return the next smallest number in the BST.
**Solution:**
1. Inorder iteration with stack.
time: O(1) for next() and hasNext();
space: O(h);
```
public class BSTIterator {

    Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /** @return the next smallest number */
    public int next() throws RuntimeException {
        TreeNode temp = stack.pop();
        pushAll(temp.right);
        return temp.val;
    }

    private void pushAll(TreeNode node) {
        while (node != null) {
            stack.push(node);
            node = node.left;
        }
    }
}
```
2. Using Morris Traversal. This could change the original tree.
    1. Initialize current as root
    2. While current is not NULL
        - If current does not have left child
    a) Preserve it as previous node
    b) Go to the right, i.e., current = current->right
        - Else
    a) Make current as right child of the rightmost node in current's left subtree
    b) Let previous node point to the left tree.
    c) Decorrelate current and current->left
    d) Go to this left child, i.e., current = current->left
    3. time: O(1) for next() and hasNext(), O(n) for constructor;
    space: O(1);

```
public class BSTIterator {

    private TreeNode cur;

    public BSTIterator(TreeNode root) {
        if (root == null) return;
        cur = root;
        while (cur.left != null) cur = cur.left;
        helper(root);
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return cur != null;
    }

    /** @return the next smallest number */
    public int next() {
        int res = cur.val;
        cur = cur.right;
        return res;
    }

    private void helper(TreeNode node) {
        TreeNode pre = null;
        while (node != null) {
            if (node.left != null) {
                TreeNode temp = node.left;
                while (temp.right != null) temp = temp.right;
                temp.right = node;
                if (pre != null) {
                    pre.right = node.left;
                }
                temp = node.left;
                node.left = null;
                node = temp;
            } else {
                pre = node;
                node = node.right;
            }
        }
    }
}
```
## 12. Binary Tree Longest Consecutive Sequence
**Resource:** [leetcode 298](https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/description/), [leetcode 549](https://leetcode.com/problems/binary-tree-longest-consecutive-sequence-ii/description/)
**Description:** Given a binary tree, you need to find the length of Longest Consecutive Path in Binary Tree. Especially, this path can be either increasing or decreasing. For example, [1,2,3,4] and [4,3,2,1] are both considered valid, but the path [1,2,4,3] is not valid. On the other hand, the path can be in the child-Parent-child order, where not necessarily be parent-child order.
**Example:**
      2
     / \
    1   3
The longest consecutive path is [1, 2, 3] or [3, 2, 1].
**Solution:** dfs recursion. Use an instance variable to find the max child-parent-child path of each node. dfs function return an integer array store the max ascending parent-child path and descending parent-child path.
time: O(n);
space: O(n);
**Code:**
```
class Solution {
    int max = 0;
    public int longestConsecutive(TreeNode root) {
        dfs(root);
        return max;
    }

    private int[] dfs(TreeNode node) {
        int[] res = new int[] {1, 1};
        if (node == null) return res;
        int[] left = dfs(node.left);
        int[] right = dfs(node.right);
        if (node.left != null) {
            if (node.left.val == node.val + 1) res[0] = Math.max(res[0], left[0] + 1);
            if (node.left.val == node.val - 1) res[1] = Math.max(res[1], left[1] + 1);
        }
        if (node.right != null) {
            if (node.right.val == node.val + 1) res[0] = Math.max(res[0], right[0] + 1);
            if (node.right.val == node.val - 1) res[1] = Math.max(res[1], right[1] + 1);
        }
        max = Math.max(res[0] + res[1] - 1, max);
        return res;
    }
}
```
## 13. Validate Binary Search Tree
**Resource:** [leetcode 98](https://leetcode.com/problems/validate-binary-search-tree/description/)
**Description:** Given a binary tree, determine if it is a valid binary search tree (BST).
**Solution:** recursion or iteration
time: O(n);
space: O(1) for recursion, O(n) for iteration, if balanced BST O(logn);
**Code:**
```
class Solution {

    //recursion
    public boolean isValidBST1(TreeNode root) {
        return isBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    private boolean isBST(TreeNode node, long min, long max) {
        if (node == null) return true;
        if (node.val >= max || node.val <= min) return false;
        return isBST(node.left, min, node.val) && isBST(node.right, node.val, max);
    }

    //iteration
    public boolean isValidBST2(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (pre != null && pre.val >= root.val) return false;
            pre = root;
            root = root.right;
        }
        return true;
    }
}
```

## 14. K-th Smallest Element in a BST
**Resource:** [leetcode 230] (https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)
**Description:** Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
**Solution:**
1. Binary Search: Count number of node for each subtree, find the node with k - 1 sub-nodes.
time: O(h);
space: O(1);
```
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        int left = count(root.left);
        if (left == k - 1) return root.val;
        else if (left > k - 1) return kthSmallest(root.left, k);
        else return kthSmallest(root.right, k - left - 1);
    }
    private int count(TreeNode node) {
        if (node == null) return 0;
        return count(node.left) + 1 + count(node.right);
    }
}
```
2. Inorder Recursion.
time: O(k);
space: O(1);
```
class Solution {

    private int count;
    private int res;

    public int kthSmallest(TreeNode root, int k) {
        count = k;
        helper(root);
        return res;
    }
    private void helper(TreeNode node) {
        if (node.left != null) helper(node.left);
        count--;
        if (count == 0) {
            res = node.val;
            return;
        } else if (count < 0) return;
        if (node.right != null) helper(node.right);
    }
}
```
3. Inorder Iteration.
time: O(k);
space: O(n) on worst case, O(logn) on average;
```
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (--k == 0) return root.val;
            root = root.right;
        }
        return -1;
    }
}
```

## 15. The Maze
**Resource:** [leetcode 490](https://leetcode.com/problems/the-maze/description/)
**Description:** There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction. Given the ball's start position, the destination and the maze, determine whether the ball could stop at the destination.
**Example:**
Input 1: a maze represented by a 2D array
0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0
Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)
Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
**Solution:** dfs
time: O(mn);
space: O(mn);
**Code:**
```
class Solution {
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        return dfs(maze, start, destination, visited);
    }
    private boolean dfs(int[][] maze, int[] start, int[] des, boolean[][] visited) {
        if (visited[start[0]][start[1]]) return false;
        if (start[0] == des[0] && start[1] == des[1]) return true;
        visited[start[0]][start[1]] = true;
        int l = start[1] - 1, r = start[1] + 1, u = start[0] - 1, d = start[0] + 1;
        while (l >= 0 && maze[start[0]][l] == 0) l--;
        if (dfs(maze, new int[]{start[0], l + 1}, des, visited)) return true;
        while (r < maze[0].length && maze[start[0]][r] == 0) r++;
        if (dfs(maze, new int[]{start[0], r - 1}, des, visited)) return true;
        while (u >= 0 && maze[u][start[1]] == 0) u--;
        if (dfs(maze, new int[]{u + 1, start[1]}, des, visited)) return true;
        while (d < maze.length && maze[d][start[1]] == 0) d++;
        if (dfs(maze, new int[]{d - 1, start[1]}, des, visited)) return true;
        return false;
    }
}
```
## 16. The Maze II
**Resource:**
**Description:** There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.
Given the ball's start position, the destination and the maze, find the shortest distance for the ball to stop at the destination. The distance is defined by the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included). If the ball cannot stop at the destination, return -1.
**Example:**
Input 1: a maze represented by a 2D array
0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0
Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)
Output: 12
Explanation: One shortest way is : left -> down -> left -> down -> right -> down -> right. The total distance is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.
**Solution:** dfs, use a 2-D array dis to store the minimum path to source.
time: O(mn * max(m,n));
space: O(mn);
**Code:**
```
public class Solution {
    public int shortestDistance(int[][] maze, int[] start, int[] dest) {
        int[][] distance = new int[maze.length][maze[0].length];
        for (int[] row: distance)
            Arrays.fill(row, Integer.MAX_VALUE);
        distance[start[0]][start[1]] = 0;
        dfs(maze, start, distance);
        return distance[dest[0]][dest[1]] == Integer.MAX_VALUE ? -1 : distance[dest[0]][dest[1]];
    }

    public void dfs(int[][] maze, int[] start, int[][] distance) {
        int[][] dirs = {{0,1}, {0,-1}, {-1,0}, {1,0}};
        for (int[] dir: dirs) {
            int x = start[0] + dir[0];
            int y = start[1] + dir[1];
            int count = 0;
            while (x >= 0 && y >= 0 && x < maze.length && y < maze[0].length && maze[x][y] == 0) {
                x += dir[0];
                y += dir[1];
                count++;
            }
            if (distance[start[0]][start[1]] + count < distance[x - dir[0]][y - dir[1]]) {
                distance[x - dir[0]][y - dir[1]] = distance[start[0]][start[1]] + count;
                dfs(maze, new int[]{x - dir[0],y - dir[1]}, distance);
            }
        }
    }
}
```
## 17. Jump Game
**Resource:** [leetcode 55](https://leetcode.com/problems/jump-game/description/)
**Description:** Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Determine if you are able to reach the last index.
**Example:**
[2,3,1,1,4], return true.
[3,2,1,0,4], return false.
**Solution:**
1. Dynamic Programming. Maintain an boolean array to find whether we can get the last element.
time: O(n^2);
space: O(n);
```
class Solution {
    public boolean canJump(int[] nums) {
        boolean[] isReachable = new boolean[nums.length];
        isReachable[0] = true;
        for (int i = 0; i < nums.length; i++) {
            if (!isReachable[i]) return false;
            int step = 1;
            while (step <= nums[i]) {
                if (i + step < nums.length - 1) {
                    isReachable[i + step] = true;
                    step++;
                } else {
                    return true;
                }
            }
        }
        return isReachable[nums.length - 1];
    }
}
```
2. Greedy Method. From iteration from the last element then find out whether it could be reached.
time: O(n);
space: O(1);
```
class Solution {
    public boolean canJump(int[] nums) {
        int lastIndex = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i + nums[i] >= lastIndex) lastIndex = i;
        }
        return lastIndex == 0;
    }
}
```

## 18. Jump Game II
**Resource:** [leetcode 45](https://leetcode.com/problems/jump-game-ii/description/)
**Description:** Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Your goal is to reach the last index in the minimum number of jumps.
**Example:** [2,3,1,1,4], return 2.
**Solution:**
1. Dynamic Programming. Maintain an minimum step array for each move.
time: O(n^2);
space: O(n);
```
class Solution {
    public int jump(int[] nums) {
        int[] min = new int[nums.length];
        Arrays.fill(min, nums.length);
        min[0] = 0;
        for (int i = 0; i < nums.length; i++) {
            int step = Math.min(nums.length - 1 - i, nums[i]);
            while (step > 0) {
                min[i + step] = Math.min(min[i + step], min[i] + 1);
                step--;
            }
        }
        return min[nums.length - 1];
    }
}
```
2. Greedy Method. Let's say the range of the current jump is [curBegin, curEnd], curFarthest is the farthest point that all points in [curBegin, curEnd] can reach. Once the current point reaches curEnd, then trigger another jump, and set the new curEnd with curFarthest, then keep the above steps, as the following:
time: O(n);
space: O(1);
```
class Solution {
    public int jump(int[] nums) {
        int min = 0, curFarthest = 0, curEnd = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            curFarthest = Math.max(i + nums[i], curFarthest);
            if (i == curEnd) {
                min++;
                curEnd = curFarthest;
                if (curFarthest >= nums.length - 1) break;
            }
        }
        return min;
    }
}
```

## 19. Longest Substring with At Most K Distinct Characters
**Resource:** [leetcode 340](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/)
**Description:** Given a string, find the length of the longest substring T that contains at most k distinct characters.
**Example:** Given s = “eceba” and k = 2, T is "ece" which its length is 3.
**Solution:**
1. Sliding Window with array.
time: O(n);
space: O(1);
```
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int[] chars = new int[256];
        int left = 0, right = 0, count = 0, max = 0;
        while (left < s.length() - max && right < s.length()) {
            if (chars[s.charAt(right)]++ == 0) count++;
            if (count <= k) max = Math.max(max, right - left + 1);
            else if (count > k) {
                while (--chars[s.charAt(left++)] > 0);
                count--;
            }
            right++;
        }
        return max;
    }
}
```
2. Sliding Window with LinkedHashMap
time: O(n);
space: O(k);
```
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        LinkedHashMap<Character, Integer> map = new LinkedHashMap<>();
        int left = 0, right = 0, count = 0, max = 0;
        while (left < s.length() - max && right < s.length()) {
            char c = s.charAt(right);
            if (!map.containsKey(c)) count++;
            map.remove(c);
            map.put(c, right);
            if (count <= k) max = Math.max(max, right - left + 1);
            else {
                left = map.remove(map.keySet().iterator().next()) + 1;
                System.out.println(left);
                count--;
            }
            right++;
        }
        return max;
    }
}
```

## 20. Random Pick Index
**Resource:** [leetcode 398](https://leetcode.com/problems/random-pick-index/description/)
**Description:** Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.
**Example:**
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);
// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);
// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
**Solution:** Reservoir Sampling. For the nth target, ++count is n. Then the probability that rnd.nextInt(++count)==0 is 1/n. Thus, the probability that return nth target is 1/n. For (n-1)th target, the probability of returning it is (n-1)/n * 1/(n-1)= 1/n.
time: O(n);
space: O(1);
**Code:**
```
public class Solution {

    int[] nums;
    Random rnd;

    public Solution(int[] nums) {
        this.nums = nums;
        this.rnd = new Random();
    }

    public int pick(int target) {
        int result = -1;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != target)
                continue;
            if (rnd.nextInt(++count) == 0)
                result = i;
        }
        return result;
    }
}
```
