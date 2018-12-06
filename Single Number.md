## Single Number<br>

### Single Number I
**Resource:**
[leetcode136](https://leetcode.com/problems/single-number/description/)
**Description:**
Given a non-empty array of integers, every element appears twice except for one. Find that single one.
**Solution:**
Implement without extra memory: XOR each number in array.
**Code:**
```
class Solution {
  public int singleNumber(int[] nums) {
    int res = 0;
    for (int num : nums) {
      res ^= num;
    }
    return res;
  }
}
```

### Single Number II
**Resource:**
[leetcode137](https://leetcode.com/problems/single-number-ii/description/)
**Description:**
Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.
**Solution:**
1. We count the sum of all number in array for each bit. The single number is consist of the bits that model 3 is 1 rather than 0. Time Complexity: O(32n);
This idea can be generalize by model k.
2. Generalization of the bitwise operation, see [here](https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers). Time Complexity: O(n);

**Code:**
```
class Solution {
  public int singleNumber(int[] nums) {
    int res = 0;
    for (int i = 0; i < 32; i++) {
      int sum = 0;
      for (int num : nums) {
        if (((num >> i) & 1) == 1) sum += 1;
      }
      sum %= 3;
      res |= sum << i;
    }
    return res;
  }
```

### Single Number III
**Resource:**
[leetcode260](https://leetcode.com/problems/single-number-iii/description/)
**Description:**
Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.
**Solution:**
First, we XOR all numbers in array. The XOR result is the XOR of two distinct unique number, and every "1" bit represents that the two distinct number are different at that bit position. Now we choose any different bit position and divide all numbers into two group:
- Group one: all numbers with "0" at that bit position.
- Group two: all numbers with "1" at that bit position.

Two distinct unique number must locate in different group. All we need to do is XOR all numbers in each group.
One more thing, how we choose that different bit position? We use "diff &= -diff", which is the right most diff position. Genius idea for bitwise op!

**Code:**
```
class Solution {
  public int[] singleNumber(int[] nums) {
    int diff = 0;
    for (int num : nums) {
      diff ^= num;
    }
    diff &= -diff;
    int[] res = {0, 0};
    for (int num : nums) {
      if ((diff & num) == 0) res[0] ^= num;
      else res[1] ^= num;
    }
    return res;
  }
}
```
