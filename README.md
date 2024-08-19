# Test environment for Claude Caching

```
Cost = c_write * x + c_read * y
c_input * y = c_write * x + c_read * y
x/y = (c_input - c_read) / c_write
x/y = (3 - 0.3) / 3.75 = 0.72
```

This is the final policy that is optimal for agents. On average you save 3x, assuming length of 100 messages.
When input gets longer savings can go even to the 6x, and in the limit 9x.

To see the results run main.py
