# Numpy vectorization
Previously to implement lma I did:
```
kernel = []
left = (2/(N+1))
for k in range(N):
  kernel.append(left *  (1 - k * (1/N)))
return kernel
```

But back in my bachelor thesis, this is really slow. So I did some searching and used llm to explain the numpy implementations.
I came up with :
```
k = np.arange(N)
kernel = (2/(N+1)) * (1-k/N)
```

Explanation: np.arange returns an **numpy array** of range [0, 1, 2, 3, .., N]
Any operations done to numpy arrange are broadcasted, meaning it runs on SIMD vectorization making it faster than regular python loop.

# Ema filter weirdness
This is cross road between **blindly following formula** and **using common sense**

ema_filter is a bit weird, I tried N=3 and alpha=0.3 but it results in
```
kernels = [0.3   0.21  0.147]
sum = 0.657
```

sma and lma sums up to 1.0 (or sometimes 0.9999999999), this can botch the weight scaling.

The spec sheet also didn't mention about this. And if we blindly follow the formula, the scale will be uneven.

The common sense as we are taught in ML and DL classes is, normalize all the values to be of the same scale.

So I implemented flag to normalize the value by doing `kernel/sum of kernels`. This way, the result became:
```
[0.456621   0.3196347  0.22374429]
0.9999999999999999
```

I know this is not in the spec sheet however we are allowed to implement custom filter and I think this 
might be more correct since the scale matches.

# Something about the kernel
If you notice it, even though we got lma as:
```
[0.5        0.33333333 0.16666667]
```

Which might look so wrong because we want greater weight on last price instead of first price:
| weight | price at |
|--------|----------|
| 0.167  |  t - 2   | 
| 0.333  |  t - 1   | 
| 0.5    |  t       | 

But we don't need to worry as if we apply it with convolve, the order is reversed, we can prove it with
testing with array of 1 and 0:
```
import numpy as np
print(lma_filter(3))
print(np.convolve([0, 0, 1], lma_filter(3), 'valid'))
print(np.convolve([0, 1, 0], lma_filter(3), 'valid'))
print(np.convolve([1, 0, 0], lma_filter(3), 'valid'))

### Results in
[0.5        0.33333333 0.16666667]
[0.5]
[0.33333333]
[0.16666667]
```

This is expected, if we read convolve source code for `def convolve(a, v, mode='full'):`,
this happens down in the code `return multiarray.correlate(a, v[::-1], mode)`
v got flipped with `v[::-1]` where v is the kernel and a is the series data.

I didn't change the source code for example provided by project 2 brief so I wonder how it is correct too.

*When in doubt, crack open the source code XD

# Something about performance
Somehow EMA works better than LMA and SMA only. EMA with the shared alpha usually have the highest return in train set, however they often never cross any threshold on test, making the equity did not change.

Weighted one is the absolute beast at the moment.

# On PSO
Both Particle Swarm Optimization (1995) and A Modified Particle Swarm Optimizer (1998) did not mention best starting
velocity. So I will set it into hyperparameter max_vel_frac. This hyperparameter will also set the maximum velocity
the particles can go.

The reason being if no velocity is set, the velocity may explode (the particle can zip from one side to the end of other size of the axis instantly)

The variable naming are weird like energy and position and that is because I "ported" my AOS code.

Shi and Eberheart in 1998 mentioned that w works best between 0.9 to 1.2, and they used decreasing function. So I implemented that. Also in order to not have the weight turn to 0. It is important to set a minimum weight possible. 
0.4 somehow became a community standard and that will be the number I go with.