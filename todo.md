
# Rework
- [x] Make tensor use `data_handler` object instead of simple variant
  with pointer and cuda pointer
	- [x] Make a 'handler' that will store both cuda and buffer holders
	- [x] Will make pointer calls through that object:
    	- fill, bytesize
  	- The 'holders' will actually implement these functions
  	- The 'handler' will keep these objects as variant of shared pointers
  	- The base 'holder' class may add field: bytesize

- [x] `ops_tensor` must be adjusted to set properly share the buffer
- [ ] Make tensor a generic class (constrained to integral types (int, uint) and floating point types (fl32, fl16, fl64))
- [x] Remove dogshit 'backend' and 'matrix' code
- [x] Rewrite the 'mat_mul_test' to seprate folder (like bench, and use proper profiling)
- [ ] Add tests for cpu and opencl operation backends

# Cpu
- [ ] Try zeroing the values that fall outside the mini-kernel when packing the matrices (on the edges) instead of using 4 different kernels (8x8, 1x8, 8x1, and naive mat mul for corner) - just use the 8x8 (it will do the additional multiplications at the cost of fast 8x8 fma) 









# Outdated

TODO:
- [ ] In cuda code, use `throw` instead of `exit` to handle errors.
- [ ] Add more tests:
	- [x] Test all operations with `cuda` backend (against cpu).
- [x] Make optimized backend default, and make it discover if cuda device is available, else use cpu.
- [x] Optimize `cuda` backend (rn 175ms) -> (127ms - on debug, 64ms) stop using `cudaMalloc` and `cudaMemcpy` for every operation,
instead just check if the previous size is enough, if not, allocate new memory.
- [ ] Make better backend, with optimizing copy calls on cuda, instead of calculating individual operations, see the chapter

### New project (for calculations)
- [x] Create an ops backend, with python like functions for performing calculations on tensors (right now only on 1D, 2D)
- [ ] Should use best libraries for matrix operations (cublas, gemm)
- [ ] Should support target device (either cpu or cuda)
- [x] Make the cpu handle tensors with && and & for better performance (ommited with new ops_tensor class)

Maybe add dispatching operations, and then call 'compute' to run them, so I can leverage the cuda streams and do async operations.

For example:

we want to calculate:
```
A, E, I

dA = derivative(A)
dC = d_cost(A, E)
P = da % dC

dW += I.T * P
dB += P
```


```cpp
operations ops()

tensor A({1, 64}, 2.0);
tensor E({1, 64}, 1.0);
tensor I({1, 64}, 5.0);
tensor W({64, 64}, 0.0);
tensor B({1, 64}, 0.0);
tensor dA({1, 64}, 0.0);
tensor dC({1, 64}, 0.0);
tensor P({1, 64}, 0.0);
tensor dW({64, 64}, 0.0);
tensor dB({1, 64}, 0.0);

ops.prepare(4); // prepare 4 buffers
ops.dispatch(ops.drelu, A) // gets index 0 result
ops.dispatch(ops.mean_squared_error, A, E) // gets index 1 result
ops.dispatch(ops.hadamard, ops.get_result(0), dC) // gets index 2 result
ops.dispatch(ops.mat_mul, I.T(), P) // gets index 3 result
ops.dispatch(ops.mat_mul, ops.get_result(3), dW) // gets index 4 result
ops.dispatch(ops.add, dW, ops.get_result(2)) // gets index 5 result
ops.dispatch(ops.add, dB, ops.get_result(2)) // gets index 6 result

```

For example:

```cpp

operations ops()

tensor m1({128, 64}, 2.0);
tensor m2({64, 64}, 1.0);
tensor m3({128, 64}, 5.0);

ops.add(
	ops.mat_mul(m1, m2),
	m3
);

ops.sub(
	m1, m3
);

tensor t = ops.relu(m3);

// the same
ops.map(t, ops.relu)

```



## Optimized cuda backend

Let's take a look at an example:

`M1 * M2 - M3`

Will result in: 
1. copy M1 -> CD1, M2 -> CD2, calculate M1 * M2 on cuda, copy result (R) from RC, 
2. Copy data again to buffers, R -> CD1, M3 -> CD2, calculate R - M3, copy result R from RC. 

This simple operation made 2 (to Device) + 1 (from Device) +
2 (tD) + 1 (fD) = 4 (td) + 2 (fd) copy operations. While using only 3 buffers. 
If optimized it won't allocate any new buffers, resulting in only these copy operations

Creating separate objects with cuda buffers for each operation and then 'merging' them, would be an interesting 
solution, if the memory allocation on gpu wasn't a bottleneck. 

For example the operation above would result in:

```text
create Object o1(M1, M2)
o1.mul()
create Object o2(o1.result, M3) // would be optimized with constructor, to not copy the 'result' (as it would be gpu memory pointer)
o2.sub()
o2.get_result(&result) 
```

This would allocate memory on cuda device for M1, M2, result of M1 * M2, and M3 (4 total), and 4 copies

One optimization for the current implementation would be chaining the operations.

```c++
backend.substract(
	backend.multiply(M1, M2),
	M3
);
```

So that, there is no copy from result device to result, result to device data 1.

```text
copy M1 -> CD1, M2 -> CD2, RC = CD1 * CD2
copy M3 -> CD1, RC = RC - CD1
copy RC -> result host
```

That is 3 copies from host to device, 1 copy from device to host.

With more complex example:

`M * V - alpha * G`

```c++
backend.substract(
	backend.multiply(M, V),
	backend.multiply_scalar(alpha, G)
);
```

In current version:

```text
copy M -> CD1 and V -> CD2, RC = CD1 * CD2
copy to host RC -> R1
copy G -> CD1, RC = alpha * CD1
copy to host RC -> R2
copy R1 -> CD1 and R2 -> CD2, RC = CD1 - CD2
copy to host RC -> R
```

Total of: 5 (tD) + 3 (fD)

In optimized version:

```text
copy M1-> CD1 and V -> CD2, RC = CD1 * CD2
copy G -> CD1, resize CD2 = CD1, CD2 = CD1 * alpha
RC = RC - CD2
copy to host RC -> R
```

Total of: 3 (tD) + 1 (fD)

What I am trying to achieve here, is to avoid redundant copying of `result` buffer on device.
However, with only 3 buffers it is impossible to calculate to avoid such copy in the following scenario:

`M * V - G * D`

M,V,G,D - matrices

So I can make another buffer to store the second result, now with total of 4 buffers (R1, R2, DC1, DC2),
such operation would be copy-free (of the results)

Creating such backend will be a pain in the ass...

And it's probably not going to be worth it anyway. If I create dedicated functions for backpropagation

