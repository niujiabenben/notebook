# Python C API

python中调用c一般有以下几个途径:

1. python ctypes: 

2. python c api: 

3. boost python:

4. swig python:

## ctypes

ctypes是Python标准库提供的调用动态链接库的模块, 使用这个模块可以直接在Python里加载动态链接库, 调用其中的函数. 使用ctypes时, 可以不用更改c代码文件, 而是通过ctypes指定输入输出的参数类型. 例如, 有c函数:

```cpp
#include <algorithm>
#include <numeric>

//// 注意, ctypes只能识别c接口, 所以这里必须用extern "C"
extern "C" double sum(const double* vec, const int N) {

  return std::accumulate(vec, vec + N, 0.0);

}
```

将上述代码保存至`ext.cpp`, 编译命令为:

```shell
g++ -g -Wall -fpic -O3 -c -o ext.o ext.cpp

g++ -shared -o libext.so ext.o
```

python端需要对函数进行包装后调用:

```python
from ctypes import *

### load动态链接库
ext = cdll.LoadLibrary("libext.so")

### 指定输入输出类型, 这里POINT是一个类型包装器, 将基础类型包装成指针类型
ext.sum.argtypes = [POINTER(c_double), c_int]

ext.sum.restype = c_double

### 调用动态链接库的函数, 注意这里不能直接传递vec, 因为vec是list类型,
### 而sum函数需要的是double*. 实际上, 除了基础类型, 指针类型和结构体
### 都需要通过ctypes中对应的结构进行封装.
vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

value = ext.sum((c_double * len(vec))(*vec), len(vec))


### 这里我们也可以通过numpy来封装.
arr = np.array(vec, dtype=np.double)
value = ext.sum(arr.ctypes.data_as(POINTER(c_double)), len(arr))


### 我们可以直接指定numpy.ndarray的类型, 这样可以直接传递numpy.ndarray
ext.sum.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C"), 
    c_int

]
value = ext.sum(arr, len(arr))
```

注意: **我们在测试时发现, 采用numpy来封装时, 有一些类型可能得不到正确的结果.** 例子如下:

```python
### float类型的sum()函数
extern "C" float sum(const float* vec, const int N) {
  return std::accumulate(vec, vec + N, 0.0f);
}

### 指定输入为float数组, 输出为float类型
ext.sum.argtypes = [POINTER(c_float), c_int]
ext.sum.restype = c_float

### 通过numpy来封装, 指定数据类型为np.float. 
### 但是, 传递到c中的sum函数之后, 似乎其内存分布仍为double*类型. 如果以
### float*类型去取数据, 则会得到错误的结果.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
value = ext.sum(arr.ctypes.data_as(POINTER(c_float)), len(arr))
```

我倾向于这是一个numpy.ctypes的一个bug, 这里并不去深究. 只是在使用的时候需要格外注意.

### Struct和内存管理

假设我们的函数输入的是struct, 则需要在python端声明这个struct中的内容. 这里我用矩阵乘积来做例子. ext.cpp的内容为:

```cpp
//// 定义一个结构体存放二维矩阵
struct Matrix2D {
  double* data;
  int rows;
  int cols;
};

//// 这里实现了一个简单的矩阵乘法, 用Matrix2D做参数
//// 这里可以用const, 但不能用引用, 因为c中没有引用
extern "C" Matrix2D matmul(Matrix2D a, Matrix2D b) {
  Matrix2D res;
  res.rows = a.rows;
  res.cols = b.cols;
  res.data = new double[res.rows * res.cols];
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      double value = 0.0;
      for (int k = 0; k < a.cols; ++k) {
        int id_a = i * a.cols + k;
        int id_b = k * b.cols + j;
        value += a.data[id_a] * b.data[id_b];
      }
      res.data[i * res.cols + j] = value;
    }
  }
  return res;
}

//// matmul返回一个Matrix2D, 其中data字段指向的内存是在c函数里申请的, 
//// 所以需要额外写一个释放函数, python端需要显式调用这个函数.
extern "C" void release(Matrix2D m) {
  delete[] m.data;
}
```

python端的内容为:

```python
from ctypes import *
import numpy as np

### 这里申明Matrix2D结构体的参数名称和类型, 必须继承自struct类
class Matrix2D(Structure):
    _fields_ = [
        ("data", POINTER(c_double)),
        ("rows", c_int),
        ("cols", c_int)
    ]

### load动态库之后, 设置函数参数和返回值类型
ext = cdll.LoadLibrary("libext.so")
ext.matmul.argtypes = [Matrix2D, Matrix2D]
ext.matmul.restype = Matrix2D
ext.release.argtypes = [Matrix2D]
ext.release.restype = None

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
MA = Matrix2D(A.ctypes.data_as(POINTER(c_double)), *A.shape)
MB = Matrix2D(B.ctypes.data_as(POINTER(c_double)), *B.shape)
MC = ext.matmul(MA, MB)

### MC.data实际上是一个double类型的指针, 可以取下标, 但是不可以取长度
for i in range(MC.rows * MC.cols): print(MC.data[i])


### np.fromiter可以读取一块连续的内存, 这里实际上做了一个拷贝
C = np.fromiter(MC.data, dtype=np.float64, count=MC.rows*MC.cols)
C = C.reshape(MC.rows, MC.cols)

### ctypes内存管理遵循谁申请谁负责的原则, MC.data中的内存由c代码申请,
### 所以需要在python端显式调用release函数
ext.release(MC)
```

上面的代码中, 给matmul函数传递参数的时候不需要内存拷贝, 但是将返回值MC转换成numpy格式的时候, 需要将MC.data中的内容拷贝到生成的numpy数组中. 并且, ctypes要求: **在c端申请的内存, 必须在c端进行释放.**  所以, 我们还需要在c端定义一个释放函数release(), 然后在python端显式调用. 这种方式, 一方面牺牲了性能, 另一方面也影响了使用的体验. 因此, 通常的做法是, 所需要的内存都从python端申请, 然后传递给c端:

```python
### 更改后matmul的接口
extern "C" void matmul(const Matrix2D a, const Matrix2D b, Matrix2D res);

### load动态库之后, 设置函数参数和返回值类型
ext = cdll.LoadLibrary("libext.so")
ext.matmul.argtypes = [Matrix2D, Matrix2D, Matrix2D]
ext.matmul.restype = None
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
### 这里, 所有的内存都是在python端申请的, 不需要额外的接口释放
C = np.zeros((3, 5), dtype=np.double)
MA = Matrix2D(A.ctypes.data_as(POINTER(c_double)), *A.shape)
MB = Matrix2D(B.ctypes.data_as(POINTER(c_double)), *B.shape)
MC = Matrix2D(C.ctypes.data_as(POINTER(c_double)), *C.shape)
ext.matmul(MA, MB, MC)
```

如果在调用动态库中的函数之前不能确认返回值所需要的内存大小, 则可以传入一个内存上限, 然后返回最终使用的实际内存. 这里用一个简单的例子说明:

```cpp
//// input_buffer: 输入buffer起始位置
//// input_bytes:  输入buffer的大小
//// output_buffer: 输出buffer的起始位置
//// output_btypes: 输出buffer的总大小
//// return: 实际写入输出buffer的字节数
extern "C" int copy(const char* input_buffer, const int input_bytes,
                    char* output_buffer, const int output_bytes) {
  memcpy(output_buffer, input_buffer, input_bytes);
  return input_bytes;
}
```

python端可以用`ctypes.create_string_buffer`来申请一块内存:

```python
from ctypes import *

ext = cdll.LoadLibrary("libext.so")
ext.copy.argtypes = [POINTER(c_char), c_int, POINTER(c_char), c_int]
ext.copy.restype = c_int

### create_string_buffer返回的类型是: POINTER(c_char)
input_buffer = create_string_buffer(b"hello world")
output_buffer = create_string_buffer(200)
size = ext.copy(input_buffer, len(input_buffer), output_buffer, len(output_buffer))
```

调用结束后, `output_buffer.value`为NULL结束的字符串, `output_buffer.raw`为bytes类型. 

如果传入的参数较为复杂, 比如是一个dict, 则较难将其重新定义成`Structure`类型. 这时可以做一个序列化, 比如将dict转化成json格式的字符串, 或者自己在python上写一个序列化函数, 然后在c端写一个反序列化函数.






