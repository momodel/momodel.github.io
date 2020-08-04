# 36-GPU 编程之从零开始实现 MNIST-CNN

很多人最开始接触“ GPU ”想必都是通过游戏，一块高性能的 GPU 能带来非凡的游戏体验。而真正使GPU被越来越多人熟知是因为机器学习、深度学习的大热（也有人用于比特币挖矿），因为庞大的数据与计算量需要更快的处理速度，GPU 编程也因此越来越普遍。
从事深度学习的工作者常常自嘲自己为“炼丹师”，因为日常工作是：搭网络，调参，调参，调参......作为刚入门深度学习的小白更是如此，虽然不停的复现着一个又一个的网络，但总有些迷茫。我想这个迷茫来源于深度学习的“黑盒”性质，而我们所做的工作绝大部分时间又仅是调参，对网络内部的计算实际是模糊的。因此，本文试着结合 GPU 编程从零开始写一个简单的 CNN 网络，从内部深入理解网络的运行，希望能够一定程度上消除这种迷茫，也通过这一个简单的应用来了解 GPU 编程。		
    因此本文主要分为两个部分：

- GPU 编程的介绍与入门。
- 使用 GPU 编程从零开始搭建 MNIST-CNN。
## 1 GPU 编程的介绍与入门
### 1.1 介绍

图 1 为 CPU 与 GPU 的物理结构对比（图片源自网络）。图中的有色方块代表了处理核心，虽然 CPU 的核心比较少（如图中只有8块），但每一个算能力都非常强；而 GPU 的核心非常多，但每一个的能力有限。核心数量决定着处理计算过程的线程数量，因此对于计算问题来说，GPU 比 CPU 能调度更多的线程，这也是为什么 GPU 可以用来加速的原因。但仅仅靠 GPU 还不行，一些复杂的控制流程仍需要 CPU 来做，因此现在的计算机基本是 CPU+GPU 的形式。CPU 控制整体程序的调度，当需要繁重的计算任务时，将其交给 GPU 完成。
![cpu-and-gpu.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/1715460/1595722142583-fb8698ba-89ec-4b65-ae18-1cc4759cf381.jpeg#align=left&display=inline&height=285&margin=%5Bobject%20Object%5D&name=cpu-and-gpu.jpg&originHeight=285&originWidth=416&size=38158&status=done&style=none&width=416)
图 1  CPU 与 GPU 物理结构对比
### 1.2 使用Numba进行GPU编程
Nvidia 提供了在自家生产的 GPU 上进行编程的框架，该框架在不同的编程语言上都有实现。本文介绍一种支持 Python 的 GPU 编程模型—— Numba（在下面的内容开始之前，请确保你的 Pyhon 编译器中是有 Numba 模块的，如果没有请先安装，安装方式与一般的模块相同）。
在 GPU 编程中我们称在 CPU 端为Host，GPU 端为 Device，在 CPU 运行的函数为主机函数，在GPU 运行的、受CPU 调用的函数为核函数， 在 GPU 运行、被 GPU 调用的函数为设备函数。在 GPU 上运行的函数的定义需要有一个专门的函数修饰符，如
```python
from numba import cuda
import numpy as np

@cuda.jit()
def my_kernel(io_array): # 核函数
	# To do

@cuda.jit(device=True)
def device_func(io_array): # 设备函数
    # To do
```
调用核函数的过程为：主机端先将数据传入设备端，主机端调用核函数在 GPU 上进行计算，完成计算后再将结果数据传回主机端。使用 Numba 的一个好处是不需要这些额外的数据传输操作，核函数能自动对 Numpy 数组进行传输。值得注意的是，核函数不能有返回值，GPU 编程的一大特点是函数的计算结果都通过参数传递。
那如何控制线程进行处理呢？——用数组索引与线程 id 相互对应的方式。比如我要实现一个函数，使得矩阵 A 的每一个元素都加 1，则代码可直接按如下方式实现
```python
from numba import cuda
import numpy as np

@cuda.jit()
def my_kernel(io_array):
	tx = cuda.threadIdx.x # 获取当前线程的 id 分量
    ty = cuda.threadIdx.y # 获取当前线程的 id 分量
    tz = cuda.blockIdx.x # 获取当前线程的 id 分量
    io_array[tz, tx, ty] += 1 # 用该id 对应数组的索引，处理对应的元素
    
A = np.zeros(shape=[10, 32, 32])
gridsize = (10)
blocksize=(32, 32)
my_kernel[gridsize, blocksize](A) # 特殊的调用方式，调用时需要告诉核函数线程的组织情况
print(A)
```
上述代码的思想是：对每一个A矩阵的元素都调用一个线程处理，互相对应的准则是线程的 id 与 A矩阵的索引相对应。gridesize 与 blockszie 定义了一个与 A 矩阵相同结构的线程矩阵，其中的每一个线程都会调用该核函数，核函数中的 tx、ty、 tz 变量表示了当前线程的 id，因此每个线程执行时实际上只处理了各自 id 所对应的元素。gridsize 与 blocksize 变量的定义可以不考虑底层的物理结构，Numba 会自动替我们分配以实现代码层到物理层线程的映射。实际 GPU 编程需要的事项有很多，这里不一一列举，想深入了解可查阅官方手册。但有了上述编程思想的基础，我们已经可以来编写简单的神经网络了。
## 2 结合 GPU 从零搭建 CNN
CNN 网络的主要结构有全连接层、卷积层、池化层、非线性激活层以及输出损失层，各层都有许多参数选择，这里仅搭建 MNIST-CNN 所包含的结构，即简单的卷积层、池化层、reLu 激活函数、全连接层以及 Cross entropy 损失输出。
本文着重用 GPU 实现卷积层与池化层的前向传播与链式求导。全连接与激活等计算只涉及到矩阵的乘法相对简单，因此直接用 Numpy 处理（也由于篇幅限制，这里也不再赘述这些操作的原理及代码，整个网络的代码可以到本人的 Github 上下载，文末附有代码链接）。
### 2.1 结合 GPU 实现池化与卷积
先实现池化层，这里只简单设计一个 ![](https://cdn.nlark.com/yuque/__latex/e9e2d0e7f31469e64f6434cd932d5861.svg#card=math&code=2%20%5Ctimes%202&height=16&width=37) 的最大池化操作，池化层的前向计算与反向计算的原理如图 2 所示（图片来源于网络）
![pool_back.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/1715460/1595724417871-73ce007d-5555-4ccf-af72-4b0e2626915c.jpeg#align=left&display=inline&height=720&margin=%5Bobject%20Object%5D&name=pool_back.jpg&originHeight=720&originWidth=1280&size=50763&status=done&style=none&width=1280)
图2  池化层传播计算原理
它前向传播的原理比较简单，其反向传播需要用到图示的 Index 矩阵。这里的实现方法为每一次前向传播都先保存输入与输出，在反向传播时将输入数据与输出数据做对比，数值相等的位置即为池化输出的位置，因此反向传播的梯度应传递到该位置，其它不相等的位置置为 0 。其代码实现如下
```python
class MaxPooling2x2(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1]//2, inputs.shape[2]//2, inputs.shape[3]), dtype=np.float32)
        grid = (inputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        pool[grid, block](self.inputs, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        grid = (self.outputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        cal_pool_gradient[grid, block](self.inputs, self.outputs, gradient_loss_this_outputs, self.g_inputs)
        return self.g_inputs   
```
前向传播与反向传播的核心计算过程由 GPU核函数实现。前向传播时，对于单个样本每个线程处理一块![](https://cdn.nlark.com/yuque/__latex/e9e2d0e7f31469e64f6434cd932d5861.svg#card=math&code=2%20%5Ctimes%202%0A&height=16&width=37) 的区域从而输出最大值，并循环处理一个 batchSize；其反向传播时线程安排与前向传播相同。代码如下
```python
@cuda.jit()
def pool(inputs, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for i in range(inputs.shape[0]):
        outputs[i, tx, ty, d] = max(inputs[i, 2 * tx, 2 * ty, d], inputs[i, 2 * tx + 1, 2 * ty, d],
                                    inputs[i, 2 * tx, 2 * ty + 1, d], inputs[i, 2 * tx + 1, 2 * ty + 1, d])


@cuda.jit()
def cal_pool_gradient(inputs, outputs, gradient_to_outputs, grident_to_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for k in range(outputs.shape[0]):
        for i in range(2):
            for j in range(2):
                if outputs[k, tx, ty, d] == inputs[k, 2 * tx + i, 2 * ty + j, d]:
                    grident_to_inputs[k, 2 * tx + i, 2 * ty + j, d] = gradient_to_outputs[k, tx, ty, d]
```
卷积操作的正向传播与前向传播原理不再赘述。程序实现上卷积层同样需要保存输入与输出。前向传播时，设置线程矩阵的 size 与输出张量的 size 相同，从而使一个线程处理一个输出，循环 batchSize 次完成所有输出的计算；在梯度反向传播时，设置线程矩阵的 size 与损失对该卷积层输出的梯度的 size 相同，每个线程各自计算与该输出元素相关的输入与参数应累加的梯度数值，并循环 batchSize 次，所有线程计算完毕后，最终结果即为损失分别对输入与参数的梯度，其代码实现如下
```python
class Conv2D(object):
    def __init__(self, in_channels, kernel_size, features):
        self.features = features
        self.ksize = kernel_size
        weights_scale = np.sqrt(kernel_size * kernel_size * in_channels / 2)
        self.weights = np.random.standard_normal((features, kernel_size, kernel_size, in_channels)) / weights_scale
        self.biases = np.random.standard_normal(features) / weights_scale

        self.g_weights = None
        self.g_biases = None
        self.g_inputs = None
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1]+(self.ksize // 2)*2,
                                      inputs.shape[2] + (self.ksize // 2)*2, inputs.shape[3]), dtype=np.float32)
        self.inputs[:, self.ksize // 2: inputs.shape[1] + self.ksize // 2,
        self.ksize // 2: inputs.shape[2] + self.ksize // 2, :] = inputs.copy()
        self.outputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.features), dtype=np.float32)
        grid = (self.features)
        block = (inputs.shape[1], inputs.shape[2])
        cov[grid, block](self.inputs, self.weights, self.biases, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        self.g_weights = np.zeros(self.weights.shape, dtype=np.float32)
        self.g_biases = np.zeros(self.biases.shape, dtype=np.float32)
        grid = (self.features)
        block = (self.inputs.shape[1], self.inputs.shape[2])
        cal_cov_grident[grid, block](self.inputs, self.weights, gradient_loss_to_this_outputs,
                                     self.g_weights, self.g_biases, self.g_inputs)

        self.g_inputs = self.g_inputs[:, self.ksize//2: self.g_inputs.shape[1] - self.ksize//2,
                        self.ksize//2: self.g_inputs.shape[2] - self.ksize//2, :]
        return self.g_inputs

    def update_parameters(self, lr):
        self.weights -= self.g_weights * lr / self.inputs.shape[0]
        self.biases -= self.g_biases * lr / self.inputs.shape[0]
        
@cuda.jit()
def cov(inputs, weights, biases, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(inputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    outputs[n, tx, ty, f_num] += (inputs[n, tx + i, ty + j, k] * weights[f_num, i, j, k])
        outputs[n, tx, ty, f_num] += biases[f_num]


@cuda.jit()
def cal_cov_grident(inputs, weights, gradient_loss_to_this_outputs, g_weights, g_biases, g_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(gradient_loss_to_this_outputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    tmp1 = gradient_loss_to_this_outputs[n, tx, ty, f_num] * weights[f_num, i, j, k]
                    tmp2 = gradient_loss_to_this_outputs[n, tx, ty, f_num] * inputs[n, tx+i, ty+j, k]
                    g_inputs[n, tx+i, ty+j, k] += tmp1
                    g_weights[f_num, i, j, k] += tmp2
        g_biases[f_num] += gradient_loss_to_this_outputs[n, tx, ty, f_num]
```
### 2.2 搭建MNIST CNN 模型
首先导入已经实现的各个模块，并读取数据集
```python
import numpy as np
import time
from read_mnist import DataSet
from my_deep_learning_pkg import FullyConnect, ReLu, CrossEntropy, MaxPooling2x2, Conv2D

# read data
mnistDataSet = DataSet()
```
利用自己写的类搭建网络
```python
# construct neural network
conv1 = Conv2D(1, 5, 32)
reLu1 = ReLu()
pool1 = MaxPooling2x2()
conv2 = Conv2D(32, 5, 64)
reLu2 = ReLu()
pool2 = MaxPooling2x2()
fc1 = FullyConnect(7*7*64, 512)
reLu3 = ReLu()
fc2 = FullyConnect(512, 10)
lossfunc = CrossEntropy()
```
训练中的前向传播、反向传播、以及梯度更新
```python
# train
lr = 1e-2
for epoch in range(10):
    for i in range(600):
        train_data, train_label = mnistDataSet.next_batch(100)

        # forward
        A = conv1.forward(train_data)
        A = reLu1.forward(A)
        A = pool1.forward(A)
        A = conv2.forward(A)
        A = reLu2.forward(A)
        A = pool2.forward(A)
        A = A.reshape(A.shape[0], 7*7*64)
        A = fc1.forward(A)
        A = reLu3.forward(A)
        A = fc2.forward(A)
        loss = lossfunc.forward(A, train_label)

        # backward
        grad = lossfunc.backward()
        grad = fc2.backward(grad)
        grad = reLu3.backward(grad)
        grad = fc1.backward(grad)
        grad = grad.reshape(grad.shape[0], 7, 7, 64)
        grad = pool2.backward(grad)
        grad = reLu2.backward(grad)
        grad = conv2.backward(grad)
        grad = grad.copy()
        grad = pool1.backward(grad)
        grad = reLu1.backward(grad)
        grad = conv1.backward(grad)

        # update parameters
        fc2.update_parameters(lr)
        fc1.update_parameters(lr)
        conv2.update_parameters(lr)
        conv1.update_parameters(lr)
```
在测试集上进行检测
```python
test_index = 0
sum_accu = 0
for j in range(100):
    test_data, test_label = mnistDataSet.test_data[test_index: test_index + 100], \
                            mnistDataSet.test_label[test_index: test_index + 100]
    A = conv1.forward(test_data)
    A = reLu1.forward(A)
    A = pool1.forward(A)
    A = conv2.forward(A)
    A = reLu2.forward(A)
    A = pool2.forward(A)
    A = A.reshape(A.shape[0], 7 * 7 * 64)
    A = fc1.forward(A)
    A = reLu3.forward(A)
    A = fc2.forward(A)
    preds = lossfunc.cal_softmax(A)
    preds = np.argmax(preds, axis=1)
    sum_accu += np.mean(preds == test_label)
    test_index += 100
print("epoch{} train_number{} accuracy: {}%".format(epoch+1, i+1, sum_accu))
```
程序的输出如图 3 所示，随着训练的进行，准确率不断上升，在一个 epoch 后准确率已经达到 91% , 说明代码是正确的。
![result.png](https://cdn.nlark.com/yuque/0/2020/png/1715460/1595726198303-060872bc-0728-4f12-aced-b371d0c6b2a3.png#align=left&display=inline&height=198&margin=%5Bobject%20Object%5D&name=result.png&originHeight=198&originWidth=488&size=8635&status=done&style=none&width=488)
图 3  程序运行结果
## 代码下载链接
[https://github.com/WHDY/mnist_cnn_numba_cuda](https://github.com/WHDY/mnist_cnn_numba_cuda)
## 引用
[[1] https://zhuanlan.zhihu.com/c_162633442](https://zhuanlan.zhihu.com/c_162633442)
[[2] https://github.com/SpiritSeeker/cnn-from-scratch](https://github.com/SpiritSeeker/cnn-from-scratch)
