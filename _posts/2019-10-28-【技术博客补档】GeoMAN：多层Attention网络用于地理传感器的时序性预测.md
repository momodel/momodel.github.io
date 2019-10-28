---
layout: post
title: GeoMAN：多层Attention网络用于地理传感器的时序性预测
date: 2019-10-28 12:00
---
作者： 魏祖昌

##  1 简介
在我们现实生活中，已经部署大量的传感器（比如气象站点）。每一个传感器都有自己独特的地理空间位置，并且不断的产生时间序列读数。一组传感器共同监测一个空间的环境，这些读数之间就会有空间相关性，我们称这些传感器的读数为**地理感知时间序列**。此外，当不同空间位置用同一种传感器来监测， 通常会产生多种地理感知的时间序列。例如，如图1(a)所示，道路上的环形探测器会及时报告过往车辆的读数以及它们的行驶速度。图1(b)表示传感器每5分钟产生三个不同的水质化学指标。除了监测之外，对地理感知时间序列预测(如交通预测)的需求也在不断增长。

![image.png](https://imgbed.momodel.cn/201919232059-4.png)

图1:(a)-(b)地理传感器时序性数据的例子

然而，对地理感知时间序列进行预测是非常复杂的，主要受以下两个复杂因素的影响：

  1. 动态的时空相关性。
  1. 外部因素。传感器的读数也受到周围环境的影响，如气象（如强风），一天中的时间（如高峰时间）和土地使用情况。

为了解决这些挑战，该论文提出了一个多层次的Attention网络(GeoMAN)来预测未来几个小时内地理传感器的读数。该论文的研究有三方面的贡献:

- **多层attention机制** 我们构建了一个多层attention机制来建模时空动态关联。尤其是在第一层，该论文提出了一种创新的attention机制（由local
spatial attention和global spatial attention组成）来捕获不同传感器时序性序列之间的复杂空间联系（比如传感器内部之间的联系）。在第二层，应用了一个temporal attention来建模在时间序列中不同时间间隔的动态时间关联（比如传感器之间的联系）。
- **外部因素抽取模块** 该模块设计了一个通用抽取模块来整合来自不同领域的外部参数。然后抽取出来的潜在代表性因素输入到多层attention网络中来增强这些外部因素的重要性。

## 2 多层Attention网络
图2展示了该片论文的整个框架。依据encoder-decoder框架，我们利用两个分离LSTM网络，一个用于对输入序列（比如地理传感器的历史时间序列）进行encoder，另一个则用于预测的输出序列。更具体的来说，论文中的GeoMAN模型主要是由两部分组成：
1. 多层attention机制。它的encoder部分用了两种spatial attention机制，decoder用了一个temporal attention机制。论文中在encoder层中用的两种不同attention机制（local spatial attention和global spatial attention）正如图2所示，它能通过encoder之前的隐藏状态，传感器的历史数据和空间信息（比如传感器网络）来捕获每个时间间隔之间的传感器内部之间的复杂关系。在decoder层中，使用了一个temporal attention来自动选择之前相似的时间间隔来进行预测。
2. 外部因素抽取。这个模块用于处理外部因素的影响，并且将其输入到decoder层，作为其输入的一部分。这里我们用ht和st来分别表示encoder层在t时刻的hidden state和cell state。类似的用dt和s'来表示decoder层的这两部分

![framework.png](https://imgbed.momodel.cn/201919232102-W.png)

图2:该论文的框架。**Attn**: attention. **Local**: local spatial attention. **Global**: global spatial attention. **Concat**: concatenation层. ![image.png](https://imgbed.momodel.cn/201919232104-Z.png): 在t时刻的predicting value. **ct**: 在t时刻的context vectors.  **h0**: encoder的初始值.

### 2.1 Spatial Attention
#### 2.1.1 Local Spatial Attention
该论文是首次引入了local spatial attention机制。对于某个传感器来说，其的局部时间序列之间存在复杂的相关性。比如，一个空气质量监测站报告不同物质的时间序列，如PM2.5(特定物质)，NO和SO2。实际上，PM2.5浓度通常受其他时间序列的影响，包括其他空气污染物和当地的天气状况。为了解决这个问题，给定第i个传感器的第k个局部特征向量(即，![image.png](https://imgbed.momodel.cn/201919232108-W.png)，我们利用attention机制自适应地捕捉目标序列与每个局部特征之间的动态相关性，其公式为:

![image.png](https://imgbed.momodel.cn/201919232108-v.png)

其中[ ·; ·]是合并操作，![image.png](https://imgbed.momodel.cn/201919232108-J.png)是学习来的参数。attention的权重的局部特征值是由输入的局部特征和encoder层中的历史状态（即![image.png](https://imgbed.momodel.cn/201919232113-k.png), ![image.png](https://imgbed.momodel.cn/201919232114-q.png)）共同决定的，这个权重值代表着每一个局部特征的重要性。一旦我们获得了attention的权值，就可以通过下面的公式算出在t时刻的local spatial atttention的输出向量：

![image.png](https://imgbed.momodel.cn/201919232109-D.png)

#### 2.1.2 Global Spatial Attention
其他传感器所监测的历史时间序列对将要预测出来的序列，会有直接的影响。然而，影响的权重是高度动态，是随时间变化的。由于会有很多不相关的序列，所以直接用所有的时间序列输入到encoder层来捕获不同传感器之间的相关性会导致非常高的计算成本并且降低性能。注意，这种影响的权重是受其他传感器的局部条件影响的。比如，当一股风从很远的地方吹来的时候，某些地区的空气质量会比之前更受这些地方的影响。受此启发，构建了一种新型的attention机制来捕获不同传感器之间的动态变化。给定第i个传感器作为我们的预测的对象，其他的传感器为l，我们就可以计算他们之间的attention权值（即影响权重），公式如下：

![image.png](https://imgbed.momodel.cn/201919232110-Y.png)

其中![image.png](https://imgbed.momodel.cn/201919232110-p.png)和![image.png](https://imgbed.momodel.cn/201919232110-N.png)是学习得来的参数。该attention机制通过参考目标序列和其他传感器的局部特征来自适应地选择相关传感器进行预测。同时，通过考虑编码器中先前hidden state: ![image.png](https://imgbed.momodel.cn/201919232113-k.png)和cell state ![image.png](https://imgbed.momodel.cn/201919232114-q.png)来跨越时间步长传播历史信息。

注意，空间因素也会影响不同传感器之间的相关性。一般来说，地理传感器是通过显式或隐式进行相互连接的。这里，我们使用矩阵![image.png](https://imgbed.momodel.cn/201919232114-w.png)来表示地理空间的相似度，其中![image.png](https://imgbed.momodel.cn/201919232115-d.png)表示传感器i和j之间的相似度。与attention权值不同的是，地理空间的相似度可以视作先验知识。尤其是，在如果**Ng**太大，选择最近或相似的传感器会更好。然后，我们使用一个softmax函数来保证所有的attention权值和为1，结合考虑地理空间相似性得出如下的公式：

![image.png](https://imgbed.momodel.cn/201919232116-0.png)

其中![image.png](https://imgbed.momodel.cn/201919232117-F.png)是一个可调的超参数。如果![image.png](https://imgbed.momodel.cn/201919232117-F.png)很大，这个公式就会使attention的权重和地理空间相似度一样大。通过这些attention权值，我们就可以计算出global spatial attention的如下输出向量：

![image.png](https://imgbed.momodel.cn/201919232117-w.png)

### 2.2 Temporal Attention
由于随着编码长度的增加，encoder-decoder结构的性能会迅速下降，所以增添一个temporal attention机制可以自适应地选择encoder层的相关hidden states来产生输出序列，即，对预测序列中不同时间间隔之间的动态时间相关性进行建模。具体来说，为了计算encoder每个hidden
state下每个输出时间t‘处的attention向量，我们定义：

![image.png](https://imgbed.momodel.cn/201919232118-5.png)

![image.png](https://imgbed.momodel.cn/201919232118-Q.png)

其中![image.png](https://imgbed.momodel.cn/201919232119-F.png)和![image.png](https://imgbed.momodel.cn/201919232119-5.png)都是学习得来的。这些值被一个softmax函数标准化，以创建encoder层隐藏状态上的attention掩码。

### 2.3 外部因素抽取
地理传感器的时间序列和空间因素（比如POIs和传感器网络之间）有很强的关系。形式上，这些因素共同决定了一个区域的功能。此外，还有很多时间因素（如气象和时间）在影响着传感器的读数。在受相关论文启发之下，该论文设计了一种简单有效的组建来处理这些因素。
正如上面的图2所示，首先合并了包括时间特征，气象特征和需要被预测传感器的SensorID等时间因素。由于未来时段的天气情况未知，我们使用天气预报来提高我们的性能。注意，这些因素大部分是分类的值，不能直接输入到神经网络中，我们将每个分类的属性分别输入到不同的embedding层中，将它们转化为一个低维向量。在空间因素方面，我们利用不同类别的POIs密度作为POIs特征。由于传感器网络的特性取决于特定的环境，我们就简单的利用了网络的结构特征（如居民和交叉口数量）。最后，我们将得到的嵌入向量和空间特征向量连接起来作为该模块的输出，记为![image.png](https://imgbed.momodel.cn/201919232119-K.png)，其中t‘表示decoder层的未来时间步长。

### 2.4 Encoder-decoder和模型训练
在encoder层中，我们将local spatial attention 和 the global spatial attention简单汇总成：

![image.png](https://imgbed.momodel.cn/201919232120-6.png)

我们把连接而成的![image.png](https://imgbed.momodel.cn/201919232120-Z.png)作为encoder层新的输入，并且用![image.png](https://imgbed.momodel.cn/201919232120-t.png)来更新t时刻的hidden state，其中![image.png](https://imgbed.momodel.cn/201919232127-1.png)是一个LSTM单元。
在decoder层中，一旦我们获得了未来t‘时刻的![image.png](https://imgbed.momodel.cn/201919232127-u.png)的环境向量，我们就可以把它和外部特征抽取模块的输出![image.png](https://imgbed.momodel.cn/201919232128-s.png)和decoder层最后一个输出![image.png](https://imgbed.momodel.cn/201919232128-T.png)结合起来去更新decoder层的hidden state，公式如下：

![image.png](https://imgbed.momodel.cn/201919232128-X.png)

其中![image.png](https://imgbed.momodel.cn/201919232129-l.png)是使用在decoder层中LSTM单元。然后，我们再把先前的环境向量![image.png](https://imgbed.momodel.cn/201919232130-M.png)和现在得到的hidden state![image.png](https://imgbed.momodel.cn/201919232130-u.png)结合起来，成为新的hidden state来做如下的最终的预测：

![image.png](https://imgbed.momodel.cn/201919232131-w.png)

最后，我们使用一个线性变换来产生最后的输出。
因为这个方法是光滑可谓的，所以是可以通过反向传播算法来进行训练的模型的。在这个训练阶段，我们使用的Adam优化器来最小化传感器i的预测向量![image.png](https://imgbed.momodel.cn/201919232131-2.png)和实际测量值![image.png](https://imgbed.momodel.cn/201919232131-U.png)之间的MSE来训练这个模型：

![image.png](https://imgbed.momodel.cn/201919232131-L.png)

其中![image.png](https://imgbed.momodel.cn/201919232132-1.png)都是在所提出的模型中学习来的。

## 3 实验
### 3.1 实验数据
该论文中用了两个数据集分别来训练该模型，数据集的详细内容如图3所示：

![image.png](https://imgbed.momodel.cn/201919232132-i.png)

图3:数据集的详细内容。

但是由于完整数据没有公开的问题，我们后面复现是使用的是一个他提供的sample_data，即是他处理完之后得到的向量，所以这部分不做深入介绍，如果对这部分还有疑问或者兴趣，可以自行参考论文相应部分。

### 3.2 评价指标
我们使用多个标准来评估我们的模型，包括根均方误差(RMSE)和平均绝对误差(MAE)，这两个标准在回归任务中都被广泛使用。

### 3.3 超参数
鉴于先前的一些研究，该论文设置时间间隔为6天来做短期预测。在训练过程中，我们设batch的大小为256，learning rate为0.001。在外部特征自动抽取模块，论文把SensorID嵌入到![image.png](https://imgbed.momodel.cn/201919232133-G.png)中，把时间特征嵌入到![image.png](https://imgbed.momodel.cn/201919232134-q.png)中。总的来说，在这个模型中有4个超参数，其中权衡参数![image.png](https://imgbed.momodel.cn/201919232134-4.png)从经验上来说是固定在0.1到0.5之间的。对于窗口长度T，我们令T∈{6,12,24,36,48}。对了简单起见，我们在encoder层和decoder层使用相同维度的hidden层，并且在{32, 64, 128, 256}上进行网格搜索。此外，我们使用堆叠的LSTMs(层数记作q)作为encoder和decoder的单位，以提高我们的性能。实验发现，在设置q=2，m=n=64,![image.png](https://imgbed.momodel.cn/201919232134-4.png)=0.2时在验证集表现的最好。

## 4. 模型对比
在本节中，我们将论文的模型与两个数据集上进行比较。为了公平起见，在图4中给出了不同参数设置下每种方法的最佳性能。

![image.png](https://imgbed.momodel.cn/201919232135-Y.png)

图4:在不同模型中的表现对比

在水质预测方面，我们提出的方法在两个指标上都明显优于其他方法。特别地，GeoMAN在MAE和RMSE上分别以14.2%和13.5%超过了最先进的方法(DA-RNN)。另一方面，由于residual chlorine（RC）的浓度遵循一定的周期规律，因此stDNN和RNN方法(即Seq2seq, DA-RNN和GeoMAN)通过考虑更长的时间关系，获得了比stMTMVL和FFA更好的性能。与LSTM对未来时间步长的预测相比，GeoMAN和Seq2seq由于解码器组件的积极作用而带来了显著的改进。值得注意的是，GBRT在大多数基线上都有较好的表现，这说明了集成方法的优越性。

与相对稳定的水质读数相比，PM2.5浓度波动较大，预测难度较大。图4是北京空气质量数据的综合比较。很容易看出，我们的模型同时达到了MAE和RMSE的最佳性能。继之前的工作关注MAE之后，我们主要讨论了这种度量。论文的方法比这些方法低了7.2%到63.5%，表明它在其他应用上有更好的泛化性能。另一个有趣的观察结果是，stMTMVL在水质预测方面效果很好，但在这方面表现出了劣势，因为空气质量预测的联合学习任务的数量远远大于水质预测的联合学习任务的数量。

## 5.总结
这篇论文提出了一种基于多层attention的时间序列预测网络。在第一个层次，应用local和global spatial attention机制来捕获地理感知数据中的动态传感器间关联。在第二层，论文利用temporal attention自适应地选择相关的时间步长进行预测。

此外，论文的模型考虑了外部因素的影响，使用通用的特征抽取模块。论文中使用在两类地理传感器的数据集上对论文的模型进行了评价，实验结果表明，论文的模型在和其他9个模型同时在RMSE和MAE两个指标获得了最佳的性能。

项目地址：[https://momodel.cn/workspace/5d96ed18673496721b0792fb?type=app](https://momodel.cn/workspace/5d96ed18673496721b0792fb?type=app)

## 6.参考资料
+ 论文：[GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction](https://www.ijcai.org/proceedings/2018/476)
+ 博客：[softmax详解](https://blog.csdn.net/bitcarmanlee/article/details/82320853)
+ 博客：[attention介绍](https://www.cnblogs.com/ydcode/p/11038064.html)
+ 博客：[Adam优化器](https://www.jianshu.com/p/aebcaf8af76e)

