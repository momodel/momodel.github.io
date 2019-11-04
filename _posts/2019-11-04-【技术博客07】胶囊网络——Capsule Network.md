---
layout: post
title: 【技术博客07】胶囊网络——Capsule Network
date: 2019-11-04 12:00
---

作者：林泽龙


# 1. 背景介绍
CNN 在处理图像分类问题上表现非常出色，已经完成了很多不可思议的任务，并且在一些项目上超过了人类，对整个机器学习的领域产生了重大的影响。而 CNN 的本质由大量的向量和矩阵的相乘或者相加，因此神经网络的计算消耗非常大，所以将一张图片上全部像素信息传递到下一层运算是十分困难的，所以出现了“卷积”和“池化”这种方法，能够在不损失数据本质的情况下帮我们简化神经网络的计算。
诚然， CNN 在分类和数据集非常接近的图像时，实验的效果非常好，但如果图像存在翻转、倾斜或任何其它方向性问题时，卷积神经网络的表现就比较糟糕了。通过在训练期间为同一图像翻转和平移可以解决这个问题（数据增强）。而问题的本质是网络中的滤波器以比较精细的级别上理解图像。举一个简单的例子，对于一张人脸而言，它的组成部分包括面部轮廓，两个眼睛，一个鼻子和一张嘴巴。对于CNN而言，这些部分就足以识别一张人脸。然而，这些组成部分的相对位置以及朝向就没有那么重要。
这些主要的原因是人类在识别图像的时候，是遵照树形的方式，由上而下展开式的，而CNN则是通过一层层的过滤，将信息一步步由下而上的进行抽象。这是胶囊网络作者认为的人与CNN神经网络的最大区别。

本文基于一篇在2017年由 Hinton 等人发表的一篇文章[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)，该文章介绍了一种神经网络可以弥补 CNN 无法处理图片位置方向等缺点。相比于 CNN 该网络更接近于人类的图像识别原理。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWdiZWQubW9tb2RlbC5jbi8yMDE5MTkwMzIzMzItVi5wbmc?x-oss-process=image/format,png)

**图 1**：人类识别图像与 CNN 的区别

# 2. 胶囊网络是如何工作的
胶囊网络和传统的人工神经网络最根本的区别在于网络的单元结构。对于传统神经网络而言，神经元的计算可分为以下三个步骤：
1. 对输入进行标量加权计算。
2. 对加权后的输入标量进行求和。
3. 标量到标量的非线性化。

而对于胶囊而言，它的计算分以下四个步骤：
1. 对输入向量做乘法，其中 $v_1$ 和 $v_2$ 分别来自与前面的 capsule 的输出，在单个 capsule 内部，对  $v_1$ 和 $v_2$  分别乘上  $w_1$ 和 $w_2$  得到了 新的  $u_1$ 和 $u_2$  。

2. 对输入向量进行标量加权，令$u_1$与$c_1$相乘，$u_2$与$c_2$相乘，其中$c_1$和$c_2$均为标量，且$c_1 + C_2 = 1$。
3. 对得到向量求和，得到$s=c_1 u_1 +c_2u_2$。
4. 向量到向量的非线性化，将得到的结果向量 $s$ 进行转换，即通过函数 $Squash(s)= \cfrac{||s||^2}{1+||s||^2} \cfrac{s}{||s||}$得到结果 $s$，作为这个capsule 的输出，且这个结果 $v$ 可以作为下一个 capsule 的输入。


![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjY4MTYzNzQtMjc1ZGY1MmItMmU3My00ZjBkLWE3MDctMzg1OGMyNTdkYTg1LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=1868&originHeight=1868&originWidth=2574&search=&size=0&status=done&width=2574)**图 2**：单个胶囊的运算方式

# 3. 胶囊网络的细节
## 3.1 胶囊网络的动态寻路算法
上一章我们了解胶囊网络工作的总体方式，这一章我们来关注胶囊内部的参数是如何进行更新的，首先我们来看论文里关于此算法的伪代码介绍：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjY5MjgyMjAtZTUzMDhhMDYtOTMwMy00YTMwLWFjNjgtNGYyYTU5MDk3M2NlLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=432&originHeight=432&originWidth=1562&search=&size=0&status=done&width=1562)**图 3**：单个胶囊内的参数更新

第一行代码：所有L层的胶囊和输入 $u_i$ ，还有迭代次数 $r$ 其中 $u_i$ 为输入向量 $u_i$ 与权重 $w_i$ 相乘得到的结果。第二行代码：开辟了新的一组临时变量 $b_{ij}$ ，每一个标量 $b_{ij}$ 初始值为0，且与 $c_{ij}$ 一一对应，在迭代结束后，这个变量将被存储在 $c_{ij}$ 中。第三行代码：我们要设定一个 iterations 即内部的迭代次数（超参数），然后开始迭代。第四行代码：首先令所有的 $c=softmax(b)$ ，因此确保了所有的 $ci$ 和为1且为非负值，第一次所有的 $c$ 值相同。第五行代码：令 $s_i=u_ic_i$，这一步只是简单的运算。第六行代码：我们将上一步的结果传入 Squash 函数（第二章中已介绍）做非线性化，从而确保向量的方向是保持不变的，并把所有输入向量都集合到一起，且将向量的数值量级保持与之前胶囊一致。第七行代码：我们将更新向量 $b_{ij}$ 。这是动态寻路算法的最关键一步。它将高层胶囊的输出和低层胶囊的输出做点积，从而扩大那些和高层胶囊输出方向一致的向量。

下图为 iterations = 3 时的算法图解：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjY5NTAxMjktZDg5Yzg5NjgtNmRlZC00NmI1LTlhOGYtYTkwNjJkZTQyYjAyLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=1856&originHeight=1856&originWidth=2722&search=&size=0&status=done&width=2722)

**图 4**：迭代次数等于三时动态寻路算法

如上图所示 $b_0^1$ 和 $b_0^2 $ 为0，因此 $c_1^1$ 和 $c_1^2$ 的值为1/2，$s^1=u_1c_1^1+u_2c_2^2$ ，$a^1 = squash(s^1)$，然后新一轮的 $b_i^r=b_i^{r-1}+a^ru^i$ 结果作为下一轮 $c_j^i$的值，迭代次数过完后将结果 $a^3$ 作为输出 $v$ 。

## 3.2 动态寻路算法直观上的理解
其中两个高层胶囊的输出用紫色向量 $v_1 v_2$ 表示，橙色向量表示接受自某个低层胶囊的输入，其他黑色向量表示接受其他低层胶囊的输入。左边的紫色输出 $v_1$ 和橙色输入 $u_1$ 指向相反的方向，所以它们并不相似，这意味着它们点积是负数，更新路由系数的时候将会减少$c_1^1$。右边的紫色输出 $v_2$ 和橙色输入 $u_2$ 指向相同方向，它们是相似的，因此更新参数的时候路由系数 $c_1^2$ 会增加。在所有高层胶囊及其所有输入上重复应用该过程，得到一个路由参数集合，达到来自低层胶囊的输出和高层胶囊输出的最佳匹配。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjY5ODg1MjUtYjhmNDNmNGQtOWU2Ny00YjdkLWJmNGEtZTRlMGY1ZmJmYWExLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=512&originHeight=512&originWidth=600&search=&size=0&status=done&width=600)

**图 5**：寻路算法直观上的理解


# 4. 论文中的网络结构
## 4. 1 训练的网络结构
论文用 minst 的数据集上的 CapsNet 架构如下图所示。架构可以简单的表示为仅有两个卷积层和一个全连接层组成。Conv1有256个9×9个卷积核，步长为1和ReLU激活。该层将像素的强度转换为之后会用于作为基本胶囊输入的局部特征探测器的激活。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjcxNjM4MzAtZjA5ODZhNmEtODQ3MC00MmM1LWI0OWQtZDA2MjIzNTU5YWRjLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=478&originHeight=478&originWidth=1632&search=&size=0&status=done&width=1632)

**图 6**：训练的网络结构

第一层为卷积层：输入：28×28图像（单色），输出：20×20×256张量，卷积核：256个步长为1的9×9×1的核，激活函数：ReLU。       
第二层为 primarycaps 层：输入：20×20×256张量，输出：6×6×8×32张量（共有32个胶囊），卷积核：8个步长为2的9×9×256的核/胶囊。       
第三层为 DigitCaps 层：输入：6×6×8×32张量（每个胶囊输出的是8维的向量），输出：16×10 矩阵（10个胶囊）。      

## 4. 2 重构的网络结构
重构器从正确的 DigitCap 中选择一个16维向量，并学习将其编码为数字图像（注意，训练时候只采用正确的 DigitCap 向量，而忽略不正确的 DigitCap ）。它接受正确的DigitCap的输出作为输入，重建一张28×28像素的图像，损失函数为重建图像和输入图像之间的欧式距离。解码器强制胶囊学习对重建原始图像有用的特征，重建图像越接近输入图像越好，下面展示重构的网络结构（最终的输出28*28）和重建图像的例子（l,p,r对应了标签，预测，重构目标）。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjczMDA0OTMtYTAyMDI0NWMtZTZlOS00MGEzLTg2MmYtNGZlMjAyMTQzNzBkLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=378&originHeight=378&originWidth=1092&search=&size=0&status=done&width=1092)**图 7**：重构的网络结构![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvNDQ5MTI5LzE1NzEzNjczNDEyMzUtOTVhYTA4Y2UtYjg0NC00ZTc3LWJiZWUtM2I1MzEyZjBkZGEzLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=386&originHeight=386&originWidth=1434&search=&size=0&status=done&width=1434)**图 8**：初始图像和重构图像
# 5. 小结
胶囊网络在如今卷积网络难以提升的阶段可以算是一种革命性的网络架构，神经元的输出从一个标量变成了一组向量，这如同让网络流动起来了。每一个识别子结构的胶囊，使原始图形中的细节高度的保真，而这个保真，又是从复杂结构里直接进化得到的。通过重构原图，模型做到了在视角变换后，通过网络结构的改变，给出相同的结果。另外需要指出的是，CNN和胶囊神经网络并不是互斥的，网络的底层也可以是卷积式的，毕竟胶囊网络善于的是在已抽象信息中用更少的比特做到高保真的重述。

**胶囊网络 demo 地址：[https://momodel.cn/workspace/5da92fb8ce9f60807bbe6d33/app](https://momodel.cn/workspace/5da92fb8ce9f60807bbe6d33/app)**

# 6. 参考资料
论文地址：[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
博客园：[CapsNet胶囊网络（理解）](https://www.cnblogs.com/CZiFan/p/9803067.html)
github：[A Keras implementation of CapsNet](https://github.com/XifengGuo/CapsNet-Keras)
github：[CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)

## 关于我们
**Mo**（网址：[**momodel.cn**](https://momodel.cn/)）是一个支持 Python 的**人工智能在线建模平台**，能帮助你快速开发、训练并部署模型。

---

**Mo 人工智能俱乐部** 是由网站的研发与产品设计团队发起、致力于降低人工智能开发与使用门槛的俱乐部。团队具备大数据处理分析、可视化与数据建模经验，已承担多领域智能项目，具备从底层到前端的全线设计开发能力。主要研究方向为大数据管理分析与人工智能技术，并以此来促进数据驱动的科学研究。

目前俱乐部每周六在杭州举办以机器学习为主题的线下技术沙龙活动，不定期进行论文分享与学术交流。希望能汇聚来自各行各业对人工智能感兴趣的朋友，不断交流共同成长，推动人工智能民主化、应用普及化。
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMzA3Nzk0LzE1NjA1NjU1NjQ5MzYtYmNkOWVjMWUtOGU0Ny00MzczLWJhMGQtMWFiM2E2OTZhZWU0LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=175&name=image.png&originHeight=349&originWidth=720&search=&size=170790&status=done&width=360)
