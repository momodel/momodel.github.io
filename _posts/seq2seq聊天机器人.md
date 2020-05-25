## seq2seq聊天机器人

作者：魏祖昌

## 一、背景介绍
人工智能技术的进步，语音识别技术、自然语言处理等技术的成熟，智能客服的发展很好的承接当下传统人工客服所面临的挑战。智能客服能够24小时在线为不同用户同时解决问题，工作效率高等特点，这是传统人工客服不能替代的，它能为公司节省大量的人工客服成本。在这次疫情当中，由于总总原因，大家肯定多多少少都见识过各种各样的智能客服。本文就基于seq2seq来介绍一个聊天机器人。


## 二、seq2seq
Seq2Seq即Sequence to Sequence，是一种时序对映射的过程，实现了深度学习模型在序列问题中的应用，其中比较突出的是自然语言中的运用。它主要由Encoder和Decoder两个部分组成，正如图一所示。

![](https://imgbed.momodel.cn/1589288342512-e05963a4-1c5b-4300-ab0a-8d04f9499991.png)

图一：论文中seq2seq结构（[https://arxiv.org/pdf/1409.3215.pdf](https://arxiv.org/pdf/1409.3215.pdf)）
seq2seq模型的encoder非常简单（上图中ABC对应的部分），就是RNN，可以是多层（GRU，LSTM）。decoder在训练和测试的时候，稍微有点不同。decoder训练的时候输入由两部分组成，一部分是encoder的last state，另一部分是target序列，如上图中的第一个<EOS> WXYZ;其中两个<EOS>表示的是序列开始符和结束符；decoder测试的时候输入也是由两部分组成，一部分是encoder的last state，另一部分是来自于上一个时刻的输出（上一个时刻的输出作为下一个时刻的输入），直到某个时刻的输出遇到结束符<EOS>为止。
但是seq2seq有一个很明显的缺点，即无论之前的encoder的context有多长，包含多少信息量，最终都要被压缩成一个固定维度的vector。这意味着context越大，decoder的输入之一的last state 会丢失越多的信息。这就会导致当输入自然语言长度越长，模型丢失的信息就越多，表现的就越差。
## 三、attention
为了解决seq2seq中压缩称固定长维度的vector，使得上一层给下一层的信息是符合一定分布的，即有一定的权值，使得更有价值的消息能够传递下去。
![](https://imgbed.momodel.cn/1589290981448-44ecacb4-5a70-4028-99cb-f80292fda540.png)

图二：attention机制原理（[https://arxiv.org/pdf/1409.0473)](https://arxiv.org/pdf/1409.0473))）

如上图所示，利用 $\overrightarrow{\alpha_{t}}$ 我们可以进行加权求和得到相应的context vector $\overrightarrow{c_{t}}=\sum_{j=1}^{T} \alpha_{t j} h_{j}$ 。这也是attention机制中的关键操作，计算encoder与decoder state之间的关联性的权重，得到Attention分布，从而对于当前输出位置得到比较重要的输入位置的权重，在预测输出时相应的会占较大的比重。

通过Attention机制的引入，我们打破了只能利用encoder最终单一向量结果的限制，从而使模型可以集中在所有对于下一个目标单词重要的输入信息上，使模型效果得到极大的改善。还有一个优点是，我们通过观察attention 权重矩阵的变化，可以更好地知道哪部分翻译对应哪部分源文字，有助于更好的理解模型工作机制，如下图三所示。
![](https://imgbed.momodel.cn/1589292444298-78af439d-53c8-4133-a248-9ba79564e281.png)

图三：attention权值变化图片

## 四、训练结果
该聊天机器人就是由seq2seq和attention机制共同构成。该模型设置了训练5000次训练次数，训练结果如图四所示。项目具体代码下方有链接。（注意需要keras 2.2.1版本才能跑起来）
![](https://imgbed.momodel.cn/1589293276534-9c42c8f7-5917-41d9-9111-d3df695dd342.png)

图四：训练结果（train.py）
在训练完成之后，会把训练模型存放在model文件夹下，然后我们运行chat_robot.py，我们就能看见图五聊天机器人的结果了。如果还想添加更多的聊天语料的话，请将question.txt和answer.txt放在corpus文件夹中，在重新进行模型训练即可。

![](https://imgbed.momodel.cn/1589293553683-051648e9-03b7-45e1-9397-0cb3f244b3a4.png)
图五：聊天机器人截图

还可以通过word2vec_plot.py看到word2vec向量分布图，效果如图六所示。

![](https://imgbed.momodel.cn/1589294612064-f28e09f8-8cf8-4a5a-ac08-d6dc9c7a21a8.png)
图六：word2vec向量分布图

**项目地址：**[https://momodel.cn/workspace/5eba52f8d99e51afef3bfea2?type=app](https://momodel.cn/workspace/5eba52f8d99e51afef3bfea2?type=app)**（推荐在电脑端使用Google Chrome浏览器进行打开）**


## 引用

1. seq2seq：[https://blog.csdn.net/rxm1989/article/details/79459739](https://blog.csdn.net/rxm1989/article/details/79459739)
1. attention：[https://zhuanlan.zhihu.com/p/47063917](https://zhuanlan.zhihu.com/p/47063917)
1. 代码来源：[https://github.com/EuphoriaYan/ChatRobot-For-Keras2](https://github.com/EuphoriaYan/ChatRobot-For-Keras2)

## 关于我们
[Mo](https://momodel.cn)（网址：https://momodel.cn） 是一个支持 Python的人工智能在线建模平台，能帮助你快速开发、训练并部署模型。

近期 [Mo](https://momodel.cn) 也在持续进行机器学习相关的入门课程和论文分享活动，欢迎大家关注我们的公众号获取最新资讯！

![](https://imgbed.momodel.cn/联系人.png)

