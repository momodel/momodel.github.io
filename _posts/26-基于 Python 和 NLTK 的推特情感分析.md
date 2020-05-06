# 26-基于 Python 和 NLTK 的推特情感分析

作者：宋彤彤


# 1. 导读
NLTK 是 Python 的一个自然语言处理模块，其中实现了朴素贝叶斯分类算法。这次 Mo 来教大家如何通过 python 和 nltk 模块实现对推文按照正面情绪（positive）和负面情绪（negative）进行归类。

在项目内部有可运行的代码教程 `naive_code.ipynb` 和 经过整理方便进行部署的部署文件 `Deploy.ipynb`，大家可以结合之前发布的[ Mo平台部署介绍 ](https://www.yuque.com/xxs3cf/gxy1e7/lvv4io)一文学习如何部署属于自己的应用。大家也可以打开下方项目地址，对部署好的应用进行一下测试，比如简单输入 ‘My house is great.’ 和 ‘My house is not great.’ 来判断它们分别是 positve 还是 nefative。

本文的内容只需要熟悉 Python 即可，快跟着小Mo一起学习吧。
**项目地址：**[https://momodel.cn/explore/5eacf3097f8b5371a8480403?type=app](https://momodel.cn/explore/5eacf3097f8b5371a8480403?type=app)



# 2. 准备工作

## 2.1 导入工具包
首先，导入我们用到的工具包 nltk。
```python
import nltk
# 如果没有这个包，可以根据下面的代码进行操作
# pip install nltk
# import nltk
# nltk.download() # 对依赖资源进行下载，一般下载 nltk.download('popular') 即可
```
## 2.2 准备数据
训练模型需要大量的标记数据才能有比较好的效果。这里我们先用少量的数据来帮助我们了解整个的流程和原理，如果需要更好的实验结果，可以加大训练数据的数量。
因为该模型是一个二分类模型，我们需要两类数据，分别标记为 'positive' 和 'negative'。初步训练好的模型需要测试数据来检验效果。
```python
# 标记为 positive 的数据
pos_tweets = [('I love this car', 'positive'),
             ('This view is amazing', 'positive'),
             ('I feel great this morning', 'positive'),
             ('I am so excited about the concert', 'positive'),
             ('He is my best friend', 'positive')]
# 标记为 negative 的数据
neg_tweets = [('I do not like this car', 'negative'),
             ('This view is horrible', 'negative'),
             ('I feel tired this morning', 'negative'),
             ('I am not looking forward to the concert', 'negative'),
             ('He is my enemy', 'negative')]
#测试数据,备用
test_tweets = [('I feel happy this morning', 'positive'),
              ('Larry is my friend', 'positive'),
              ('I do not like that man', 'negative'),
              ('My house is not great', 'negative'),
              ('Your song is annoying', 'negative')]
```
## 2.3 特征提取
我们需要从训练数据中提取有效的特征对模型进行训练。这里的特征是标签即其对应的推特中的有效单词。那么，怎么提取这些有效单词呢？
首先，分词并将所有单词变成小写，取长度大于 2 的单词，得到的列表代表一条 tweet；然后，将训练数据所有 tweet 包含的单词进行整合。
```python
# 数据整合及划分成词，删除长度小于2的单词
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))
print(tweets)

# 提取训练数据中所有单词，单词特征列表从推特内容中提取出的单词来表示
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words
words_in_tweets = get_words_in_tweets(tweets)
print(words_in_tweets)
```
为了训练分类器，我们需要一个统一的特征，那就是是否包含我们词库中的单词，下面的特征提取器可以对输入的 tweet 单词列表进行特征提取。
```python
# 对一条 tweet 提取特征，得到的字典表示 tweet 包含哪几个单词
def extract_features(document):
    document_words = set(document)
    features = { }
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
print(extract_features(['love', 'this', 'car']))
```
## 2.4 制作训练集并训练分类器
利用 nltk 的 classify 模块的 apply_features 方法制作训练集。
```python
# 利用 apply_features 方法制作训练集
training_set = nltk.classify.apply_features(extract_features, tweets)
print(training_set)
```
训练朴素贝叶斯分类器。
```python
# 训练朴素贝叶斯分类器
classifier = nltk.NaiveBayesClassifier.train(training_set)
```
到此，我们的分类器初步训练完成，可以使用了。
# 3. 测试工作
我们的分类器效果如何呢？先用我们事先准备好的测试集检验一下。可以得到 0.8 的正确率。
```python
count = 0
for (tweet, sentiment) in test_tweets:
    if classifier.classify(extract_features(tweet.split())) == sentiment:
        print('Yes, it is '+sentiment+' - '+tweet)
        count = count + 1
    else:
        print('No, it is '+sentiment+' - '+tweet)
rate = count/len(test_tweets)
print('Our correct rate is:', rate)
```
![](https://imgbed.momodel.cn/1588474472614-5b345bf7-18b2-45dd-8e93-d651b49229b2.png)

关于 'Your song is annoying' 这一句分类错误的原因，是我们的词库里没有关于 'annoying' 一词的任何信息。这也说明了数据集的重要性。

# 4. 分析总结

1. 分类器的 _label_probdist 是标签的先验概率。 在我们的例子中，标记为 positive 和 negtive 标签的概率都是 0.5。
1. 分类器的 _feature_probdist 是 特征/值概率词典。它与 _label_probdist 一起用于创建分类器。特征/值概率词典将预期似然估计与特征和标签相关联。我们可以看到，当输入包含 'best' 一词时，输入值被标记为 negative 的概率为 0.833。
1. 我们可以通过 show_most_informative_features() 方法来显示分类器中最有信息价值的特征。我们可以看到，如果输入中不包含 'not'，那么标记为 positive 的可能性是 negative 的 1.6 倍；不包含 'best'，标记为 negative 的可能性是 positive 的1.2倍。
```python
print(classifier._label_probdist.prob('positive'))
print(classifier._label_probdist.prob('negative'))
print(classifier._feature_probdist)
print(classifier._feature_probdist[('negative', 'contains(best)')].prob(True))
print(classifier.show_most_informative_features())
```
更多详细内容，请大家进入项目：[https://momodel.cn/explore/5eacf3097f8b5371a8480403?type=app](https://momodel.cn/explore/5eacf3097f8b5371a8480403?type=app) Fork 到你的 工作台 进行实际操作和学习。

# 5. 参考资料

1. 学习资料：[http://www.nltk.org/book/](http://www.nltk.org/book/)
1. 参考博客：[http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/](http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)



## 关于我们
[Mo](https://momodel.cn)（网址：https://momodel.cn） 是一个支持 Python的人工智能在线建模平台，能帮助你快速开发、训练并部署模型。

近期 [Mo](https://momodel.cn) 也在持续进行机器学习相关的入门课程和论文分享活动，欢迎大家关注我们的公众号获取最新资讯！

![](https://imgbed.momodel.cn/联系人.png)