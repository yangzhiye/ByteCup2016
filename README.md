# ByteCup2016
##数据描述
ByteCup2016数据挖掘竞赛,问题推荐

专家表：专家ID，专家Label，专家描述(词),专家描述(字)  

问题表：问题ID，问题Label(不对应于专家Label)，问题描述(词),问题描述(字),点赞数，回答数，精品回答数  

训练集：问题ID，专家ID，专家是否回答  

测试集/验证集：问题ID，专家ID  
##结果描述
NDCG结果:0.49

最高NDCG结果:0.53
##主要特征:

###问题特征:
*这类问题被回答的概率 

*这个问题被回答的概率

*点赞数(归一化)

*回答数(归一化)

*精品回答数(归一化)

###专家特征:
*这个专家回答以往被推送问题的概率

*这个专家的标签数

###问题与专家之间联系特征:
*word2vec取mean之后得到句子表示之后计算距离

*word2vec取max之后得到句子表示之后计算距离

*LDA计算句子相似度。

*离散化的label特征

##尝试过的深度学习模型(最优结果0.49)
###结果:0.49
![](https://github.com/yangzhiye/ImageCache/blob/master/ByteCup2016/%20dp1.png?raw=true)
###结果:0.46
![](https://github.com/yangzhiye/ImageCache/blob/master/ByteCup2016/dp2.png?raw=true)
###结果:0.42
![](https://github.com/yangzhiye/ImageCache/blob/master/ByteCup2016/dp3.png?raw=true)
###结果:0.49
![](https://github.com/yangzhiye/ImageCache/blob/master/ByteCup2016/dp4.png?raw=true)

##备注
###特征提取过程参见:create_features.py,其实还提了很多特征，不过都不work。
###word2vec调参结果:HS+Skipgram+1minwords+10windows+50维