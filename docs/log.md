任务日志

# 4.8
- 测试NGram性能

- 测试了纯CNN结构和LSTM+CNN结构在NGram数据上（序列长度为100）的效果，结果
显示正确率大约在３４％左右徘徊，没有出现过拟合，损失值下降到1.4左右，且使用
CNN进行解码效果好于使用注意力

- 测试了在固定任务（给定任务seed）的情况下模型的效果，结果显示在固定任务上模型
能够拟合，但是出现过拟合现象。这说明并非模型不能捕获序列特征，而是无法捕获任务间
的元特征

- 固定embedding层的权重进行训练不能收敛

- 使用ProtoNet里面的Conv-4(2D)不能收敛

- Transformer使用CNN和自注意力归约都不能收敛

# 4.9 TODO

- 测试Transformer在本数据集上的fine-tuning（考虑使用原API名称） ×

# 4.9

- 在seq_len=50的数据集上，LSTM+CNN的表现比100更好，说明存在长序列噪声
的可能。但是短序列的过拟合现象更加明显，val_loss 保持在1.57左右的水平，
而train_loss已经下降到1.3左右的水平(1.png是CNN在先，2.png是CNN在LSTM后)

- 根据Mutiple Metric论文中指出的：可能任务间存在较大的差异(variation)，
导致common metric方法不能很好奏效

# 4.10 TODO

- 测试如ConvProtoNet, InductionNet和Hybrid-attention Net等其他模型


# 4.15

- 训练集没有很好收敛的原因也可能在于基于API的数据集进行跨任务学习的难度较高，
主要因为任务空间较大，采样比例小。于是测试减少meta-training的class数量，
检验训练正确率是否提升。结果显示，训练集类减少到20个时上升大概7-8%，减小到10
个类时正确率上升15%及以上，说明了该问题。

- 减少metatraining类数目的代价时元训练过程可能只在小任务空间中进行，从而
限制了模型的泛化能力。结果显示，验证正确率还是停留在35%-40%水平

# 4.19

- 使用任务Batch可以显著减小训练正确率的震荡，其他效果目前没有观测到

- MetaSGD的训练正确率稍高一些（embedding不adapt），但是验证正确
率更低

# 4.21 

- 将词嵌入维度提升至300，效果并无较大差别

# 4.22 TODO

- TCN时域卷积作为待选嵌入结构

- Hybrid Attention结构进行性能测试

- ATAML中的嵌入部分作为公用共享部分，时间步的注意力权重矩阵和softmax
分类器作为adapt对象，分离common和adapted参数

- 将支持集进一步划分为辅助支持集和辅助查询集，将支持集内部的辅助损失作为
损失的一部分

# 5.12

- ATAML在对序列每一个元素进行attend之后，将每一个state相加取平均的论文
中方法在复现时没有取得训练效果，模型训练停止；但是如果将模型的特征dim维度
相加取平均，则模型能够训练得动

- ATAML训练不动的谜题解开了：inner learning rate 过小导致outer loop
无法有效根据inner adapt 的结果进行训练

- 注意：从version=23之后，所有early stop从loss为criteria改为以
acc作为criteria

- 对于定制grad，直接torch.autograd.grad先求grad再遍历parameter
一个一个赋值: par.grad = g

- 测试一阶MAML(FOMAML)使用ATAML（分离task-specific和task-shared）
进行训练，并且使用自注意力规约来从BiLSTM中获取唯一表示

# 5.13

- PyTorch不支持RNN的高阶导数和RNN在eval阶段的一切反向传播，因此使用adapt
  方法时需要小心，至少Reptile不好做（因为所有参数更新都来自inner-loop的adapt，
  因此需要在测试阶段计算LSTM的导数来adapt，然而不支持）
  

# 5.14
- PreLayerATAML确实可以提高ATAML的性能，且学习到的层学习率逐层递减，符合
  实践中越后的update对应着越小的学习率。可以考虑相比于模型其他权重，逐层学习
  率参数的基础学习率更小一些（如5e-4）
  
- 提高adapt层数可以略微提高性能，也容易导致模型的不稳定和后期严重的过拟合


# 5.15 TODO

- Task Condition，基于TADAM论文中的思路，引入task prototype 来对固定
  嵌入引入一些task-specific信息以优化原share common embedding
  
  
# 5.15

- 直接使用TADAM中的task prototype方法，使用ResidualFC和post-multiplier
  创建TEN任务嵌入模块。但是原文中，从任务中生成的affine parameter作用在卷积后
  的每一个feature map上（BN后），相当于一个卷积核dense层。BiLSTM没有卷积核，
  因此选择作用在feature map级别上，即序列的step上。multiplier没有介绍清楚到
  底是一个标量还是一个与affine参数维度相同的向量，此处当做向量
  
  
# 5.16

- 测试LayerNorm效果：

    - 移除了Embedding之后，Encoding之前的LayerNorm后，模型收敛速度大幅下降，
      且最终正确率也下降些许
    
    - 在Encoder和点乘注意力之前加入一个LayerNorm导致模型不稳定，且收敛速度和正确率
    大大下降

- 多层LSTM测试结果：
    
    - 2层的LSTM收敛比1层慢得多
    
    - 2层的性能与1层的相比几乎没有上升,损失值相比之下要大一些

- 使用MSE+sigmoid激活的性能与nll+softmax几乎相当，甚至更好


# 5.17 TODO

- 改变Task-conditioning的结果，例如使用一个单独的task embedding来生成任务
  原型，代替原TEN中使用类中心作为任务原型使用相同嵌入的方法
  
  