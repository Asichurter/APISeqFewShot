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
  
  
# 5.17

- TCN在大部分任务下并没有达到和BiLSTM相同的性能（特别是在adapt方法中），在
  metric-based方法中性能略微低于BiLSTM（ProtoNet，HAP），但是在adapt
  方法中性能大幅下降（ATAML， PerLayerATAML），而且训练过程震荡较大，对
  inner-loop学习率极度敏感，收敛后的性能也远低于BiLSTM

- 据观察，TCN占用的显存数量并不比BiLSTM小（原因可能是使用了residual使得
  深度增加）
  
- 由于使用了预训练的embedding，因此大部分模型在较少的几次迭代中很快到达了性能
  上限，此时较大的学习率容易导致过拟合。因此，考虑减小初始学习率
  
  
# 5.19

- FEAT

    - FEAT的contrastive loss部分几乎没有作用
    
    - 将query也adapt会起到反作用，应该只adapt支持集
    
    - Transformer性能逊于DeepSet
    
    - Post-Avg优于Pre-Avg，因为获得了更多的task信息（序列更长）
    
    - 增加LSTM的隐藏层维度和层数几乎没有性能提升
    
    - 减小Embedding部分的学习率几乎没有提升

- 验证集中，训练开始正确率就特别高，但是损失也特别高，原因在于少量预测错误的样本的
  预测置信度过高导致；随着训练进行，验证损失值下降，但是验证正确率提升很小，原因
  在于之前分类错误的样本的分类置信度下降，但是分类结果没有发生改变。最后经过观察，
  分类错误的验证集样本一般正确率标签的置信度排行第二，仅略微低于错误类

- 使用FastText——直接对序列嵌入按照序列维度取均值作为最终嵌入（该方法会丢失序列
  信息，仅保留统计信息）——结果表明，在FEAT模型上，FastText的训练正确率最高仅为
  80%，但是验证正确率却可以达到89%，略低于保留序列信息的91.7%。这说明验证集中
  的推断似乎大部分依赖于统计信息，很少依赖于序列信息


# 5.20 TODO

- 重新分配数据集进行测试


# 5.20

- 使用非NGram的长度为100的原始序列进行FEAT实验，结果发现过拟合非常严重，训练正确
  率99%，但是验证正确率只有77%左右（最高不超过80%）
  
# 5.22
- 改变数据集分割以后，测试集正确率下滑到85%水平，说明实验对数据集分割很敏感

- MatchNet使用逐个样本而不是原型进行分类，效果很差

- 使用deepset的FEAT虽然正确率相比ProtoNet有提升，但是经过实验发现，在adapt之
  后获得的prototypes的性能相比于adapt之前几乎没有提升


# 5.24

- LSTM只取最后一个状态作为提取输出效果很差，验证正确率不到80%（第二次分割）

- 很多模型的过拟合发生在学习率改变的时候(15000 epoch)，考虑取消学习率衰减
  或者延长衰减周期
  
- FEAT中使用的set function不是原生的Transformer Encoder，只是一个简单的
  多头注意力


# 5.25

- 序列长度增加可以缓解过拟合

- 训练类过少会导致严重过拟合

- 训练类内的样本过少是导致过拟合的原因之一

# 5.27

- 改进IMP——ImpIMP的问题在于，只要被正确分类，原型就不会因为支持集中的样本多峰分布
  而选择进行分裂，这会导致看起来明显是两个峰的分布被判定为不需要分割
  
- 原生IMP的问题在于，其分裂条件只取决于与最近原型的距离，因此即使因为多峰分布而
  被错误分类，只要周围有一个其他类的原型，其就可以选择不分裂，造成正确率下降(有误！！)
  
- 两个模型都存在的问题：为了让过程可微，在更新prototype的时候会考虑所有的样本
  （即使是类标签不匹配的样本或者相距很远的样本），计算所欲样本的分配系数，然后使用所有
  样本来更新原型 （与hard assignment相对的soft assignment，不强行让样本只属于
  一个类簇中）。这会导致生成的类原型在更新过程中因为其他不相关点的干扰远离较为理想的
  初始点
  
# 5.29 

- 继续IMP的问题：
    - 类簇的重新分配取决于同类的所有样本，即使他们不同簇，因此类簇的充分配
      会收到较远的同类样本的影响
      
    - 由于遍历样本时，每遇到一个样本就进行类簇生成，因此最终的模型会取决于遍历样本
      的顺序

# 5.30

- 在第一次分割的数据集上，IMP和ImpIMP没有ProtoNet表现的好，可能是因为非多峰分布的
  原因

- 在第二次分割上多峰分布表现得非常明显，此时长度为100的数据集的正确率已经非常接近
  长度为200的数据集了
  

# 6.2 

- LayerNorm非常吃显存(seq=200时要占据接近1G现存)

# 7.1

- LSTM cell 的最终性能表现好于LSTM,可能是mask的原因

# 7.9 

- 放弃virushare-45-rmsub,因为各个模型性能太过于接近(似乎移除重复子串的处理方式
  会使得大部分模型性能接近)
  
# 7.28


  
  