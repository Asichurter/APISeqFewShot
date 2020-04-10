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