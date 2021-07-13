# unirep-and-resnet
这个模型主要是用来预测蛋白质氨基酸之间的残基接触
它由两部分构成，分别是unirep和resnet
### unirep
第一个部分是讲氨基酸序列编码成embedding
称之为unirep模型
unirep模型是mLSTM构成的rnn模型
通过预测下一个氨基酸的任务来训练模型
从而提取出氨基酸之间的结构信息

### resnet
第一个部分输出的embedding会经过一个二维转三维的变化
然后输入到第二个部分中
这一步便将氨基酸接触预测问题转换为像素级分类的问题
而第二个部分是resnet残差网络
它提取出氨基酸之间的两两相关的信息
从而预测氨基酸之间的接触

### 64 weight
该文件夹中存储着Unirep模型的预训练参数，可以预先加载或者从头训练


### 数据集
Unirep的预训练的数据集为uniref50数据集
unirep-resnet模型的训练集为pdb-25数据集
