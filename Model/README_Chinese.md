### Model

Model类定义了多种标准化以及结构化的模型训练框架，可以直接调用拟合。



### DataSetConstructor

DataSetConstructor类定义了生成训练集X和Y的函数



#### 开发日志

##### 2021-09-04

-- 定义了DataSetConstructor类，改成标准化的Data类输入

##### 2021-09-10

-- 新增：使用网格搜索方法对因子搜索系数，以优化指定区间内得分最高的n个股票的平均收益

##### 2021-09-11

-- 新增：lightgbm方法，以及Lasso和lightgbm做boosting

##### 2021-11-09

-- 新增：多种NN模型，LSTM模型，多种loss函数

-- 新增：针对LSTM模型的时序训练数据构造

##### 2021-11-18

-- 更新：不同模型共享父类，减少重复代码量

-- 更新：损失函数单独封装，提高代码可读性；新增PolyLoss，WeightedPolyLoss和HingePolyLoss

-- 更新：DataSetConstructor可以选择输出y的形式

##### 2021-11-22

-- 更新：新增HuberLoss和QuantileLoss

##### 2021-12-17

-- 更新：新增factor_selector，进行因子筛选
