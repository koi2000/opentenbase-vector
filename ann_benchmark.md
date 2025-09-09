# ann benchmark测试结果

针对 OpenTenbase 和 postgres 进行了ann benchmark测试，选用的测试数据集为mnist-784-euclidean。本机仅部署了一个OptenBase节点，且数据量较小，仅用于测试向量功能是否可以正常运行。

下图为postgres的测试结果
![alt text](./images/pg_ann.png)

OpenTenbase的测试结果如下：
![alt text](./images/opentenbase_ann.png)