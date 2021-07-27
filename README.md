### 遇到的问题

###### 问题1

```python
import torch
import pointnet2_cuda as pointnet2 # 输入这句话之前，必须先导入import torch，否则会报错

a = torch.ones(2,3)
a.cuda()

输出：RuntimeError: CUDA error: unknown error
```

###### 解决方法：

原本环境为pytorch1.2，将pytorch换成1.5就行了
https://github.com/sshaoshuai/Pointnet2.PyTorch/issues/19
