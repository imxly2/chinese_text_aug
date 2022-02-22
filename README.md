# text-data-augmentation 
简单好用的文本数据增强

## EDA(Easy Data Augmentation) [参考论文](https://arxiv.org/abs/1901.11196v1)  
支持list和str的输入；用近义词词典代替syn库，用起来更方便。
```python
from tda import eda
s1 = '我是需要翻译的我需要超过10个词，一定要超过10个词'
s2 = ['我是需要翻译的我需要超过10个词，一定要超过10个词', 
      '我是需要翻译的我需要超过10个词，一定要超过10个词']
r1 = eda(s1)
r2 = eda(s2)
 ```

## 回译
用facebook的m2m100模型代替翻译api，缺点就是慢一点，用gpu能提速。
### requirements
- torch
- transformers

```python
from tda import back_translate
s1 = '我是需要增强的'
s2 = ['我是需要增强的', '我是需要翻译的']
r1 = back_translate(s1)
r2 = back_translate(s2, device='cuda:0')
 ```
## 安装方法  
pip install git+https://github.com/imxly2/chinese_text_aug.git