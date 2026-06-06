# task [评价，该海报数据集质量不是很高；同时给出的mask，其实是视觉文本的mask]
写一个程序，从海报数据集中抽取10张图片
程序开头如下所示：
``` bash
from datasets import load_dataset

dataset = load_dataset("PosterCraft/Poster100K")
print(dataset)

```

dataset的结构如下所示：

``` bash
DatasetDict({
    train: Dataset({
        features: ['image', 'caption', 'mask_regions', 'file_name', 'folder_path', 'batch_name', 'normalized_path'],
        num_rows: 93033
    })
})
```

基于该结构，从该数据集中提取原始图片，captions，mask_regions，file_name，normalized_path；
原始图片和基于normalized_path获取到的图片分别存储在两个文件夹，叫做raw_images和normalized_images。
然后提取的captions，mask_regions，file_name存储成一个json文件：
诸如{
    "file_name”: {
        "caption": "xxx",
        "mask_regions": []
    }
}


# 使用豆包生成对应的描述 
请给出10个样例的 poster 描述，并且给出你认为的海报中的各个元素的合理的布局：
每个单独海报希望的格式：
{'prompt': 'a yellow chair and a blue airplane.', '0': {'description': 'yellow chair', 'mask': [63, 130, 250, 310]}, '1': {'description': 'blue airplane', 'mask': [270, 75, 480, 260]}}
10个海报的主题尽量都不一样


