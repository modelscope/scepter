<h1 align="center">Dataset Management</h1>

SCEPTER supports three types of dataset formats: TXT, CSV, and ModelScope.
Below are examples for each format, illustrating their details and basic usage.

## Modelscope Format

We use a [custom-stylized dataset](https://modelscope.cn/datasets/damo/style_custom_dataset/summary), which included classes 3D, anime, flat illustration, oil painting, sketch, and watercolor, each with 30 image-text pairs.

```python
# pip install modelscope
from modelscope.msdatasets import MsDataset
ms_train_dataset = MsDataset.load('style_custom_dataset', namespace='damo', subset_name='3D', split='train_short')
print(next(iter(ms_train_dataset)))
```

## CSV Format

For the data format used by SCEPTER Studio, please refer to [3D_example_csv.zip](https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip) and [hed_pair.zip](https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets%2Fhed_pair.zip).
```shell
mkdir -p cache/datasets/ && wget 'https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip' -O cache/datasets/3D_example_csv.zip && unzip cache/datasets/3D_example_csv.zip -d cache/datasets/ && rm cache/datasets/3D_example_csv.zip
mkdir -p cache/datasets/ && wget 'https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/hed_pair.zip' -O cache/datasets/hed_pair.zip && unzip cache/datasets/hed_pair.zip -d cache/datasets/ && rm cache/datasets/hed_pair.zip
```

## TXT Format

To facilitate starting training in command-line mode, you can use a dataset in text format, please refer to [3D_example_txt.zip](https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_txt.zip)

```shell
mkdir -p cache/datasets/ && wget 'https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_txt.zip' -O cache/datasets/3D_example_txt.zip && unzip cache/datasets/3D_example_txt.zip -d cache/datasets/ && rm cache/datasets/3D_example_txt.zip
```
