# 工具组件（Tools）

支持对框架注册的组件进行查询、获取参数模版。

## 总览
1、查询模块类型：scepter.module_list
2、按模块查询对象：scepter.objects_by_module
3、按对象查询参数配置：scepter.configures_by_objects

<hr/>

## 基础用法

```python
from scepter import module_list, objects_by_module, configures_by_objects

# 查询模块类型
module_list()
# 按照模块查询对象列表
objects_by_module("BACKBONES")
# 按照对象查询参数配置
configures_by_objects("BACKBONES", "ResNet3D_TAda")
```
<hr/>

### <font color="#0FB0E4">function **module_list**</font>
()

**Returns**

- **list** —— 模块名列表。

### <font color="#0FB0E4">function **objects_by_module**</font>
(module_name: str)

**Parameters**

- **module_name** —— 模块名

**Returns**

- **list** —— 模块名列表。

### <font color="#0FB0E4">function **get_module_object_config**</font>
(module_name: str, object_name: str)

**Parameters**

- **module_name** —— 模块名
- **object_name** —— 对象名

**Returns**

- **str** —— 参数模版。
