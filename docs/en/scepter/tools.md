# Tool Components（Tools）

Supports querying components registered with the framework and obtaining parameter templates.

## Overview
1、Query module types：scepter.module_list
2、Query objects by module：scepter.objects_by_module
3、Query parameter configurations by object：scepter.configures_by_objects

<hr/>

## Basic Usage

```python
from scepter import module_list, objects_by_module, configures_by_objects

# Query module types
module_list()
# Query object list by module
objects_by_module("BACKBONES")
# Query parameter configurations by object
configures_by_objects("BACKBONES", "ResNet3D_TAda")
```
<hr/>

### <font color="#0FB0E4">function **module_list**</font>
()

**Returns**

- **list** —— A list of module names.

### <font color="#0FB0E4">function **objects_by_module**</font>
(module_name: str)

**Parameters**

- **module_name** —— Module name

**Returns**

- **list** —— A list of module names.

### <font color="#0FB0E4">function **get_module_object_config**</font>
(module_name: str, object_name: str)

**Parameters**

- **module_name** —— Module name
- **object_name** —— Object name

**Returns**

- **str** —— Parameter template.
