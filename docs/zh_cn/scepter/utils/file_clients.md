# 文件系统（File System）

文件系统模块

## 总览

支持3类文件IO Handler:

1. scepter.modules.utils.file_clients.AliyunOssFs
2. scepter.modules.utils.file_clients.LocalFs
3. scepter.modules.utils.file_clients.HttpFs
4. scepter.modules.utils.file_clients.ModelscopeFs


<hr/>

## 基础用法

```python
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.config import Config

fs_cfg = Config(load=False, cfg_dict={
    "NAME": "AliyunOssFs",
    # ENDPOINT DESCRIPTION: the oss endpoint TYPE: str default: ''
    "ENDPOINT": "xxxxx",
    # BUCKET DESCRIPTION: the oss bucket TYPE: str default: ''
    "BUCKET": "xxxxx",
    # OSS_AK DESCRIPTION: the oss ak TYPE: str default: ''
    "OSS_AK": "xxxxx",
    # OSS_SK DESCRIPTION: the oss sk TYPE: str default: ''
    "OSS_SK": "xxxxx",
    # TEMP_DIR DESCRIPTION: default is None, means using system cache dir and auto remove! If you set dir, the data will be saved in this temp dir without autoremoving default. TYPE: NoneType default: None
    "TEMP_DIR": "cache",
    # AUTO_CLEAN DESCRIPTION: when TEMP_DIR is not None, if you set AUTO_CLEAN to True, the data will be clean automatics. TYPE: bool default: False
    "AUTO_CLEAN": False
})

fs_prefix = FS.init_fs_client(fs_cfg, logger=None)
with FS.get_from("xxxxx", wait_finish=True) as local_object:
# do sth. using local_object here.
# 一次下载多个文件
generator = FS.get_batch_objects_from(["xxx", "xxxx"])
for local_path in generator:
    print(local_path)
```
<hr/>

## **scepter.modules.utils.file_system.FileSystem**

通过build多类File IO Handler，支持不同类型文件的读写操作。
<br>

### <font color="#0FB0E4">function **\_\_init\_\_**</font>

()

**Parameters**

<br>

### <font color="#0FB0E4">function **init_fs_client**</font>

( cfg: *scepter.modules.utils.config.Config* = None, logger = None ) -> str

通过cfg参数来实例化fs_client，存储在self._prefix_to_clients属性中，可通过prefix来access对应的fs_client.

**Parameters**

- **cfg** —— The Config used to build fs_client. If None, use LocalFs as default.
- **logger** —— Instantiated Logger to print or save log.

**Returns**

- *str* —— The prefix of instantiated fs_client

<br>

### <font color="#0FB0E4">function **get_fs_client**</font>

( target_path: *str*, safe: *bool* = False )

通过target_path的前缀来获取对应的fs_client

**Parameters**

- **target_path** —— 目标文件路径
- **safe** —— 安全模式，返回client的copy，否则返回client本身

**Returns**

- *BaseFs* —— 实例化的fs_client

<br>

### <font color="#0FB0E4">function **get_from**</font>

( target_path: *str*, local_path: *str* = None, wait_finish: *bool* = False ) -> str

将远程文件下载到本地

**Parameters**

- **target_path** —— 远程文件路径
- **local_path** —— 本地文件路径，如果为None则使用cache路径
- **wait_finish** —— if True，则只有每台机器的0卡下载数据，其他卡等待0卡下载结束

**Returns**

- *str* —— 本地保存文件路径

<br>

### <font color="#0FB0E4">function **get_dir_to_local_dir**</font>

( target_path: *str*, local_path: *str* = None, wait_finish: *bool* = False, timeout: *int* = 3600, worker_id: *int* = 0 ) -> str

将远程路径的文件夹下载到本地

**Parameters**

- **target_path** —— 远程文件夹路径
- **local_path** —— 本地文件夹路径，如果为None则使用cache路径
- **wait_finish** —— if True，则只有每台机器的0卡下载数据，其他卡等待0卡下载结束
- **timeout** —— 下载超时时间
- **worker_id** —— Deprecated

**Returns**

- *str* —— 本地文件夹路径

<br>

### <font color="#0FB0E4">function **get_object**</font>

( target_path: *str* ) -> byte

读取远程文件到内存中

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *bytes* —— 目标文件的二进制数据

<br>

### <font color="#0FB0E4">function **get_object**</font>

( target_path: *str* ) -> byte

读取远程文件到内存中

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *bytes* —— 目标文件的二进制数据

<br>

### <font color="#0FB0E4">function **put_object**</font>

( local_data: *byte*, target_path: *str* ) -> bool

上传数据流到指定文件

**Parameters**

- **local_data** —— 本地数据流
- **target_path** —— 目标文件路径

**Returns**

- *bool* —— 是否上传成功

<br>

### <font color="#0FB0E4">function **delete_object**</font>

( target_path: *str* ) -> bool

删除目标文件

**Parameters**

- **target_path** ——

**Returns**

- *bool* —— 是否删除成功

<br>

### <font color="#0FB0E4">function **get_batch_objects_from**</font>

( target_path_list: *str*, wait_finish: *bool* ) -> *str*

批量下载文件

**Parameters**

- **target_path_list** —— 下载文件列表

**Returns**

- *local_path* —— 本地文件generator

<br>

### <font color="#0FB0E4">function **put_batch_objects_to**</font>

(local_path_list: *str*, target_path_list: *str*, wait_finish: *bool* ) -> *str*

批量上传文件

**Parameters**

- **local_path_list** —— 上传文件列表
- **target_path_list** —— 目标文件列表

**Returns**

- *local_path, target_path* —— 返回本地文件和目标文件的pair。

<br>

### <font color="#0FB0E4">function **get_object_stream**</font>

(target_path: *str*, start: *int*, size: *int*, delimiter: *str*) -> *byte*, *int*

批量上传文件

**Parameters**

- **target_path** —— 目标文件
- **start** —— 目标流的开始字符位置
- **size** —— 目标流的开始字符流大小
- **delimiter** —— 目标流的结束字符

**Returns**

- *local_data, end* —— 返回数据流字节和结束字符位置。

<br>

### <font color="#0FB0E4">function **get_object_chunk_list**</font>

( target_path: *str*, chunk_num: *int* = 1, delimiter: *str* = None ) -> list[bytes]

获取远程文件且分块

**Parameters**

- **target_path** —— 目标文件路径
- **chunk_num** —— 分块个数
- **delimiter** —— 分隔符，确保下载数据是完整的一条，不会从中间截断

**Returns**

- *list[bytes]* —— 分块数据

<br>

### <font color="#0FB0E4">function **get_url**</font>

( target_path: *str*, set_public = False, lifecycle: *int* = 360000 ) -> str

获取远程文件的url（仅支持AliyunOssFs）

**Parameters**

- **target_path** —— 目标文件路径
- **lifecycle** —— 有效时间
- **set_public** —— 反馈公开链接

**Returns**

- *str* —— 目标文件的url

<br>

### <font color="#0FB0E4">function **put_to**</font>

( target_path: *str* )

支持将本地文件上传到远程

**Parameters**

- **target_path** —— 远程文件路径

**Returns**

- **None**

```python
# 作为上下文管理器使用
with FS.put_to(target_path) as local_path:
    # some operations on local_path.
```

<br>

### <font color="#0FB0E4">function **put_object_from_local_file**</font>

( local_path: *str*, target_path: *str* ) -> bool

将本地文件push到远程路径

**Parameters**

- **local_path** —— 本地文件路径
- **target_path** —— 远程文件路径

**Returns**

- *bool* —— 是否上传成功

<br>

### <font color="#0FB0E4">function **put_dir_from_local_dir**</font>

( local_dir: *str*, target_dir: *str* ) -> bool

将本地文件夹push到远程路径

**Parameters**

- **local_dir** —— 本地文件夹路径
- **target_dir** —— 远程文件夹路径

**Returns**

- *bool* —— 是否上传成功

<br>

### <font color="#0FB0E4">function **add_target_local_map**</font>

( target_dir: *str*, local_dir: *str* ) -> None

将远程文件夹和本地文件夹路径的映射关系以key-value对的形式保存到self._target_local_mapper

**Parameters**

- **target_dir** —— 远程文件夹路径
- **local_dir** —— 本地文件夹路径

**Returns**

- *None*

<br>

### <font color="#0FB0E4">function **make_dir**</font>

( target_dir: *str* ) -> bool

创建远程文件夹

**Parameters**

- **target_dir** —— 远程文件夹路径

**Returns**

- *bool* —— 是否创建成功

<br>

### <font color="#0FB0E4">function **exists**</font>

( target_path: *str* ) -> bool

判断目标路径是否存在

**Parameters**

- **target_path** —— 远程文件路径

**Returns**

- *bool* —— 是否存在

<br>

### <font color="#0FB0E4">function **map_to_local**</font>

( target_path: *str* ) -> str, bool

将远程文件路径映射到本地路径

**Parameters**

- **target_path** —— 远程文件路径

**Returns**

- *str* —— 本地文件路径
- *bool* —— 本地文件是否tmp文件

<br>

### <font color="#0FB0E4">function **walk_dir**</font>

( target_dir: *str*, recurse = True) -> Iterator

获取远程文件夹下的文件列表

**Parameters**

- **target_dir** —— 远程文件夹路径
- **recurse** —— 是否遍历子文件夹，默认遍历为True

**Returns**

- *Iterator* —— 子文件路径列表

<br>

### <font color="#0FB0E4">function **is_local_client**</font>

( target_path: *str* ) -> bool

判断目标文件client是不是LocalFs

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *bool* —— client是否LocalFs

<br>

### <font color="#0FB0E4">function **size**</font>

( target_path: *str* ) -> int

判断目标文件大小

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *int* —— 目标文件size

<br>

### <font color="#0FB0E4">function **isfile**</font>

( target_path: *str* ) -> bool

判断目标路径是不是object

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *bool* —— 目标路径是不是object

<br>

### <font color="#0FB0E4">function **isdir**</font>

( target_path: *str* ) -> bool

判断目标路径是不是文件夹

**Parameters**

- **target_path** —— 目标文件路径

**Returns**

- *bool* —— 目标路径是不是文件夹

<hr/>

## **scepter.modules.utils.file_clients.AliyunOssFs**

- target_path 格式: `oss://{bucket_name}/xxx/yy`

```yaml
NAME: AliyunOssFs
TEMP_DIR: None
AUTO_CLEAN: False
ENDPOINT:
BUCKET:
OSS_AK:
OSS_SK:
PREFIX: ""
WRITABLE: True
CHECK_WRITABLE: False
RETRY_TIMES: 10
```

<hr/>

## **scepter.modules.utils.file_clients.LocalFs**

- target_path 格式: 本地路径

```yaml
NAME: LocalFs
TEMP_DIR: None
AUTO_CLEAN: False
```

<hr/>

## **scepter.modules.utils.file_clients.HttpFs**

- target_path 格式: `http://xx/yy/zz`

```yaml
NAME: HttpFs
TEMP_DIR: None
AUTO_CLEAN: False
RETRY_TIMES: 10
```

<hr/>

## **scepter.modules.utils.file_clients.ModelscopeFs**

- target_path 单文件格式: `ms://{group}/{name}/:{revision}@{filename}`
- target_path 全文件格式: `ms://{group}/{name}/:{revision}`

```yaml
NAME: ModelscopeFs
TEMP_DIR: None
AUTO_CLEAN: False
RETRY_TIMES: 10
```

<hr/>
