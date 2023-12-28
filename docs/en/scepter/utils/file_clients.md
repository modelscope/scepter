# File System

This is the File System Module, designed to handle file transfer functionalities.

## Overview

The component currently supports three types of IO Handler:

1. scepter.utils.file_clients.AliyunOssFs
2. scepter.utils.file_clients.LocalFs
3. scepter.utils.file_clients.HttpFs

<hr/>

## Basic Usage

```python
from scepter.utils.file_system import FS
from scepter.utils.config import Config

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
# Download multiple files at once.
generator = FS.get_batch_objects_from(["xxx", "xxxx"])
for local_path in generator:
    print(local_path)
```
<hr/>

## **scepter.modules.utils.file_system.FileSystem**

By building various File IO Handlers, it supports read and write operations for different types of files.
<br>

### <font color="#0FB0E4">function **\_\_init\_\_**</font>

()

**Parameters**

<br>

### <font color="#0FB0E4">function **init_fs_client**</font>

( cfg: *scepter.modules.utils.config.Config* = None, logger = None ) -> str

The fs_client is instantiated through the cfg parameter and stored in the self._prefix_to_clients attribute, allowing access to the corresponding fs_client via the prefix.

**Parameters**

- **cfg** —— The Config used to build fs_client. If None, use LocalFs as default.
- **logger** —— Instantiated Logger to print or save log.

**Returns**

- *str* —— The prefix of instantiated fs_client

<br>

### <font color="#0FB0E4">function **get_fs_client**</font>

( target_path: *str*, safe: *bool* = False )

Retrieve the corresponding fs_client based on the prefix of the target_path.

**Parameters**

- **target_path** —— Target file path.
- **safe** —— In safe mode, return a copy of the client; otherwise, return the client itself.

**Returns**

- *BaseFs* —— Instantiated fs_client.

<br>

### <font color="#0FB0E4">function **get_from**</font>

( target_path: *str*, local_path: *str* = None, wait_finish: *bool* = False ) -> str

Download remote files to the local system.

**Parameters**

- **target_path** —— Remote file path.
- **local_path** —— Local file path; if None, use the cache path.
- **wait_finish** —— if True, only the card 0 of each machine will download the data, and the other cards will wait for the download by card 0 to finish.

**Returns**

- *str* —— Local save file path.

<br>

### <font color="#0FB0E4">function **get_dir_to_local_dir**</font>

( target_path: *str*, local_path: *str* = None, wait_finish: *bool* = False, timeout: *int* = 3600, worker_id: *int* = 0 ) -> str

Download a folder from a remote path to the local system.

**Parameters**

- **target_path** —— Remote folder path.
- **local_path** —— Local folder path; if None, use the cache path.
- **wait_finish** —— if True，only the 0-card of each machine downloads the data, while the other cards wait for the 0-card to finish downloading.
- **timeout** —— Download timeout duration.
- **worker_id** —— Deprecated

**Returns**

- *str* —— 本地文件夹路径

<br>

### <font color="#0FB0E4">function **get_object**</font>

(target_path: *str*) -> bytes

Read a remote file into memory

**Parameters**

- **target_path** — Target file path

**Returns**

- *bytes* — Binary data of the target file

<br>

### <font color="#0FB0E4">function **put_object**</font>

(local_data: *bytes*, target_path: *str*) -> bool

Upload a data stream to a specified file

**Parameters**

- **local_data** — Local data stream

- **target_path** — Target file path

**Returns**

- *bool* — Whether the upload was successful

<br>


### <font color="#0FB0E4">function **delete_object**</font>

(target_path: *str*) -> bool

Delete the target file

**Parameters**

- **target_path** — Target file path

**Returns**

- *bool* — Whether the deletion was successful

<br>

### <font color="#0FB0E4">function **get_batch_objects_from**</font>

(target_path_list: *list[str]*, wait_finish: *bool*) -> Iterator[str]

Batch download files

**Parameters**

- **target_path_list** — List of files to download

**Returns**

- *Iterator[str]* — Iterator for local file paths

- <br>

### <font color="#0FB0E4">function **put_batch_objects_to**</font>

(local_path_list: *list[str]*, target_path_list: *list[str]*, wait_finish: *bool*) -> Iterator[tuple[str, str]]

Batch upload files

**Parameters**

- **local_path_list** — List of files to upload

- **target_path_list** — List of target file paths

**Returns**

- *Iterator[tuple[str, str]]* — Returns pairs of local file and target file paths

<br>

### <font color="#0FB0E4">function **get_object_stream**</font>

(target_path: *str*, start: *int*, size: *int*, delimiter: *str*) -> bytes, int

Batch upload files

**Parameters**

- **target_path** — Target file

- **start** — Starting character position of the target stream

- **size** — Size of the target stream starting from the character position

- **delimiter** — Delimiter character for the end of the target stream

**Returns**

- *bytes, int* — Returns the data stream bytes and the end character position

<br>

### <font color="#0FB0E4">function **get_object_chunk_list**</font>

(target_path: *str*, chunk_num: *int* = 1, delimiter: *str* = None) -> list[bytes]

Get a remote file and divide it into chunks

**Parameters**

- **target_path** — Target file path

-**chunk_num** — Number of chunks

- **delimiter** — Delimiter to ensure the data downloaded is a complete record and not truncated in the middle

**Returns**

- *list[bytes]* — Chunked data

<br>

### <font color="#0FB0E4">function **get_url**</font>

(target_path: *str*, set_public=False, lifecycle: *int* = 360000) -> str

Get the URL of a remote file (only supports AliyunOssFs)

**Parameters**

- **target_path** — Target file path

- **lifecycle** — Valid duration

- **set_public** — Whether to provide a public link

**Returns**

- *str* — URL of the target file

<br>

### <font color="#0FB0E4">function **put_to**</font>

(target_path: *str*)

Supports uploading a local file to a remote path

**Parameters**

- **target_path** — Remote file path

**Returns**

- **None**

<br>

```python
# Used as a context manager
with FS.put_to(target_path) as local_path:
    # some operations on local_path.
```

<br>

### <font color="#0FB0E4">function **put_object_from_local_file**</font>

(local_path: *str*, target_path: *str*) -> bool

Push a local file to a remote path

**Parameters**

- **local_path** — Local file path

- **target_path** — Remote file path

**Returns**

- *bool* — Whether the upload was successful

<br>

### <font color="#0FB0E4">function **put_dir_from_local_dir**</font>

(local_dir: *str*, target_dir: *str*) -> bool

Push a local directory to a remote path

**Parameters**

- **local_dir** — Local directory path

- **target_dir** — Remote directory path

**Returns**

- *bool* — Whether the upload was successful

<br>

### <font color="#0FB0E4">function **add_target_local_map**</font>

(target_dir: *str*, local_dir: *str*) -> None

Save the mapping relationship between the remote directory path and the local directory path as key-value pairs to self._target_local_mapper

**Parameters**

- **target_dir** — Remote directory path

- **local_dir** — Local directory path

**Returns**

- *None*

<br>

### <font color="#0FB0E4">function **make_dir**</font>

(target_dir: *str*) -> bool

Create a remote directory

**Parameters**

- **target_dir** — Remote directory path

**Returns**

- *bool* — Whether the creation was successful

<br>

### <font color="#0FB0E4">function **exists**</font>

(target_path: *str*) -> bool

Check if the target path exists

**Parameters**

- **target_path** — Remote file path

**Returns**

- *bool* — Whether it exists

<br>

### <font color="#0FB0E4">function **map_to_local**</font>

(target_path: *str*) -> str, bool

Map the remote file path to a local path

**Parameters**

- **target_path** — Remote file path

**Returns**

- *str* — Local file path

- *bool* — Whether the local file is a temporary file

<br>

### <font color="#0FB0E4">function **walk_dir**</font>

(target_dir: *str*, recurse=True) -> Iterator

Get the file list under the remote directory

**Parameters**

- **target_dir** — Remote directory path

- **recurse** — Whether to traverse subdirectories, default is True

**Returns**

- *Iterator* — List of subfile paths

<br>

### <font color="#0FB0E4">function **is_local_client**</font>

(target_path: *str*) -> bool

Determine if the target file client is LocalFs

**Parameters**

- **target_path** — Target file path

**Returns**

- *bool* — Whether the client is LocalFs

<br>

### <font color="#0FB0E4">function **size**</font>

(target_path: *str*) -> int

Determine the size of the target file

**Parameters**

- **target_path** — Target file path

**Returns**

- *int* — Size of the target file

<br>

### <font color="#0FB0E4">function **isfile**</font>

(target_path: *str*) -> bool

Determine if the target path is an object

**Parameters**

- **target_path** — Target file path

**Returns**

- *bool* — Whether the target path is an object

<br>

### <font color="#0FB0E4">function **isdir**</font>

(target_path: *str*) -> bool

Determine if the target path is a directory

**Parameters**

- **target_path** — Target file path

**Returns**

- *bool* — Whether the target path is a directory

<hr/>

## **scepter.modules.utils.file_clients.AliyunOssFs**

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

```yaml
NAME: LocalFs
TEMP_DIR: None
AUTO_CLEAN: False
```

<hr/>

## **scepter.modules.utils.file_clients.HttpFs**

```yaml
NAME: HttpFs
TEMP_DIR: None
AUTO_CLEAN: False
RETRY_TIMES: 10
```

<hr/>
