# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import subprocess
import sys

if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version

source_folder = os.path.join(os.path.dirname(__file__), "scepter/workflow")
current_dir = os.path.dirname(__file__)
destination_folder = os.path.join(os.path.dirname(current_dir), "ComfyUI-Scepter")

if not os.path.exists(destination_folder):
    shutil.copytree(source_folder, destination_folder)
    print(f"{os.path.abspath(source_folder)} copy to {os.path.abspath(destination_folder)} success!")
else:
    print(f"{os.path.abspath(destination_folder)} exist.")

# pip install scepter
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scepter'])