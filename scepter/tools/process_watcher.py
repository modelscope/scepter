# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import random
import subprocess
import time
from datetime import datetime

import psutil

parser = argparse.ArgumentParser(
    description='Argparser for process watcher:\n')

parser.add_argument(
    '--python_engine',
    dest='python_engine',
    help='the engine path of python interpreter!',
    default='python',
    type=str,
)

parser.add_argument(
    '--script',
    dest='script',
    help='the script to run!',
    default='main_mmpose.py',
    type=str,
)
parser.add_argument(
    '--cfg',
    dest='cfg',
    help='the cfg file for script!',
    default='',
    type=str,
)

parser.add_argument(
    '--offset',
    dest='offset',
    help='the offset for modify the current seed!',
    default=0,
    type=int,
)

parser.add_argument(
    '--interval',
    dest='interval',
    help='the interval bettween two tasks!',
    default=0,
    type=int,
)

parser.add_argument(
    '--ins',
    dest='ins',
    help='the number of instance to start!',
    default=1,
    type=int,
)

parser.add_argument(
    '--runtime',
    dest='runtime',
    help='total runtime for all tasks!',
    default=144,
    type=int,
)

args = parser.parse_args()
seed = datetime.now().hour
random.seed(seed + args.offset)
random_port_start = random.randint(1100, 7080)
all_tasks = []
start_time = time.time()
for i in range(args.ins):
    cmd = (
        f'MASTER_PORT={random_port_start + i} {args.python_engine} -W ignore '
        f'{args.script} --cfg={args.cfg} ')
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    all_tasks.append([proc, cmd])
    time.sleep(args.interval)

while True:
    if time.time() - start_time > args.runtime * 60 * 60:
        print(f'reach to the max running time {args.runtime}h')
        for proc, cmd in all_tasks:
            print(f'kill process {proc.pid} with cmd {cmd}')
            os.system('kill -term {}'.format(proc.pid))
        break
    time.sleep(120)
    print(f'Have processed {time.time() - start_time}s')
    is_finished = False
    status_list = []
    for idx, (proc, cmd) in enumerate(all_tasks):
        current_process = psutil.Process(proc.pid)
        if not current_process.status() in ('running', 'sleeping',
                                            'disk-sleep', 'waking'):
            retcode = proc.wait()
            print(
                f'process {proc.pid} with cmd {cmd} exit with code {retcode}')
            if retcode == 0:
                print(f'task {proc.pid} finished!')
                status_list.append(idx)
            else:
                print(f'restart {proc.pid} with cmd {cmd}.')
                proc = subprocess.Popen(cmd, shell=True)
                all_tasks[idx] = [proc, cmd]
    for idx in status_list:
        all_tasks.pop(idx)
    if len(all_tasks) == 0:
        print('all tasks finished!')
        break
