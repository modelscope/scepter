# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import subprocess
import threading
import time

import psutil
import torch

from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import as_time


class TaskStatus():
    def __init__(self):
        self.pid = -1
        self.retcode = -999
        self.error_log = None

    def __repr__(self):
        return f'Process {self.pid} retcode {self.retcode}, error msg: {self.error_msg}.'

    @property
    def error_msg(self):
        if os.path.exists(self.error_log):
            return "\n".join(open(self.error_log, "r").readlines()[-50:])
        else:
            return "No error msg."


def kill_job(pid):
    try:
        current_process = psutil.Process(pid)
        children = current_process.children(recursive=True)
        for child in children:
            child.terminate()
        current_process.terminate()
    except Exception:
        try:
            os.system(f'kill -9 {pid}')
        except Exception:
            pass


class Trainer():
    def __init__(self, run_script, status_message):
        self.run_script = run_script
        self.status_message = status_message
        self.proc = None

    def __call__(self, task_name):
        torch.cuda.empty_cache()
        error_folder = "./error_logs"
        os.makedirs(error_folder, exist_ok=True)
        self.status_message.error_log = f"{error_folder}/{int(time.time())}.log"
        cmd = f'PYTHONPATH=. python {self.run_script} ' \
              f'--cfg={task_name}/train.yaml 2> {self.status_message.error_log}'
        # cmd = [f"python {self.run_script}"]
        print(cmd)
        try:
            # self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.proc = subprocess.Popen(cmd, shell=True)
            self.status_message.pid = self.proc.pid
            self.status_message.retcode = self.proc.wait(
            )  # self.proc.wait(3600*24*14)
            # self.status_message.error_msg = self.proc.stderr.read()
        except Exception:
            self.status_message.retcode = -2
            # self.status_message.error_msg = self.proc.stderr.read()

    def terminate(self):
        print(f'Terminate {self.proc.pid} ...')
        kill_job(self.proc.pid)
        self.proc.terminate()
        print(f'Terminate {self.proc.pid} success.')


class TrainManager():
    '''
        Manage all the submitted training tasks and control the status of tasks.
        When instance this class, we should start the threading, which used to get the task name
        and start the training task.
    '''
    def __init__(self, run_script, work_dir):
        self.task_queue = []
        self.runing_tasks = {}
        self.run_script = run_script
        self.work_dir = work_dir

        def task_dispatch():
            while True:
                all_running_tasks = [k for k in self.runing_tasks]
                for k in all_running_tasks:
                    now_task = self.runing_tasks[k]
                    # print(now_task["train_status"].pid, psutil.pid_exists(now_task["train_status"].pid))
                    if (not psutil.pid_exists(now_task['train_status'].pid) and
                            not now_task['train_status'].retcode in (-1, 0)):
                        now_task['train_status'].retcode = -2
                    # print(now_task["train_status"])
                    if not now_task['train_status'].retcode == -999:
                        status_file = os.path.join(self.work_dir, k,
                                                   'status.json')
                        if FS.exists(status_file):
                            with FS.get_from(status_file) as local_status:
                                task_status = json.load(open(
                                    local_status, 'r'))
                            code = now_task['train_status'].retcode
                            task_status['code'] = code
                            task_status['end_time'] = time.time()
                            duration = task_status['end_time'] - task_status[
                                'start_time']
                            if code == 0:
                                message = f'''
                                            Training completed! \n
                                            Save in [ {k} ] \n
                                            Detail log please export the log. \n
                                            Take time [ {duration:.4f}s ] \n
                                            {self.check_memory()}
                                        '''
                                task_status['msg'] = message
                                task_status['status'] = 'success'
                            else:
                                err_msg = now_task['train_status'].error_msg
                                message = f'''
                                            Training failed! \n
                                            Error msg: {err_msg} \n
                                            Take time [ {duration:.4f}s ] \n
                                            {self.check_memory()}
                                        '''
                                task_status['msg'] = message
                                task_status['status'] = 'failed'
                            with FS.put_to(status_file) as local_path:
                                json.dump(task_status,
                                          open(local_path, 'w'),
                                          ensure_ascii=False)
                            train_ins = now_task['train_ins']
                            train_ins.terminate()
                            kill_job(now_task['train_status'].pid)
                        self.runing_tasks.pop(k)
                # print(len(self.task_queue))
                if len(self.task_queue) > 0 and len(self.runing_tasks) == 0:
                    task_name = self.task_queue.pop(0)
                    print(f'start task {task_name}')
                    status_message = TaskStatus()
                    train_ins = Trainer(self.run_script, status_message)
                    train_thread = threading.Thread(target=train_ins,
                                                    args=(os.path.join(
                                                        self.work_dir,
                                                        task_name), ))
                    train_thread.start()
                    time.sleep(10)
                    self.runing_tasks[task_name] = {
                        'train_ins': train_ins,
                        'train_status': status_message
                    }
                    status_file = os.path.join(self.work_dir, task_name,
                                               'status.json')
                    with FS.get_from(status_file) as local_status:
                        task_status = json.load(open(local_status, 'r'))
                    task_status['status'] = 'running'
                    for _ in range(10):
                        if status_message.pid > 0:
                            task_status['pid'] = status_message.pid
                            break
                        time.sleep(10)
                    task_status['update_time'] = time.time()
                    with FS.put_to(status_file) as local_path:
                        json.dump(task_status,
                                  open(local_path, 'w'),
                                  ensure_ascii=False)
                time.sleep(5)

        self.task_manage = threading.Thread(target=task_dispatch, daemon=True)
        self.task_manage.start()

    def check_memory(self):
        # Check Cuda Memory
        mem_msg = ''
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                free_mem = free_mem / (1024**3)
                total_mem = total_mem / (1024**3)
                mem_msg += f'GPU {device_id}: free mem {free_mem:.3f}G, total mem {total_mem:.3f}G \n'
        else:
            mem_msg += 'GPU is not available!'
        return mem_msg

    def get_status(self, task_name):
        if task_name is None:
            return ''
        status_file = os.path.join(self.work_dir, task_name, 'status.json')
        if not FS.exists(status_file):
            return ''
        with FS.get_from(status_file) as local_status:
            task_status = json.load(open(local_status, 'r'))
        return task_status['status']

    def get_log(self, task_name):
        if task_name is None:
            return ''
        status_file = os.path.join(self.work_dir, task_name, 'status.json')
        if not FS.exists(status_file):
            return ''
        with FS.get_from(status_file) as local_status:
            task_status = json.load(open(local_status, 'r'))
        time.sleep(2)
        log_msg = ''
        if task_status['status'] == 'queue':
            start_time = task_status['start_time']
            if task_name in self.task_queue:
                log_msg += f'Task status: Queuing. Still have {self.task_queue.index(task_name) + 1} tasks.\n'
            else:
                log_msg += 'Task status: Queuing. Still have 0 tasks.\n'
            log_msg += f'Have waited for {as_time(time.time() - start_time)}\n\n'
            log_msg += f'Memory status: {self.check_memory()}\n'
        elif task_status['status'] == 'running':
            start_time = task_status['start_time']
            update_time = task_status['update_time']
            log_msg += (
                f'Task status: Running. Have run for {as_time(time.time() - update_time)} after waiting'
                f'for {as_time(update_time - start_time)}.\n\n')
            std_log = os.path.join(self.work_dir, task_name, 'std_log.txt')
            out_log = os.path.join(self.work_dir, task_name,
                                   'output_std_log.txt')
            output_msg = []
            if os.path.exists(std_log):
                fp_w = open(out_log, 'w')
                output_msg.append(f'Model {task_name} Start Training....')
                with open(std_log, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line == '':
                            continue
                        if 'OSS_AK' in line or 'OSS_AK'.lower() in line:
                            continue
                        if 'OSS_SK' in line or 'OSS_SK'.lower() in line:
                            continue
                        if 'Restored' in line or 'Restored'.lower() in line:
                            continue
                        # if 'Stage' in line:
                        output_msg.append(line)
                        fp_w.write(f'{line}\n')
                fp_w.close()
            else:
                fp_w = open(out_log, 'w')
                fp_w.close()
            log_msg += f'Memory status: {self.check_memory()}\n\n'
            log_msg += 'Recent output as follows:\n\n'
            log_msg += '\n\n'.join(output_msg)
        elif task_status['status'] == 'success':
            start_time = task_status['start_time']
            update_time = task_status['update_time']
            end_time = task_status['end_time']
            output_msg = []
            out_log = os.path.join(self.work_dir, task_name,
                                   'output_std_log.txt')
            std_log = os.path.join(self.work_dir, task_name, 'std_log.txt')
            if os.path.exists(std_log):
                fp_w = open(out_log, 'w')
                with open(std_log, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line == '':
                            continue
                        if 'OSS_AK' in line or 'OSS_AK'.lower() in line:
                            continue
                        if 'OSS_SK' in line or 'OSS_SK'.lower() in line:
                            continue
                        if 'Restored' in line or 'Restored'.lower() in line:
                            continue
                        output_msg.append(line)
                        fp_w.write(f'{line}\n')
                fp_w.close()
            else:
                fp_w = open(out_log, 'w')
                fp_w.close()
            log_msg += (
                f'Task status: Success. Have run for {as_time(end_time - update_time)} after waiting'
                f'for {as_time(update_time - start_time)}.\n\n')
            log_msg += f'Memory status: {self.check_memory()}\n\n'
            log_msg += '\n\n'.join(output_msg)
        elif task_status['status'] == 'failed':
            start_time = task_status['start_time']
            update_time = task_status['update_time']
            end_time = task_status['end_time']
            err_msg = task_status['msg']
            log_msg += (
                f'Task status: Failed. Have run for {as_time(end_time - update_time)} after waiting'
                f'for {as_time(update_time - start_time)}.\n\n')
            log_msg += f'Memory status: {self.check_memory()}\n\n'
            log_msg += 'The error msg is as follows:\n\n'
            log_msg += f'{err_msg}'
        return log_msg

    def start_task(self, task_name):
        if task_name is None:
            return
        self.task_queue.append(task_name)
        task_status = {
            'status': 'queue',
            'position': len(self.task_queue),
            'start_time': time.time(),
            'end_time': time.time()
        }
        status_file = os.path.join(self.work_dir, task_name, 'status.json')
        with FS.put_to(status_file) as local_path:
            json.dump(task_status, open(local_path, 'w'), ensure_ascii=False)

    def stop_task(self, task_name):
        if task_name in self.runing_tasks:
            task_info = self.runing_tasks.pop(task_name)
            train_status = task_info['train_status']
            train_ins = task_info['train_ins']
            kill_job(train_status.pid)
            train_ins.terminate()
        elif task_name in self.task_queue:
            self.task_queue.remove(task_name)
        else:
            pass
        # modify the task's status met

    def __del__(self):
        for task_name in self.task_queue:
            status_file = os.path.join(self.work_dir, task_name, 'status.json')
            with FS.get_from(status_file) as local_status:
                task_status = json.load(open(local_status, 'r'))
            task_status['status'] = 'failed'
            task_status['msg'] = 'Main process has been killed.'
            with FS.put_to(status_file) as local_path:
                json.dump(task_status,
                          open(local_path, 'w'),
                          ensure_ascii=False)

        pass


if __name__ == '__main__':
    train_ins = TrainManager('', '')
    for i in range(5):
        train_ins.start_task(f'{i}')
    time.sleep(300)
