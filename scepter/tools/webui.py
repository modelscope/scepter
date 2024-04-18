# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import datetime
import importlib
import os
import random
import sys

import gradio as gr

import scepter
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger, init_logger

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)


def prepare(config):
    if 'FILE_SYSTEM' in config:
        for fs_info in config['FILE_SYSTEM']:
            FS.init_fs_client(fs_info)

    if 'LOG_FILE' in config:
        logger = get_logger()
        tid = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()) + ''.join(
            [str(random.randint(1, 10)) for i in range(3)])
        init_logger(logger, log_file=config['LOG_FILE'].format(tid))


class TabManager():
    def __init__(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        dest='config',
                        type=str,
                        default=os.path.join(
                            os.path.dirname(scepter.dirname),
                            'scepter/methods/studio/scepter_ui.yaml'))
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='Switch debug mode.')
    parser.add_argument(
        '--host',
        dest='host',
        default=None,
        help='The host of Gradio, default is set in ui config.')
    parser.add_argument('--port',
                        dest='port',
                        default=None,
                        help='The port of Gradio.')
    parser.add_argument('--language',
                        dest='language',
                        choices=['en', 'zh'],
                        default='en',
                        help='Now we only support english(en) and chinese(zh)')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(
            f"{args.config} doesn't exist, find this file in {os.path.dirname(scepter.dirname)}"
        )
        args.config = os.path.join(os.path.dirname(scepter.dirname),
                                   args.config)
        assert os.path.exists(args.config)
    config = Config(load=True, cfg_file=args.config)
    prepare(config)

    tab_manager = TabManager()
    interfaces = []
    for info in config['INTERFACE']:
        name = info.get('NAME_EN',
                        '') if args.language == 'en' else info['NAME']
        ifid = info['IFID']
        if not FS.exists(info['CONFIG']):
            info['CONFIG'] = os.path.join(os.path.dirname(scepter.dirname),
                                          info['CONFIG'])
        if not FS.exists(info['CONFIG']):
            raise f"{info['CONFIG']} doesn't exist."
        interface = None
        if ifid == 'home':
            from scepter.studio.home.home import HomeUI

            interface = HomeUI(info['CONFIG'],
                               is_debug=args.debug,
                               language=args.language,
                               root_work_dir=config.WORK_DIR)
        if ifid == 'preprocess':
            from scepter.studio.preprocess.preprocess import PreprocessUI

            interface = PreprocessUI(info['CONFIG'],
                                     is_debug=args.debug,
                                     language=args.language,
                                     root_work_dir=config.WORK_DIR)
        if ifid == 'self_train':
            from scepter.studio.self_train.self_train import SelfTrainUI

            interface = SelfTrainUI(info['CONFIG'],
                                    is_debug=args.debug,
                                    language=args.language,
                                    root_work_dir=config.WORK_DIR)
        if ifid == 'tuner_manager':
            from scepter.studio.tuner_manager.tuner_manager import TunerManagerUI
            interface = TunerManagerUI(info['CONFIG'],
                                       is_debug=args.debug,
                                       language=args.language,
                                       root_work_dir=config.WORK_DIR)
        if ifid == 'inference':
            from scepter.studio.inference.inference import InferenceUI

            interface = InferenceUI(info['CONFIG'],
                                    is_debug=args.debug,
                                    language=args.language,
                                    root_work_dir=config.WORK_DIR)
        if ifid == '':
            pass  # TODO: Add New Features
        if interface:
            interfaces.append((interface, name, ifid))
            setattr(tab_manager, ifid, interface)

    css = """
    .upload_zone { height: 100px; }
    """

    with gr.Blocks(css=css) as demo:
        if 'BANNER' in config:
            gr.HTML(config.BANNER)
        else:
            gr.Markdown(
                f"<h2><center>{config.get('TITLE', 'scepter studio')}</center></h2>"
            )
        setattr(tab_manager, 'user_name',
                gr.Text(value='admin', visible=False, show_label=False))
        with gr.Tabs(elem_id='tabs') as tabs:
            setattr(tab_manager, 'tabs', tabs)
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id=f'tab_{ifid}'):
                    interface.create_ui()
            for interface, label, ifid in interfaces:
                interface.set_callbacks(tab_manager)
        auth_info = {}
        if config.have('AUTH_INFO'):
            for auth_user in config.AUTH_INFO:
                auth_info[auth_user.USER] = auth_user.PASSWD

        def check_auth(user_name, password):
            if user_name in auth_info:
                return auth_info[user_name] == password
            else:
                return False

        def init_value(req: gr.Request):
            print(req.username, 'have login')
            return gr.Text(value=req.username, visible=False)

        if len(auth_info) > 0:
            demo.load(init_value, outputs=[tab_manager.user_name])

    demo.queue(status_update_rate=1).launch(
        server_name=args.host if args.host else config['HOST'],
        server_port=int(args.port) if args.port else config['PORT'],
        root_path=config['ROOT'],
        show_error=True,
        debug=True,
        enable_queue=True,
        auth=check_auth if len(auth_info) > 0 else None)
