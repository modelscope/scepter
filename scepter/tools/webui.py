# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import datetime
import os
import random

import gradio as gr

import scepter
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger, init_logger


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

    with gr.Blocks() as demo:
        if 'BANNER' in config:
            gr.HTML(config.BANNER)
        else:
            gr.Markdown(
                f"<h2><center>{config.get('TITLE', 'scepter studio')}</center></h2>"
            )
        with gr.Tabs(elem_id='tabs') as tabs:
            setattr(tab_manager, 'tabs', tabs)
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id=f'tab_{ifid}'):
                    interface.create_ui()
            for interface, label, ifid in interfaces:
                interface.set_callbacks(tab_manager)

    demo.queue(status_update_rate=1).launch(
        server_name=args.host if args.host else config['HOST'],
        server_port=args.port if args.port else config['PORT'],
        root_path=config['ROOT'],
        show_error=True,
        debug=True,
        enable_queue=True)
