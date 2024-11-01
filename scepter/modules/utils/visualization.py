# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum


class Media(Enum):
    TEXT = 1
    IMAGE = 2
    VIDEO = 3
    AUDIO = 4
    IMAGE_PAIR = 5


class HtmlVisualization(object):
    def __init__(self,
                 allow_annotation=False,
                 slice_size=1000,
                 align='center',
                 width_scale='60%',
                 title='Visualization',
                 height=600,
                 width=None,
                 text_cols=40):
        self.content_list = []
        self.rows_meta = []
        self.allow_annotation = allow_annotation
        self.slice_size = slice_size
        self.align = align
        self.width_scale = width_scale
        self.title = title
        self.html_start = '<html>'
        self.html_head = f'<head><meta charset="utf-8"><title>{title}</title></head>'
        self.height = height if height is not None else '600'
        self.width = width if width is not None else 'auto'
        self.text_cols = text_cols if text_cols is not None else 'auto'
        self.html_style = ('''
            <style> \n
                .container {
                    display: flex;
                    position: relative; \n
                    overflow: hidden; \n
                    justify-content: center; \n
                    align-items: center; \n
                    width: 600; \n
                    height: {pair_height};
                    border: 2px solid #ccc; \n
                } \n
                .image {
                    display: flex;
                    position: absolute; \n
                    width: 100%; \n
                    height: 100%; \n
                    transition: 0.4s ease; \n
                }\n
                .image img {
                    width: 100%; \n
                    height: 100%; \n
                    object-fit: contain; \n
                } \n
                .slider {
                    position: absolute; \n
                    cursor: ew-resize; \n
                    height: 100%; \n
                    background-color: rgba(255, 255, 255, 0.5); \n
                    z-index: 10; \n
                } \n
                video { \n
                    width: auto;
                    height: 100%;
                    margin: 0px; \n
                    border: 0px solid #ccc; \n
                    padding: 0px; \n
                } \n
                textarea { \n
                    margin: 0px; \n
                                border: 0px; \n
                    padding: 0px; \n
                    resize: none; \n
                    border: 1px solid #ccc; \n
                } \n
            </style> \n
            \n
        '''.replace('{width_scale}', self.width_scale).replace(
            '{align}', self.align).replace('{pair_height}', f'{self.height}'))

        self.html_body_script = '''
                <script>\n
                    const containers = document.querySelectorAll('.container'); \n
                    containers.forEach(container => {\n
                        let isDragging = true;\n
                        const slider = container.querySelector('.slider')\n
                        const image2 = container.querySelector('#image2')\n

                        container.addEventListener('mousedown', () => {\n
                            isDragging = true;\n
                        });\n


                        container.addEventListener('mouseup', () => {\n
                            isDragging = true;\n
                        });\n

                        container.addEventListener('mousemove', (event) => {\n
                            if (!isDragging) return;\n
                            const { clientX } = event;\n

                            const { left, width } = container.getBoundingClientRect();\n

                            let percentage = (clientX - left) / width * 100;\n


                            // 限制百分比在0到100之间\n

                            percentage = Math.max(0, Math.min(100, percentage));\n

                            image2.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;\n

                            slider.style.left = `${percentage}%`;\n

                            console.info(slider.style.left);\n

                        });\n
                        // 初始化滑块位置\n
                        slider.style.left = '50%';\n
                    });\n
                </script>\n
                <script> \n
                    const videos = document.querySelectorAll('video'); \n
                    \n
                    const observer = new IntersectionObserver((entries) => { \n
                        entries.forEach(entry => { \n
                            if (entry.isIntersecting) { \n
                                const video = entry.target; \n
                                video.src = video.dataset.src; \n
                                video.load(); \n
                                observer.unobserve(video); \n
                            } \n
                        }); \n
                    }); \n
                    \n
                    videos.forEach(video => { \n
                        observer.observe(video); \n
                    }); \n
                    \n
                    function adjustHeight() { \n
                        const textareas = document.querySelectorAll('textarea'); \n
                        textareas.forEach(textarea => { \n
                        const td = textarea.parentNode; \n
                        const tdHeight = td.clientHeight; \n
                        textarea.style.height = tdHeight + 'px'; \n
                        }); \n
                    } \n
                    window.onload = adjustHeight; \n
                    window.onresize = adjustHeight; \n
                </script> \n

        '''

        self.html_body = '<body>{BODY}\n' + self.html_body_script + '</body>\n'

        self.html_end = '</html>'

        self.html_script = '''
                <script>
                    function saveSamples() {
                        let selectedSamples = document.querySelectorAll('input[name="sample[]"]:checked');
                        let notSelectedSamples = document.querySelectorAll('input[name="sample[]"]:not(:checked)');
                        let sampleUrls = [];
                        for (let i=0; i<selectedSamples.length; i++) {
                            sampleUrls.push(selectedSamples[i].value + "#;#" + "1");
                        }
                        for (let i=0; i<notSelectedSamples.length; i++) {
                            sampleUrls.push(notSelectedSamples[i].value + "#;#" + "0");
                        }
                        let fileContent = sampleUrls.join('\\n');
                        let file = new Blob([fileContent], {type: 'text/plain'});
                        let a = document.createElement('a');
                        a.href = URL.createObjectURL(file);
                        a.download = 'result.txt';
                        a.click();
                    }
                </script>

                '''
        self.label_button = (
            '<table><tr><td>' +
            "<button style='height: 50px;' type=\"button\" onclick=\"saveSamples()\">Save Samples</button>"
            + '</td></tr></table>')

    def format_col(self,
                   content='',
                   label='',
                   type=Media.TEXT,
                   show_label=True,
                   cols_span=1):
        if type == Media.TEXT:
            ret_str = '<textarea'  # noqa: E501
            if self.height is not None:
                rows = f"rows={self.height//30}"
                ret_str += f" {rows}"
            if self.width is not None:
                cols = f"cols={self.text_cols * cols_span}"
                ret_str += f" {cols}"
            ret_str += f'>"{content}"</textarea>'
            sec_ret_str = f'<font size="3"><strong>{label}<strong></font>' if show_label else ''
        elif type == Media.IMAGE:
            ret_str = f'<img src="{content}"'
            if self.height is not None:
                height = f'height="{self.height}"'
                ret_str += f" {height}"
            if self.width is not None:
                width = f'width="{self.width}"'
                ret_str += f" {width}"
            ret_str += '  >'
            sec_ret_str = f'<font size="3"><strong>{label}<strong></font>' if show_label else ''
        elif type == Media.VIDEO:
            ret_str = '<video'  # noqa
            if self.height is not None:
                height = f'height="{self.width}"'
                ret_str += f" {height}"
            if self.width is not None:
                width = f'width="{self.width}"'
                ret_str += f" {width}"
            ret_str += ' preload="none" controls>'
            ret_str += f'<source src="{content}" type="video/mp4"></video>'
            sec_ret_str = f'<font size="3"><strong>{label}<strong></font>' if show_label else ''
        elif type == Media.AUDIO:
            ret_str = f'<audio src="{content}" controls>'
            sec_ret_str = f'<font size="3"><strong>{label}<strong></font>' if show_label else ''
        elif type == Media.IMAGE_PAIR:
            assert isinstance(content, (list, tuple)) and len(content) == 2
            ret_str = '\n'
            ret_str += '       <div class="container"'
            ret_str += (
                f'> \n'
                f'      <div class="image" id="image1">'  # noqa
                f'          <img src="{content[1]}" alt="before">\n'  # noqa
                f'      </div>\n'  # noqa
                f'      <div class="image" id="image2" style="clip-path: inset(0 50% 0 0);">\n'  # noqa
                f'            <img src="{content[0]}" alt="after">\n'  # noqa
                f'      </div>\n'  # noqa
                f'      <div class="slider" id="slider"></div>\n'  # noqa
                f'')
            sec_ret_str = f'<font size="3"><strong>{label}<strong></font>' if show_label else ''
        else:
            raise NotImplementedError
        if cols_span > 1:
            ret_str = f'<th colspan="{cols_span}">{ret_str}</th>\n'
            sec_ret_str = f'<th colspan="{cols_span}">{sec_ret_str}</th>\n' if not sec_ret_str == '' else sec_ret_str
        else:
            ret_str = f'<td>{ret_str}</td>\n'
            sec_ret_str = f'<td align="center">{sec_ret_str}</td>\n' if not sec_ret_str == '' else sec_ret_str
        return [ret_str, sec_ret_str]

    def format_row(self):
        sample_id = 0
        all_sample_html = []
        for one_content, one_row_meta in zip(self.content_list,
                                             self.rows_meta):
            one_row_str = '<tr>'
            one_row_str += '\n'.join([v[0] for v in one_content])
            if self.allow_annotation:
                row_meta = '#;#'.join(one_row_meta)
                one_row_str += (
                    f'<td><input type="checkbox" class="large-checkbox" '
                    f'id="sample{sample_id}" name="sample[]" value="{row_meta}"></td>\n'
                )
            one_row_str += '</tr><tr>'
            one_row_str += '\n'.join([v[1] for v in one_content])  # noqa
            if self.allow_annotation:  # noqa
                one_row_str += f'<td></td>\n'  # noqa
            one_row_str += '</tr>'
            if self.allow_annotation:
                one_row_str = f'<label for="sample{sample_id}">{one_row_str}</label>'
            all_sample_html.append(one_row_str)
            sample_id += 1

        return '<table>' + '\n'.join(all_sample_html) + '</table>'

    def add_record(self,
                   content,
                   label='',
                   type=Media.TEXT,
                   row_id=1,
                   col_id=1,
                   cols_span=1,
                   annotation_meta=None,
                   show_label=True):
        if row_id >= len(self.content_list):
            self.content_list.append([])
            self.rows_meta.append([])
            if row_id != len(self.content_list) - 1:
                raise RuntimeError(
                    'row_id should be next number of the last row_id.')
        if col_id > len(self.content_list[row_id]):
            raise RuntimeError(
                'col_id should be next number of the last col_id.')
        format_col = self.format_col(content,
                                     f"{row_id}-{col_id}: {label}",
                                     type,
                                     show_label=show_label,
                                     cols_span=cols_span)

        annotation_meta = annotation_meta if annotation_meta else ''
        if col_id == len(self.content_list[row_id]):
            self.content_list[row_id].append(format_col)
            self.rows_meta[row_id].append(annotation_meta)
        else:
            self.content_list[row_id][col_id] = format_col
            self.rows_meta[row_id][col_id] = annotation_meta

    def save_html(self, path):
        html_body = self.format_row()
        ret_html_list = [
            self.html_start, self.html_head, self.html_style,
            self.html_body.replace('{BODY}', html_body)
        ]
        if self.allow_annotation:
            ret_html_list.append(self.label_button)
            ret_html_list.append(self.html_script)
        ret_html_list.append(self.html_end)
        ret_html = '\n'.join(ret_html_list)
        with open(path, 'w') as f:
            f.write(ret_html)
