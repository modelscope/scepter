# -*- coding: utf-8 -*-
from enum import Enum


class Media(Enum):
    TEXT = 1
    IMAGE = 2
    VIDEO = 3
    AUDIO = 4


class HtmlVisualization(object):
    def __init__(
        self,
        allow_annotation=False,
        slice_size=1000,
        align='center',
        width_scale='60%',
        title='Visualization',
    ):
        self.content_list = []
        self.rows_meta = []
        self.allow_annotation = allow_annotation
        self.slice_size = slice_size
        self.align = align
        self.width_scale = width_scale
        self.title = title
        self.html_start = '<html>'
        self.html_head = f'<head><meta charset="utf-8"><title>{title}</title></head>'
        self.html_style = '''
            <style>
                table {
                    border-collapse: collapse;
                }
                td {
                    width: "{width_scale}";
                    align: "{align}";
                    margin: 0px;
                    border: 0px;
                    padding: 0px;
                    vertical-align: top;
                }
                video {
                    margin: 0px;
                    border: 0px solid #ccc;
                    padding: 0px;
                }
                textarea {
                    margin: 0px;
                                border: 0px;
                    padding: 0px;
                    resize: none;
                    border: 1px solid #ccc;
                }
            </style>
            <script>
                function adjustHeight() {
                    const textareas = document.querySelectorAll('textarea');
                    textareas.forEach(textarea => {
                    const td = textarea.parentNode;
                    const tdHeight = td.clientHeight;
                    textarea.style.height = tdHeight + 'px';
                    });
                }
                window.onload = adjustHeight;
                window.onresize = adjustHeight;
            </script>
        '''.replace('{width_scale}',
                    self.width_scale).replace('{align}', self.align)
        self.html_body = '<body>{BODY}</body>\n'
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
                   content_height=400,
                   content_width=600):
        if type == Media.TEXT:
            ret_str = '<td><textarea'  # noqa: E501
            # if content_height is not None:
            #     rows = f"rows={content_height//30}"
            #     ret_str += f" {rows}"
            if content_width is not None:
                cols = f"cols={content_width//15}"
                ret_str += f" {cols}"
            ret_str += f'>"{content}"</textarea></td>\n'
            sec_ret_str = f'<td align="center"><font size="3"><strong>{label}<strong></font></td>\n'
            return [ret_str, sec_ret_str]
        elif type == Media.IMAGE:
            ret_str = f'<td><img src="{content}"'
            if content_height is not None:
                height = f'height="{content_height}"'
                ret_str += f" {height}"
            if content_width is not None:
                width = f'width="{content_width}"'
                ret_str += f" {width}"
            ret_str += '  ></td>\n'
            sec_ret_str = f'<td align="center"><font size="3"><strong>{label}<strong></font></td>\n'
            return [ret_str, sec_ret_str]
        elif type == Media.VIDEO:
            ret_str = '<td><video'  # noqa
            if content_height is not None:
                height = f'height="{content_height}"'
                ret_str += f" {height}"
            if content_width is not None:
                width = f'width="{content_width}"'
                ret_str += f" {width}"
            ret_str += ' controls>'
            ret_str += f'<source src="{content}" type="video/mp4"></video></td>\n'
            sec_ret_str = f'<td align="center"><font size="3"><strong>{label}<strong></font></td>\n'
            return [ret_str, sec_ret_str]
        elif type == Media.AUDIO:
            ret_str = f'<td><audio src="{content}" controls></td>\n'
            sec_ret_str = f'<td align="center"><font size="3"><strong>{label}<strong></font></td>\n'
            return [ret_str, sec_ret_str]
        else:
            raise NotImplementedError

    def format_row(self):
        sample_id = 0
        all_sample_html = []
        for one_content, one_row_meta in zip(self.content_list,
                                             self.rows_meta):
            one_row_str = '<table><tr>'
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
            one_row_str += '</tr></table>'
            if self.allow_annotation:
                one_row_str = f'<label for="sample{sample_id}">{one_row_str}</label>'
            all_sample_html.append(one_row_str)
            sample_id += 1

        return '\n'.join(all_sample_html)

    def add_record(self,
                   content='',
                   label='',
                   type=Media.TEXT,
                   row_id=1,
                   col_id=1,
                   annotation_meta=None,
                   content_height=None,
                   content_width=None):
        if row_id >= len(self.content_list):
            self.content_list.append([])
            self.rows_meta.append([])
            if row_id != len(self.content_list) - 1:
                raise RuntimeError(
                    'row_id should be next number of the last row_id.')
        if col_id > len(self.content_list[row_id]):
            raise RuntimeError(
                'col_id should be next number of the last col_id.')
        format_col = self.format_col(content, f"{row_id}-{col_id}: {label}",
                                     type, content_height, content_width)

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


if __name__ == '__main__':
    from scepter.modules.utils.config import Config
    from scepter.modules.utils.file_system import FS
    FS.init_fs_client(Config(cfg_dict={}, load=False))

    image_content_oss = '0_probe_0_[1024_2048_3].jpg'
    content_oss = '6UTWGRG1lx08iRBx5REA01041200dzcb0E010.mp4'
    caption = 'a little girl says hello.'

    html_ins = HtmlVisualization(allow_annotation=True,
                                 slice_size=1000,
                                 title='Visualization',
                                 width_scale='100%')

    for i in range(4):
        content_url = FS.get_url(content_oss, skip_check=True)
        html_ins.add_record(content=content_url,
                            label='caption',
                            type=Media.VIDEO,
                            row_id=i,
                            col_id=0,
                            annotation_meta=None,
                            content_height=600,
                            content_width=None)
        html_ins.add_record(content=caption,
                            label='caption',
                            type=Media.TEXT,
                            row_id=i,
                            col_id=1,
                            annotation_meta=None,
                            content_height=600,
                            content_width=750)
        image_content_url = FS.get_url(image_content_oss, skip_check=True)
        html_ins.add_record(content=image_content_url,
                            label='caption',
                            type=Media.IMAGE,
                            row_id=i,
                            col_id=2,
                            annotation_meta=None,
                            content_height=600,
                            content_width=None)

    with FS.put_to('visualize.html') as local_path:
        html_ins.save_html(local_path)
