# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os

import setuptools


def get_long_description():
    with open('readme.md') as f:
        long_description = f.read()
    return long_description


def get_version():
    version_path = 'scepter/version.py'
    with open(version_path) as f:
        exec(compile(f.read(), version_path, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            relative_base = os.path.dirname(fname)
            absolute_target = os.path.join(relative_base, target)
            for info in parse_require_file(absolute_target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('http'):
                    print('skip http requirements %s' % line)
                    continue
                if line and not line.startswith('#') and not line.startswith(
                        '--'):
                    for info in parse_line(line):
                        yield info
                elif line and line.startswith('--find-links'):
                    eles = line.split()
                    for e in eles:
                        e = e.strip()
                        if 'http' in e:
                            info = dict(dependency_links=e)
                            yield info

    def gen_packages_items():
        items = []
        deps_link = []
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                if 'dependency_links' not in info:
                    parts = [info['package']]
                    if with_version and 'version' in info:
                        parts.extend(info['version'])
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            parts.append(';' + platform_deps)
                    item = ''.join(parts)
                    items.append(item)
                else:
                    deps_link.append(info['dependency_links'])
        return items, deps_link

    return gen_packages_items()


def backupfile():
    contents = open('scepter/__init__.py', 'r').read()
    with open('scepter/__init__.py', 'w') as f:
        f.write(contents.replace('from scepter import task', ''))
    return contents


def restorefile(contents):
    with open('scepter/__init__.py', 'w') as f:
        f.write(contents)


required = parse_requirements()

contents = backupfile()
print([
    pkg for pkg in setuptools.find_packages()
    if '__pycache__' not in pkg and 'scepter' in pkg
])
setuptools.setup(
    name='scepter',
    version=get_version(),
    author='Tongyi Lab',
    author_email='',
    description='',
    keywords='compute vision, framework, generation, image edition.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='',
    packages=[
        pkg for pkg in setuptools.find_packages()
        if '__pycache__' not in pkg and 'scepter' in pkg
    ],
    include_package_data=True,
    # package_data={'': ['*.yaml']},
    data_files=[('lib/docs', glob.glob('docs/*.md') + glob.glob('docs/*/*.md'))
                ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=required,
)

restorefile(contents)
