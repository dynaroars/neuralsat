from setuptools import setup, find_packages

config = {
    'name': 'Libra',
    'version': '1.0',
    'author': 'Caterina Urban',
    'author_email': 'caterina.urban@gmail.com',
    'description': 'Static Program Analyzer for Python Data Science Applications',
    'url': 'https://caterinaurban.github.io/project/libra/',
    'license': 'MPL-2.0',
    'packages': find_packages('src'),
    'package_dir': {'': 'src'},
    'entry_points': {
             'console_scripts': [
                 'libra = libra.main:main',
                 ]
             },
    'install_requires': [
        'apronpy'
    ],
    'scripts': [],
}

setup(**config)
