from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'PCS',
    version = '0.1',
    description = 'PCS-WP7 yield estimation package',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/avigan/ELT-PCS/pulse',
    author = 'Arthur Vigan',
    author_email = 'arthur.vigan@lam.fr',
    license = 'MIT',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords = 'spectroscopy high-contrast exoplanet elt pcs',
    packages = ['pcs'],
    install_requires = [
        'numpy', 'scipy', 'astropy', 'matplotlib', 'pandas'
    ],
    include_package_data = True,
    package_data = {
        'pcs': ['data/*.fits', 'data/*.csv', 'data/*.dat'],
    },
    zip_safe=False
)
