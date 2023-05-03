from setuptools import setup, find_packages

DISTNAME = 'multiearth-challenge'
DESCRIPTION = 'This repository provides tools for working with imagery supplied by the MultiEarth 2023 public challenge.'
LICENSE = 'MIT'
AUTHOR = 'Greg Angelides'
URL = 'https://github.com/MIT-AI-Accelerator/multiearth-challenge'
INSTALL_REQUIRES = [
    'jupyter >= 1.0',
    'matplotlib >= 3.5',
    'netcdf4 >= 1.5',
    'numba >= 0.54',
    'numpy >= 1.18',
    'xarray >= 0.20',
]
if __name__ == '__main__':
    setup(name=DISTNAME,
          description=DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          install_requires=INSTALL_REQUIRES,
          url=URL,
          python_requires='>=3.8',
          packages=find_packages(where='./src'),
          package_dir={'': 'src'},
          include_package_data=True,
          version='1.0.0',
    )
