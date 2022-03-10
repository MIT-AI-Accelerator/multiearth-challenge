from setuptools import setup, find_packages

DISTNAME = 'multiearth-challenge'
DESCRIPTION = 'Repository for the MultiEarth 2022 public challenge.'
LICENSE = 'MIT'
AUTHOR = 'Greg Angelides'
URL = 'https://github.com/MIT-AI-Accelerator/multiearth-challenge'
INSTALL_REQUIRES = []
TESTS_REQUIRE = ['pytest >= 3.8']

if __name__ == '__main__':
    setup(name=DISTNAME,
          description=DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          install_requires=INSTALL_REQUIRES,
          url=URL,
          python_requires='>=3.7',
          packages=find_packages(where='src', exclude=['tests*']),
          package_dir={'': 'src'},
    )
