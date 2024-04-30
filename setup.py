from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Streaming video data via networks'

# Setting up
setup(
    name="vidstream",
    version=VERSION,
    author="Camille Lavigne",
    author_email="",
    description=DESCRIPTION,
    long_description=long_description,
    packages=find_packages(),
    install_requires=['networkx'],
    keywords=['python', 'levenshtein distance', 'edit type', 'error type'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
