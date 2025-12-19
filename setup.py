from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='digital_twinning',
    version='0.1.5',
    author='András Urbanics, Áron Friedman, Bence Popovics, Emese Vastag, Noémi Friedman',
    author_email='popbence@hun-ren.sztaki.hu',
    description='A comprehensive package for digital twin model updating and predictive modeling using machine learning and uncertainty quantification techniques',
    long_description = Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    url='https://github.com/TRACE-Structures/digital_twinning/',
    packages=find_packages(),
    py_modules=['digital_twinning'],
    install_requires=[
        'asyncio',
        'catboost',
        'emcee',
        'gPCE-model',
        'IPython',
        'lightgbm',
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'SALib',
        'scipy',
        'seaborn',
        'shap',
        'scikit-learn',
        'torch',
        'uncertain-variables',
        'xgboost'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)
