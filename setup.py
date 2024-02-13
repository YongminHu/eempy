from setuptools import setup, find_packages

setup(
    name='eempy',
    version='1.0',
    author='Yongmin Hu',
    author_email='yongminhu@outlook.com',
    packages=find_packages(),
    url='https://github.com/YongminHu/eempy',
    license='MIT',
    description='A python toolkit for excitation-emission matrix (EEM) analysis',
    install_requires=[
        "tensorly",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "scikit-image",
        "ipywidgets",
        "tlviz"
    ]
)
