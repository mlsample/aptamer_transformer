from setuptools import setup, find_packages

setup(
    name='aptamer_transformer',
    version='0.01',
    packages=find_packages(),
    description='Transformers for aptamer sequences',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mlsample/aptamer_transformer',
    author='Matthew Sample',
    author_email='matsample1@gmail.com',
    license='MIT',
    # dependencies can be listed under install_requires
)