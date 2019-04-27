from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Sentiment classification Keras',
      author='Wasutan Kitijerapat',
      author_email='wasutan.kiti@gmail.com',
      license='',
      install_requires=[
          'keras',
          'deepcut',
          'h5py',
          'pandas',
          'scikit-learn'
      ],
      zip_safe=False)