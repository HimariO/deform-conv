from setuptools import setup

setup(name='deform_conv',
      version='0.1',
      install_requires=[
          'pillow',
          'progress',
          'scipy',
          'numpy',
          'tensorflow-gpu',
          'keras',
          'joblib',
          'scikit-image',
          'matplotlib',
          'google-api-python-client',
          'h5py',
      ],
      zip_safe=False)
