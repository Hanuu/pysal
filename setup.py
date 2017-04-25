try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = dict(
      name='pyimgsaliency',
      version='0.1.1',
      description='A package for calculating image saliency',
      url='https://github.com/mamrehn/pyimgsaliency',
      author='Yann Henon',
      author_email='none',
      license='Apache',
      packages=['pyimgsaliency'],
      zip_safe=False
)

requires = ('numpy', 'scipy', 'skimage')

setup(requires=requires, **config)
