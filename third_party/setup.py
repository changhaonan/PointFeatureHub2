from distutils.core import setup
setup(name="r2d2",
      version="1.0",
      description="r2d2 module.",
      author="Jerome Revaud",
      url="https://github.com/naver/r2d2",
      packages=["r2d2"],
      package_dir = {"" : "."},
      install_requires=["numpy"]
    )