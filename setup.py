"""az/_fast.pyx (自己対戦の探索を速くする Cython の拡張) のビルド.

bl_ex のディレクトリで:
    python setup.py build_ext --inplace

az/ に _fast.*.so (Windows では _fast.*.pyd) ができる. ビルドしていなくても Python 版で動く (遅いだけ).
az/_fast.pyx を変更したら, もう一度ビルドする.
"""
from setuptools import Extension, setup
from Cython.Build import cythonize

ext = Extension("az._fast", sources=["az/_fast.pyx"])

setup(
    name="bl_ex_fast",
    ext_modules=cythonize([ext], compiler_directives={"language_level": 3}),
    packages=[],
)
