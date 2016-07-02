from distutils.core import setup, Extension

extension_mod = Extension(
  "_chess", 
  sources=["py/_chess.cpp", "chess/game.cpp", "chess/utils.cpp"],
  extra_compile_args=["-std=c++11", "-fopenmp"],
  extra_link_args=["-lgomp"])

setup(name="chess", ext_modules=[extension_mod])
