from distutils.core import setup, Extension

extension_mod = Extension(
  "_chess", 
  sources=["py/_chess.cpp", "chess/game.cpp", "chess/io.cpp"],
  extra_compile_args=["-std=c++11"])

setup(name="chess", ext_modules=[extension_mod])
