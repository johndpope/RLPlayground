CPP_FLAGS=-std=c++11 -O3
chess=chess/*.cpp
HDR=chess/*.h

all: py/_chess.so

py/_chess.cpp: ${chess} ${HDR} py/chess.i
	swig -python -c++ -Ichess -o py/_chess.cpp py/chess.i

py/_chess.so: py/_chess.cpp py/setup.py
	python py/setup.py build_ext --build-lib=py -Ichess

.PHONY: clean
clean:
	rm py/chess.py py/_chess.cpp py/_chess.so
	rm py/*.pyc
	rm -rf build
