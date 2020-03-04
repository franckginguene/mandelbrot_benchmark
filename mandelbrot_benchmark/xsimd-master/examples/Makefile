CC=g++
NVCC=nvcc
CXXFLAGS= -g -std=c++11 -fpermissive -O3 -fopenmp
INC=-I/home/franck/Documents/projet/tedics_hpc/xsimd-master/include
mandelbrot:	mandelbrot.cpp
	$(CC)	$(CXXFLAGS)	$(INC) mandelbrot.cpp -o mandelbrot
clean:
	rm mandelbrot mandelbrot.o