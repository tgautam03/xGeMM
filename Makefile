CC = nvcc

COMPILE_FLAG = -c 
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/
OPTIMIZE = -O3 -march=native -ffast-math

# MatrixFP32
build/MatrixFP32.o: src/MatrixFP32.cu
	$(CC) $(COMPILE_FLAG) src/MatrixFP32.cu -o build/MatrixFP32.o 

# Utils
build/utils.o: src/utils.cpp
	$(CC) $(COMPILE_FLAG) $(ADD_EIGEN) src/utils.cpp -o build/utils.o 

# CPU vs cBLAS
build/cpu_xgemm.o: src/cpu_xgemm.cpp
	$(CC) $(COMPILE_FLAG) src/cpu_xgemm.cpp -o build/cpu_xgemm.o

benchmark_blas.out: test/benchmark_blas.cpp build/MatrixFP32.o build/utils.o build/cpu_xgemm.o
	$(CC) $(ADD_EIGEN) $(OPTIMIZE) build/MatrixFP32.o build/utils.o build/cpu_xgemm.o test/benchmark_blas.cpp -o benchmark_blas.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o