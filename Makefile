CC = nvcc

COMPILE_FLAG = -c 
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/

# Utils
build/utils.o: src/utils.cpp
	$(CC) $(COMPILE_FLAG) src/utils.cpp -o build/utils.o 

# Tiled vs CUBLAS
build/tiled_xgemm.o: src/tiled_xgemm.cu
	$(CC) $(COMPILE_FLAG) src/tiled_xgemm.cu -o build/tiled_xgemm.o

benchmark_tiled.out: build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu
	$(CC) $(LINK_CUBLAS) build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu -o benchmark_tiled.out

# cBLAS
benchmark_blas.out: test/benchmark_blas.cpp
	$(CC) $(ADD_EIGEN) test/benchmark_blas.cpp -o benchmark_blas.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o