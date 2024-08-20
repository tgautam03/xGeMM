CC = nvcc

COMPILE_FLAG = -c 
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/

# Utils
build/utils.o: src/utils.cpp
	$(CC) $(COMPILE_FLAG) src/utils.cpp -o build/utils.o 

# cBLAS
benchmark_blas.out: test/benchmark_blas.cpp
	$(CC) $(ADD_EIGEN) test/benchmark_blas.cpp -o benchmark_blas.out

# Naive vs CUBLAS
build/naive_xgemm.o: src/naive_xgemm.cu
	$(CC) $(COMPILE_FLAG) src/naive_xgemm.cu -o build/naive_xgemm.o

benchmark_naive.out: build/naive_xgemm.o build/utils.o test/benchmark_naive.cu
	$(CC) $(LINK_CUBLAS) build/naive_xgemm.o build/utils.o test/benchmark_naive.cu -o benchmark_naive.out

# Coalesced vs CUBLAS
build/coalesced_xgemm.o: src/coalesced_xgemm.cu
	$(CC) $(COMPILE_FLAG) src/coalesced_xgemm.cu -o build/coalesced_xgemm.o

benchmark_coalesced.out: build/coalesced_xgemm.o build/utils.o test/benchmark_coalesced.cu
	$(CC) $(LINK_CUBLAS) build/coalesced_xgemm.o build/utils.o test/benchmark_coalesced.cu -o benchmark_coalesced.out

# Tiled vs CUBLAS
build/tiled_xgemm.o: src/tiled_xgemm.cu
	$(CC) $(COMPILE_FLAG) src/tiled_xgemm.cu -o build/tiled_xgemm.o

benchmark_tiled.out: build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu
	$(CC) $(LINK_CUBLAS) build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu -o benchmark_tiled.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o