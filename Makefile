CC = nvcc

COMPILE_FLAG = -c 
LINK_CUBLAS = -lcublas

# Utils
build/utils.o: src/utils.cpp
	$(CC) $(COMPILE_FLAG) src/utils.cpp -o build/utils.o 

# Tiled vs CUBLAS
build/cublas_sgemm.o: src/cublas_sgemm.cu
	$(CC) $(COMPILE_FLAG) src/cublas_sgemm.cu -o build/cublas_sgemm.o 

build/tiled_xgemm.o: src/tiled_xgemm.cu
	$(CC) $(COMPILE_FLAG) src/tiled_xgemm.cu -o build/tiled_xgemm.o

benchmark_tiled.out: build/cublas_sgemm.o build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu
	$(CC) $(LINK_CUBLAS) build/cublas_sgemm.o build/tiled_xgemm.o build/utils.o test/benchmark_tiled.cu -o benchmark_tiled.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o