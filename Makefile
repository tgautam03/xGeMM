CC = nvcc

HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/
CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto"

# MatrixFP32
build/MatrixFP32.o: src/MatrixFP32.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/MatrixFP32.cu -o build/MatrixFP32.o

# Utils
build/utils.o: src/utils.cpp
	$(CC) $(HOST_COMPILE_FLAG) $(ADD_EIGEN) src/utils.cpp -o build/utils.o

# Naive CPU and cBLAS
benchmark_cpu.out: test/benchmark_cpu.cpp src/cpu_xgemm.cpp build/MatrixFP32.o build/utils.o
	$(CC) $(ADD_EIGEN) $(CPU_OPTIMIZE) build/MatrixFP32.o build/utils.o src/cpu_xgemm.cpp test/benchmark_cpu.cpp -o benchmark_cpu.out

# Naive vs cuBLAS
benchmark_naive.out: src/naive_xgemm.cu test/benchmark_naive.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/naive_xgemm.cu test/benchmark_naive.cu -o benchmark_naive.out

# coalesced vs cuBLAS
build/coalesced_xgemm.o: src/coalesced_xgemm.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/coalesced_xgemm.cu -o build/coalesced_xgemm.o

benchmark_coalesced.out: test/benchmark_coalesced.cu build/MatrixFP32.o build/utils.o build/coalesced_xgemm.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o build/coalesced_xgemm.o test/benchmark_coalesced.cu -o benchmark_coalesced.out

# tiled vs cuBLAS
build/tiled_xgemm.o: src/tiled_xgemm.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/tiled_xgemm.cu -o build/tiled_xgemm.o

benchmark_tiled.out: test/benchmark_tiled.cu build/MatrixFP32.o build/utils.o build/tiled_xgemm.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o build/tiled_xgemm.o test/benchmark_tiled.cu -o benchmark_tiled.out

# coarse_1d vs cuBLAS
build/coarse_1d_xgemm.o: src/coarse_1d_xgemm.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/coarse_1d_xgemm.cu -o build/coarse_1d_xgemm.o

benchmark_coarse_1d.out: test/benchmark_coarse_1d.cu build/MatrixFP32.o build/utils.o build/coarse_1d_xgemm.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o build/coarse_1d_xgemm.o test/benchmark_coarse_1d.cu -o benchmark_coarse_1d.out

# coarse_2d vs cuBLAS
build/coarse_2d_xgemm.o: src/coarse_2d_xgemm.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/coarse_2d_xgemm.cu -o build/coarse_2d_xgemm.o

benchmark_coarse_2d.out: test/benchmark_coarse_2d.cu build/MatrixFP32.o build/utils.o build/coarse_2d_xgemm.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o build/coarse_2d_xgemm.o test/benchmark_coarse_2d.cu -o benchmark_coarse_2d.out


# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o