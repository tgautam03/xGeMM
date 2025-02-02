CC = nvcc

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/
CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto"

# MatrixFP32
build/MatrixFP32.o: src/MatrixFP32.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/MatrixFP32.cu -o build/MatrixFP32.o

# Utils
build/utils.o: src/utils.cu
	$(CC) $(DEVICE_COMPILE_FLAG) $(ADD_EIGEN) src/utils.cu -o build/utils.o

# Naive CPU and cBLAS
00a_benchmark_cpu.out: test/00a_benchmark_cpu.cpp build/MatrixFP32.o build/utils.o
	$(CC) $(ADD_EIGEN) $(CPU_OPTIMIZE) build/MatrixFP32.o build/utils.o test/00a_benchmark_cpu.cpp -o 00a_benchmark_cpu.out

# cuBLAS
00b_benchmark_cuBLAS.out: test/00b_benchmark_cuBLAS.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o test/00b_benchmark_cuBLAS.cu -o 00b_benchmark_cuBLAS.out

# Naive vs cuBLAS
01_benchmark_naive.out: src/01_naive_xgemm.cu test/01_benchmark_naive.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/01_naive_xgemm.cu test/01_benchmark_naive.cu -o 01_benchmark_naive.out

# coalesced vs cuBLAS
02_benchmark_coalesced.out: src/02_coalesced_xgemm.cu test/02_benchmark_coalesced.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/02_coalesced_xgemm.cu test/02_benchmark_coalesced.cu -o 02_benchmark_coalesced.out

# tiled vs cuBLAS
03_benchmark_tiled.out: src/03_tiled_xgemm.cu test/03_benchmark_tiled.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/03_tiled_xgemm.cu test/03_benchmark_tiled.cu -o 03_benchmark_tiled.out

# coarse_1d vs cuBLAS
04_benchmark_coarse_1d.out: src/04_coarse_1d_xgemm.cu test/04_benchmark_coarse_1d.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/04_coarse_1d_xgemm.cu test/04_benchmark_coarse_1d.cu -o 04_benchmark_coarse_1d.out

# coarse_2d vs cuBLAS
05_benchmark_coarse_2d.out: src/05_coarse_2d_xgemm.cu test/05_benchmark_coarse_2d.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/05_coarse_2d_xgemm.cu test/05_benchmark_coarse_2d.cu -o 05_benchmark_coarse_2d.out

# coarse_2d_vec vs cuBLAS
06_benchmark_coarse_2d_vec.out: src/06_coarse_2d_vec_xgemm.cu test/06_benchmark_coarse_2d_vec.cu build/MatrixFP32.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/utils.o src/06_coarse_2d_vec_xgemm.cu test/06_benchmark_coarse_2d_vec.cu -o 06_benchmark_coarse_2d_vec.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o