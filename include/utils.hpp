#ifndef UTILS
#define UTILS
#include "../include/MatrixFP32.cuh"

// Initalizing MatrixFP32 with val
void init_mat(MatrixFP32 mat, float val);

// Initalizing MatrixFP32 randomly between (MAX_VAL, MIN_VAL)
void init_mat(MatrixFP32 mat, int MAX_VAL, int MIN_VAL);

// Print Matrix (Full or Partial)
void print_mat(MatrixFP32 mat, bool full);
void print_mat(Eigen::MatrixXd mat, bool full);

// Asserting matrices are same within the tolerance (eps)
void assert_mat(MatrixFP32 A_mat, MatrixFP32 B_mat, float eps);
void assert_mat(MatrixFP32 A_mat, Eigen::MatrixXd B_mat, float eps);
#endif