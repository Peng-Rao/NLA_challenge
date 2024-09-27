#include <cstdio>
#include <ctime>
#include "lis.h"
#include "lis_config.h"


LIS_INT main(LIS_INT argc, char *argv[]) {
    LIS_Comm comm;
    LIS_MATRIX A0, A;
    LIS_VECTOR x, b, u;
    LIS_SOLVER solver;
    LIS_INT m, n, nn, nnz;
    LIS_INT i, j, ii, jj, ctr;
    LIS_INT is, ie;
    int nprocs, my_rank;
    LIS_INT nsol;
    LIS_INT err, iter, mtype, iter_double, iter_quad;
    double time, itime, ptime, p_c_time, p_i_time;
    LIS_REAL resid;
    char solvername[128];
    LIS_INT *ptr, *index;
    LIS_SCALAR *value;

    LIS_DEBUG_FUNC_IN;

    lis_initialize(&argc, &argv);

    lis_matrix_create(LIS_COMM_WORLD, &A);
    lis_vector_create(LIS_COMM_WORLD, &b);
    lis_vector_create(LIS_COMM_WORLD, &x);
    lis_solver_create(&solver);
    lis_solver_set_option("-i bicg -p none", solver);
    lis_solver_set_option("-tol 1.0e-9", solver);
    lis_matrix_set_type(A, LIS_MATRIX_CSR);

    lis_input(A, b, x, "/Users/raopend/Workspace/NLA_ch1/A2_w.mtx");
    lis_vector_duplicate(A, &x);
    lis_solve(A, b, x, solver);

    lis_solver_get_iter(solver, &iter);
    lis_solver_get_time(solver, &time);
    printf("number of iterations = %d\n", iter);
    printf("elapsed time = %e\n", time);

    lis_output_vector(x, LIS_FMT_MM, "result.mtx");
    -lis_solver_destroy(solver);
    lis_matrix_destroy(A);
    lis_vector_destroy(b);
    lis_vector_destroy(x);
    lis_finalize();
    return 0;
}
