/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool CPU, bool GPU, typename T, typename Ud, typename Uh>
void ormtr_sb2st_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nb,
                          Ud& dA,
                          const rocblas_int lda,
                          Uh& hA,
                          const rocblas_int bc)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // Generate A matrix
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Ud, typename Uh>
void ormtr_sb2st_getError(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nb,
                          Ud& dA,
                          const rocblas_int lda,
                          Ud& dC,
                          const rocblas_int ldc,
                          Uh& hA,
                          Uh& hC,
                          Uh& hCRes,
                          double* max_err)
{
    // input data initialization
    ormtr_sb2st_initData<true, true, T>(handle, n, nb, dA, lda, hA, 1);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ormtr_sb2st(handle, n, nb, dA.data(), lda, dC.data(), ldc));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // CPU lapack

    double err;
    *max_err = 0;
}

template <typename T>
void testing_ormtr_sb2st(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", n);

    rocblas_int hot_calls = argus.iters;

    // determine sizes
    size_t size_A = lda * n;
    size_t size_C = ldc * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cres = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nb < 0 || lda < n || ldc < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_sb2st(handle, n, nb, (T*)nullptr, lda, (T*)nullptr, ldc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_ormtr_sb2st(handle, n, nb, (T*)nullptr, lda, (T*)nullptr, ldc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_Cres, 1, size_Cres, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());

    // check quick return
    if(nb == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_sb2st(handle, n, nb, dA.data(), lda, dC.data(), ldc),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormtr_sb2st_getError<T>(handle, n, nb, dA, lda, dC, ldc, hA, hC, hCRes, &max_error);

    // // collect performance data
    // if(argus.timing && hot_calls > 0)
    //     ormtr_sb2st_getPerfData<T>(handle, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
    //                          argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nb", "lda", "ldc");
            rocsolver_bench_output(n, nb, lda, ldc);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_ORMTR_SB2ST(...) \
    extern template void testing_ormtr_sb2st<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMTR_SB2ST, FOREACH_SCALAR_TYPE, APPLY_STAMP)
