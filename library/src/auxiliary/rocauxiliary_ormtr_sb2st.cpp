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

#include "rocauxiliary_ormtr_sb2st.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename TA, typename TC>
rocblas_status rocsolver_ormtr_sb2st_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nb,
                                          TA A,
                                          const rocblas_int lda,
                                          TC C,
                                          const rocblas_int ldc)
{
    ROCSOLVER_ENTER_TOP("ormtr_sb2st", "-n", n, "-nb", nb, "--lda", lda, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // execution
    return rocsolver_ormtr_sb2st_template<T, rocblas_int, rocblas_stride, TA, TC>(
        handle, n, nb,

        A, shiftA, lda, strideA,

        C, shiftC, ldc, strideC,

        batch_count);
}

ROCSOLVER_END_NAMESPACE

/*
  * ===========================================================================
  *    C wrapper
  * ===========================================================================
  */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sormtr_sb2st(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       const rocblas_int nb,
                                                       float* A,
                                                       const rocblas_int lda,
                                                       float* C,
                                                       const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_sb2st_impl<float>(handle, n, nb, A, lda, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dormtr_sb2st(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       const rocblas_int nb,
                                                       double* A,
                                                       const rocblas_int lda,
                                                       double* C,
                                                       const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_sb2st_impl<double>(handle, n, nb, A, lda, C, ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cunmtr_hb2st(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       const rocblas_int nb,
                                                       rocblas_float_complex* A,
                                                       const rocblas_int lda,
                                                       rocblas_float_complex* C,
                                                       const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_sb2st_impl<rocblas_float_complex>(handle, n, nb, A, lda, C,
                                                                        ldc);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zunmtr_hb2st(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       const rocblas_int nb,
                                                       rocblas_double_complex* A,
                                                       const rocblas_int lda,
                                                       rocblas_double_complex* C,
                                                       const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormtr_sb2st_impl<rocblas_double_complex>(handle, n, nb, A, lda, C,
                                                                         ldc);
}

} // extern C
