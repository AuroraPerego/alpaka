/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclNvidia.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"
#include "alpaka/platform/PlatformGpuSyclNvidia.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclNvidia = BufGenericSycl<TElem, TDim, TIdx, PlatformGpuSyclNvidia>;
} // namespace alpaka

#endif
