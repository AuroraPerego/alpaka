/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclAmd.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"
#include "alpaka/platform/PlatformGpuSyclAmd.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_AMD)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclAmd = BufGenericSycl<TElem, TDim, TIdx, PlatformGpuSyclAmd>;
} // namespace alpaka

#endif
