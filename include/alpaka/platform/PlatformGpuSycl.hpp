/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"

#include <string>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        struct SYCLGpuSelector
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_gpu() ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformGpuSycl = PlatformGenericSycl<detail::SYCLGpuSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PlatformGpuSycl>
    {
        using type = DevGenericSycl<PlatformGpuSycl>; // = DevGpuSycl
    };
} // namespace alpaka::trait

#endif
