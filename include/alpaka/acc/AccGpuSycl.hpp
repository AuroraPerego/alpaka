/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGpuSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/kernel/TaskKernelGpuSycl.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/PlatformGpuSycl.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <string>
#include <utility>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    //! The Intel GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel GPU target device.
    template<typename TDim, typename TIdx>
    class AccGpuSycl final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccGpuSycl<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The Intel GPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccGpuSycl<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccGpuSycl<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Intel GPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccGpuSycl<TDim, TIdx>>
    {
        using type = DevGpuSycl;
    };

    //! The Intel GPU SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccGpuSycl<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return TaskKernelGpuSycl<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Intel GPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PlatformType<AccGpuSycl<TDim, TIdx>>
    {
        using type = PlatformGpuSycl;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccGpuSycl<TDim, TIdx>>
    {
        using type = alpaka::TagGpuSycl;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagGpuSycl, TDim, TIdx>
    {
        using type = alpaka::AccGpuSycl<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
