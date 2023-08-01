/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/st/Traits.hpp"

#include <cstddef>
#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The generic SYCL shared memory allocator.
    template<typename TDim>
    struct BlockSharedMemStGenericSycl
        : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl<TDim>>
    {
        explicit BlockSharedMemStGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item_st{work_item}
        {
        }

        sycl::nd_item<TDim::value> my_item_st;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename T, std::size_t TUniqueId, typename TDim>
    struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl<TDim>>
    {
        static auto declareVar(BlockSharedMemStGenericSycl<TDim> const& smem) -> T&
        {
            auto shMemBuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(smem.my_item_st.get_group());
            return *(shMemBuff.get());
        }
    };

    template<typename TDim>
    struct FreeSharedVars<BlockSharedMemStGenericSycl<TDim>>
    {
        static auto freeVars(BlockSharedMemStGenericSycl<TDim> const&) -> void
        {
            // shared memory block data will be reused
        }
    };
} // namespace alpaka::trait

#endif
