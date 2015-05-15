/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/traits/Atomic.hpp> // AtomicOp

#include <alpaka/core/Ops.hpp>      // Add, Sub, ...
#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC_CUDA_ONLY

//#define ALPAKA_ATOMIC_ADD_FLOAT_CAS   // Undefine this to use the floating point implementation of atomic add using atomicCAS for device with compute capability < 2.0.
// This should be slower then the version using atomicExch.

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA accelerator atomic ops.
                //#############################################################################
                class AtomicCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda(AtomicCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda(AtomicCuda &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(AtomicCuda const &) -> AtomicCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~AtomicCuda() noexcept = default;
                };
            }
        }
    }

    namespace traits
    {
        namespace atomic
        {
            //#############################################################################
            //! The specializations to execute the requested atomic ops of the CUDA accelerator.
            // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with CAS
            //#############################################################################
            //-----------------------------------------------------------------------------
            // Add.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Add,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Add,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Add,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Add,
                float>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    float * const addr,
                    float const & value)
                -> float
                {
                    boost::ignore_unused(atomic);
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Add,
                double>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    double * const addr,
                    double const & value)
                -> double
                {
                    boost::ignore_unused(atomic);
                    // Code from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

                    unsigned long long int * address_as_ull(reinterpret_cast<unsigned long long int *>(addr));
                    unsigned long long int old(*address_as_ull);
                    unsigned long long int assumed;
                    do
                    {
                        assumed = old;
                        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
                        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    }
                    while(assumed != old);
                    return __longlong_as_double(old);
                }
            };
            //-----------------------------------------------------------------------------
            // Sub.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Sub,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Sub,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Min.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Min,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Min,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            /*template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Min,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicMin(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Max.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Max,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Max,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            /*template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Max,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicMax(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Exch.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Exch,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Exch,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Exch,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Exch,
                float>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    float * const addr,
                    float const & value)
                -> float
                {
                    boost::ignore_unused(atomic);
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Inc.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Inc,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicInc(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Dec.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Dec,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicDec(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // And.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::And,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::And,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            /*template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::And,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicAnd(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Or.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Or,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Or,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            /*template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Or,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicOr(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Xor.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Xor,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Xor,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            /*template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Xor,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicXor(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Cas.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Cas,
                int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    int * const addr,
                    int const & compare,
                    int const & value)
                -> int
                {
                    boost::ignore_unused(atomic);
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Cas,
                unsigned int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned int * const addr,
                    unsigned int const & compare,
                    unsigned int const & value)
                -> unsigned int
                {
                    boost::ignore_unused(atomic);
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The CUDA accelerator atomic operation functor.
            //-----------------------------------------------------------------------------
            template<>
            struct AtomicOp<
                accs::cuda::detail::AtomicCuda,
                ops::Cas,
                unsigned long long int>
            {
                ALPAKA_FCT_ACC_CUDA_ONLY static auto atomicOp(
                    accs::cuda::detail::AtomicCuda const & atomic,
                    unsigned long long int * const addr,
                    unsigned long long int const & compare,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    boost::ignore_unused(atomic);
                    return atomicCAS(addr, compare, value);
                }
            };
        }
    }
}
