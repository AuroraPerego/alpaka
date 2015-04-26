/**
* \file
* Copyright 2015 Benjamin Worpitz
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

#include <alpaka/traits/Rand.hpp>       // CreateNormalReal, ...
#include <alpaka/traits/mem/Space.hpp>  // SpaceT

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST_ACC

#include <alpaka/host/mem/Space.hpp>    // SpaceHost

#include <random>                       // std::mt19937, std::uniform_real_distribution, ...
#include <type_traits>                  // std::enable_if

namespace alpaka
{
    namespace traits
    {
        namespace rand
        {
            namespace distribution
            {
                //#############################################################################
                //! The host random number float normal distribution get trait specialization.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T>
                struct CreateNormalReal<
                    TAcc,
                    T,
                    typename std::enable_if<
                        std::is_same<
                            alpaka::mem::SpaceT<TAcc>,
                            alpaka::mem::SpaceHost>::value
                        && std::is_floating_point<T>::value>::type>
                {
                    ALPAKA_FCT_HOST_ACC static auto createNormalReal(
                        TAcc const & acc)
                    -> std::normal_distribution<T>
                    {
                        return std::normal_distribution<T>();
                    }
                };
                //#############################################################################
                //! The host random number float uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T>
                struct CreateUniformReal<
                    TAcc,
                    T,
                    typename std::enable_if<
                        std::is_same<
                            alpaka::mem::SpaceT<TAcc>,
                            alpaka::mem::SpaceHost>::value
                        && std::is_floating_point<T>::value>::type>
                {
                    ALPAKA_FCT_HOST_ACC static auto createUniformReal(
                        TAcc const & acc)
                    -> std::uniform_real_distribution<T>
                    {
                        return std::uniform_real_distribution<T>();
                    }
                };
                //#############################################################################
                //! The host random number integer uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T>
                struct CreateUniformUint<
                    TAcc,
                    T,
                    typename std::enable_if<
                        std::is_same<
                            alpaka::mem::SpaceT<TAcc>,
                            alpaka::mem::SpaceHost>::value
                        && std::is_integral<T>::value>::type>
                {
                    ALPAKA_FCT_HOST_ACC static auto createUniformUint(
                        TAcc const & acc)
                    -> std::uniform_int_distribution<T>
                    {
                        return std::uniform_int_distribution<T>(
                            0,  // For signed integer: std::numeric_limits<T>::lowest()
                            std::numeric_limits<T>::max());
                    }
                };
            }
            
            namespace generator
            {
                //#############################################################################
                //! The host random number default generator get trait specialization.
                //#############################################################################
                template<
                    typename TAcc>
                struct CreateDefault<
                    TAcc,
                    typename std::enable_if<
                        std::is_same<
                            alpaka::mem::SpaceT<TAcc>,
                            alpaka::mem::SpaceHost>::value>::type>
                {
                    ALPAKA_FCT_HOST_ACC static auto createDefault(
                        TAcc const & acc,
                        std::uint32_t const & seed)
                    -> std::mt19937
                    {
                        return std::mt19937(seed);
                    }
                };
            }
        }
    }
}
