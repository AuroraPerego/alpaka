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

#include <alpaka/core/DevProps.hpp> // DevProps
#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The device traits.
        //-----------------------------------------------------------------------------
        namespace dev
        {
            //#############################################################################
            //! The device type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct DevType;

            //#############################################################################
            //! The device manager type trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct DevManType;

            //#############################################################################
            //! The device get trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetDev;

            //#############################################################################
            //! The device properties get trait.
            //#############################################################################
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetProps;

            //#############################################################################
            //! The device reset trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetFreeMemBytes;

            //#############################################################################
            //! The device reset trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct Reset;
        }
    }

    //-----------------------------------------------------------------------------
    //! The device trait accessors.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        //#############################################################################
        //! The device type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using DevT = typename traits::dev::DevType<typename std::decay<T>::type>::type;

        //#############################################################################
        //! The device manager type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using DevManT = typename traits::dev::DevManType<typename std::decay<T>::type>::type;

        //-----------------------------------------------------------------------------
        //! \return The device this object is bound to.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST auto getDev(
            T const & t)
        -> decltype(traits::dev::GetDev<T>::getDev(std::declval<T const &>()))
        {
            return traits::dev::GetDev<
                T>
            ::getDev(
                t);
        }

        //-----------------------------------------------------------------------------
        //! \return All the devices available on this accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TDevMan>
        ALPAKA_FCT_HOST auto getDevs()
        -> std::vector<DevT<TDevMan>>
        {
            std::vector<DevT<TDevMan>> vDevices;

            std::size_t const uiDeviceCount(TDevMan::getDevCount());
            for(std::size_t uiDeviceIdx(0); uiDeviceIdx < uiDeviceCount; ++uiDeviceIdx)
            {
                vDevices.push_back(TDevMan::getDevByIdx(uiDeviceIdx));
            }

            return vDevices;
        }

        //-----------------------------------------------------------------------------
        //! \return The device properties.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FCT_HOST auto getProps(
            TDev const & dev)
        -> DevProps
        {
            return traits::dev::GetProps<
                TDev>
            ::getProps(
                dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The free memory on the device in Bytes.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FCT_HOST auto getFreeMemBytes(
            TDev const & dev)
        -> std::size_t
        {
            return traits::dev::GetFreeMemBytes<
                TDev>
            ::getFreeMemBytes(
                dev);
        }

        //-----------------------------------------------------------------------------
        //! Resets the device.
        //! What this method does is dependent of the accelerator.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FCT_HOST auto reset(
            TDev const & dev)
        -> void
        {
            traits::dev::Reset<
                TDev>
            ::reset(
                dev);
        }
    }
}
