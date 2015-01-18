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

#include <alpaka/traits/Acc.hpp>    // GetAccName

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The threads accelerator.
    //-----------------------------------------------------------------------------
    namespace threads
    {
        namespace detail
        {
            // forward declarations
            class AccThreads;
        }
    }
    using AccThreads = threads::detail::AccThreads;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The serial accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                threads::detail::AccThreads>
            {
                static std::string getAccName()
                {
                    return "AccThreads";
                }
            };
        }
    }
}
