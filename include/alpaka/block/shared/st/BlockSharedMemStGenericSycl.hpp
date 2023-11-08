/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/st/Traits.hpp"
#include "alpaka/core/Sycl.hpp"

#include <cstddef>
#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

constexpr size_t MaxKeys = 100;

enum class TypeIdentifier {
    Unknown,
    Int,
    Unsigned,
    UnsignedShort,
    Float,
    Double
};

template <typename T>
struct GetTypeIdentifier {
    static constexpr TypeIdentifier value = TypeIdentifier::Unknown;
};

template <>
struct GetTypeIdentifier<int> {
    static constexpr TypeIdentifier value = TypeIdentifier::Int;
};

template <>
struct GetTypeIdentifier<std::uint8_t> {
    static constexpr TypeIdentifier value = TypeIdentifier::UnsignedShort;
};

template <>
struct GetTypeIdentifier<unsigned int> {
    static constexpr TypeIdentifier value = TypeIdentifier::Unsigned;
};

template <>
struct GetTypeIdentifier<float> {
    static constexpr TypeIdentifier value = TypeIdentifier::Float;
};

template <>
struct GetTypeIdentifier<double> {
    static constexpr TypeIdentifier value = TypeIdentifier::Double;
};

template<int TDim>
class KeyValueStore {
public:
    KeyValueStore() {
        for (size_t i = 0; i < MaxKeys; ++i) {
            keys[i] = MaxKeys*100; //FIXME_ -1;  // Initialize keys with -1 (indicating empty slot)
            values[i] = TypeIdentifier::Unknown; // Initialize values
            pointers[i] = nullptr;
        }
    }

    template<typename T>
    void insert(int key, T* ptr) {
        size_t index = findEmptySlot();
        printf("inserting %d key in postition %d\n", key, index);
        if (index < MaxKeys) {
            keys[index] = key;
            values[index] = GetTypeIdentifier<T>::value;
            pointers[index] = ptr;
        }
    }

    template<typename T>
    T& getValue(int key) {
        printf("getting %d key \n", key);
        for (size_t i = 0; i < MaxKeys; ++i) {
            if (keys[i] == key and GetTypeIdentifier<T>::value == values[i]) {
                printf("getting old memory\n");
                return *(static_cast<T*>(pointers[i]));
            }
        }
        // allocate new and insert key in map
        printf("allocating new memeory\n");
        auto my_group = sycl::ext::oneapi::experimental::this_group<TDim>();
        auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(my_group);
        insert<T>(key, static_cast<T*>(ptr.get()));
        return *ptr; 
    }

private:
    size_t findEmptySlot() {
        for (size_t i = 0; i < MaxKeys; ++i) {
            if (keys[i] == MaxKeys*100) {
                return i;
            }
        }
        return MaxKeys; // No empty slots available
    }

    size_t keys[MaxKeys];
    TypeIdentifier values[MaxKeys];
    void* pointers[MaxKeys];
};


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
        mutable KeyValueStore<TDim::value> keyValueMap;
    };
} // namespace alpaka

namespace alpaka::trait
{

    template<typename T, std::size_t TUniqueId, typename TDim>
    struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl<TDim>>
    {
        static auto declareVar(BlockSharedMemStGenericSycl<TDim> const& smem) -> T&
        {
            return smem.keyValueMap.template getValue<T>(TUniqueId);
            //auto shMemBuff = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(smem.my_item_st.get_group());
            //smem.keyValueMap[key] = (shMemBuff.get());
            //return *(shMemBuff.get());
            
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
