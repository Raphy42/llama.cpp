set(TARGET ffi)

add_library(${TARGET} STATIC
    ffi.h
    ffi.cpp
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE llama)