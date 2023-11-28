set(TARGET ffi)

add_library(${TARGET} STATIC
    ffi.h
    ffi.cpp
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()

target_include_directories(${TARGET} PUBLIC .)
target_link_libraries(${TARGET} PRIVATE common llama)
target_compile_features(${TARGET} PUBLIC cxx_std_11)

set_target_properties(ffi PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/ffi.h)
install(TARGETS ffi LIBRARY PUBLIC_HEADER)

if (LLAMA_STATIC)
    add_link_options(-static)
    if (MINGW)
        add_link_options(-static-libgcc -static-libstdc++)
    endif()
endif()