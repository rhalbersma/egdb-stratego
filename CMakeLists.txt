#          Copyright Rein Halbersma 2026.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.30)
project(egdb-stratego LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# generic 2- and 3-piece databases with a configurable tweeveldenregel, rank collapsing, retrograde analysis
add_executable(stratego_egdb src/stratego_egdb.cpp)
# 2 lieutenants vs 1 and 2 sergeants without the tweeveldenregel (4 pieces)
add_executable(stratego22 src/stratego22.cpp)

foreach(target stratego_egdb stratego22)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -march=native>)
endforeach()
