# 指定CMake最低版本要求（兼容3.12到3.26）
cmake_minimum_required(VERSION 3.12...3.26)

# 处理构建类型（单配置生成器，如Makefile）：默认设为Release（优化编译）
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release default")
endif()

# 定义项目名称为tinyrenderer，指定编程语言为C++
project(tinyrenderer LANGUAGES CXX)

# 设置C++标准为C++20（编译时启用C++20特性）
set(CMAKE_CXX_STANDARD 20)

# 可选功能：启用include-what-you-use（检查头文件冗余/缺失）
option(iwyu "Run include-what-you-use")
if(iwyu)
  find_program(IWYU_EXE NAMES include-what-you-use REQUIRED)
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_EXE})
endif()

# 针对Clang/GCC/Intel编译器，添加编译选项-Wall（显示所有警告）
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
  add_compile_options(-Wall)
endif()

# 查找OpenMP（多核并行库），当前代码暂未用到
find_package(OpenMP COMPONENTS CXX)

# ================================ 真正开始设置 ================================

# 生成一些库文件，用于后续target_link_libraries
add_library(tgaimage STATIC
  tgaimage.cpp
)

add_library(renderer STATIC
  renderer.cpp
)

# 生成可执行文件，并将.cpp生成的库文件链接
set(SOURCES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(renderer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
)

target_link_libraries(${PROJECT_NAME} PRIVATE tgaimage renderer $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

file(GENERATE OUTPUT .gitignore CONTENT "*")
