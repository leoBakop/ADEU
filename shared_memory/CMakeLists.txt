cmake_minimum_required(VERSION 3.1.0)

project(shared_memory_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
  set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()
include_directories(src)


find_package(Qt5 COMPONENTS Widgets Concurrent REQUIRED)

add_executable(${CMAKE_PROJECT_NAME}
  src/ui/sharedMem.h
  src/ui/sharedMem.cpp
  src/ui/main.cpp
  src/ui/abstract_ipc.h
  src/ui/abstract_ipc.cpp
  src/ui/consumer_ipc.h
  src/ui/consumer_ipc.cpp
  src/ui/producer_ipc.h
  src/ui/producer_ipc.cpp
)


target_link_libraries(${CMAKE_PROJECT_NAME} Qt5::Widgets Qt5::Concurrent)
