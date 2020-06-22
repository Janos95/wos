cmake_minimum_required(VERSION 3.16)
project(kdtree)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

find_package(Magnum CONFIG REQUIRED COMPONENTS Sdl2Application GL SceneGraph Shaders Trade MeshTools Primitives DebugTools)
find_package(MagnumPlugins CONFIG REQUIRED AssimpImporter)

add_subdirectory(scoped_timer)

corrade_add_resource(wos_Rcs resources.conf)

add_executable(app
        application.cpp
        arc_ball.cpp
        upload.hpp
        upload.cpp
        subdivision.hpp
        subdivision.cpp
        ${wos_Rcs}
        )

target_link_libraries(app PUBLIC
        Magnum::Magnum
        Magnum::Application
        Magnum::GL
        Magnum::SceneGraph
        Magnum::Shaders
        Magnum::Trade
        Magnum::MeshTools
        Magnum::DebugTools
        Magnum::Primitives
        scoped_timer::scoped_timer
        )


