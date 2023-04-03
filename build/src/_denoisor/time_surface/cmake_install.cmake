# Install script for directory: /home/kuga/Workspace/tmp-emlb-v2/src/_denoisor/time_surface

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so"
         RPATH "/usr/local/share/dv/modules:/opt/inivation/boost/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/kuga/Workspace/tmp-emlb-v2/build/../modules" TYPE SHARED_LIBRARY FILES "/home/kuga/Workspace/tmp-emlb-v2/modules/Km_time_surface.so")
  if(EXISTS "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so"
         OLD_RPATH "/opt/inivation/boost/lib::::::::::::::::::::::::::::"
         NEW_RPATH "/usr/local/share/dv/modules:/opt/inivation/boost/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/kuga/Workspace/tmp-emlb-v2/build/../modules/Km_time_surface.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()
