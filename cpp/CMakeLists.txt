cmake_minimum_required(VERSION 3.4...3.18)
set (PYTHON_EXECUTABLE "/Users/chrisbolig/anaconda3/bin/python")

project(kalmanfilter_cpp)
add_subdirectory(pybind11)
include_directories(Eigen)

pybind11_add_module(kalmanfilter_cpp main.cpp kf.h)
