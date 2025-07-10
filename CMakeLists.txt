cmake_minimum_required(VERSION 3.29)
project(proyecto)

set(CMAKE_CXX_STANDARD 20)

add_executable(proyecto main.cpp
        neural_network.h
        nn_activation.h
        nn_dense.h
        nn_interfaces.h
        nn_loss.h
        nn_optimizer.h
        tensor.h
)
