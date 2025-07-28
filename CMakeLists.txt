cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(neural_net
    main.cpp
    src/matrix.cpp
    src/layer.cpp
    src/neuralnet.cpp
    src/loss.cpp
    src/utils.cpp
)