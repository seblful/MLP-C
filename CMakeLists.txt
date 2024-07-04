cmake_minimum_required(VERSION 3.0.0)
project(mlp VERSION 0.1.0 LANGUAGES C)


add_executable(mlp main.c MLP.c mnist.c)