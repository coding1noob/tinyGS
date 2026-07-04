
#pragma once
#include <torch/extension.h>
// #include <cstdio>
// #include <tuple>
// #include <string>

torch::Tensor MultiplyCUDA(torch::Tensor a1, torch::Tensor a2);