clang++ -fsycl -fsycl-device-only -fno-sycl-use-bitcode -o kernel.spv sycl_kernel.cpp
spirv-dis kernel.spv -o kernel.spvasm