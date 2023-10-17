#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <iostream>

int LoadSPIRVInLevelzero(const sycl::context& sycl_context, const sycl::device& sycl_device, sycl::queue& sycl_queue, 
                            const char* spir, size_t size, const char* kernel_name, sycl::kernel** sycl_kernel){
    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
    // create ze module
    ze_module_handle_t ze_module = nullptr;
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 size,
                                 (const uint8_t*)spir,
                                 nullptr,
                                 nullptr};
    ze_module_build_log_handle_t buildlog;
    ze_result_t status = zeModuleCreate(ze_context, ze_device, &moduleDesc, &ze_module, &buildlog);
    if (status != 0) {
        size_t szLog = 0;
        zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

        std::unique_ptr<char> PLogs(new char[szLog]);
        zeModuleBuildLogGetString(buildlog, &szLog, PLogs.get());
        std::string PLog(PLogs.get());
        std::cout << "L0 error " << status << ": " << PLog << std::endl;
    }
    zeModuleBuildLogDestroy(buildlog);
    // create ze kernel
    ze_kernel_handle_t ze_kernel = nullptr;
    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernelDesc.pKernelName = kernel_name;
    zeKernelCreate(ze_module, &kernelDesc, &ze_kernel);
    // create sycl kernel
    sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>
        ({ze_module}, sycl_context);
    auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>({kernel_bundle, ze_kernel}, sycl_context);
    *sycl_kernel = new sycl::kernel(kernel);
    // launch sycl kernel
    auto sycl_global_range = sycl::range<3>(1, 1, 1);
    auto sycl_local_range = sycl::range<3>(1, 1, 1);
    sycl::nd_range<3> sycl_nd_range(sycl::nd_range<3>(sycl_global_range, sycl_local_range));
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range, *sycl_kernel);
    });
    return 0;
}

int main(){
    auto platforms = sycl::platform::get_platforms();
    std::string key = "Level-Zero";
    sycl::context sycl_context;
    sycl::device sycl_device;
    sycl::queue sycl_queue;
    for (auto &platform : platforms){
        std::string name = platform.get_info<sycl::info::platform::name>();
        std::cout<<name<<std::endl;
        if (name.find(key) == std::string::npos)
            continue;
        sycl_device = platform.get_devices(sycl::info::device_type::gpu)[0];
        sycl_context = sycl::context(sycl_device);
        sycl_queue = sycl::queue(sycl_context, sycl_device);
    }
    std::string spv_path = "kernel.spv";
    std::ifstream file(spv_path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        auto length = file.tellg();
        file.seekg(0, file.beg);
        std::unique_ptr<char[]> spirvInput(new char[length]);
        file.read(spirvInput.get(), length);
        LoadSPIRVInLevelzero(sycl_context, sycl_device, sycl_queue, spirvInput.get(), length, "kernel", nullptr);
    }
    return 0;
}