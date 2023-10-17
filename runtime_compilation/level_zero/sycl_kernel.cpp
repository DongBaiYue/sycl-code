#include <CL/sycl.hpp>
using namespace sycl;

int func() {
    queue q;
    q.submit([&](sycl::handler &h) {
        sycl::stream os(1024, 768, h);
        h.parallel_for(32, [=](sycl::id<1> i) {
            os<<"A";
        });
    }).wait();
}