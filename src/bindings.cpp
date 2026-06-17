#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "trellis.hpp"

namespace py = pybind11;

static std::vector<int> symbols_from_array(py::array_t<int> arr) {
    auto buf = arr.request();
    const int* ptr = static_cast<const int*>(buf.ptr);
    return std::vector<int>(ptr, ptr + buf.size);
}

static std::vector<uint8_t> mask_from_array(py::array_t<uint8_t> arr) {
    auto buf = arr.request();
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    return std::vector<uint8_t>(ptr, ptr + buf.size);
}

PYBIND11_MODULE(achmm_trellis, m) {
    m.doc() = "Active Context HMM forward-backward trellis (C++ core)";

    py::class_<achmm::ACHMMParams>(m, "ACHMMParams")
        .def(py::init<>())
        .def_readwrite("K", &achmm::ACHMMParams::K)
        .def_readwrite("D", &achmm::ACHMMParams::D)
        .def_readwrite("dirichlet_alpha", &achmm::ACHMMParams::dirichlet_alpha)
        .def_readwrite("occupancy_tau", &achmm::ACHMMParams::occupancy_tau)
        .def_readwrite("convergence_delta", &achmm::ACHMMParams::convergence_delta)
        .def_readwrite("max_iterations", &achmm::ACHMMParams::max_iterations);

    py::class_<achmm::TrainResult>(m, "TrainResult")
        .def_readonly("log_likelihood", &achmm::TrainResult::log_likelihood)
        .def_readonly("iterations", &achmm::TrainResult::iterations)
        .def_readonly("converged", &achmm::TrainResult::converged)
        .def_readonly("flops", &achmm::TrainResult::flops);

    py::class_<achmm::ACHMMTrellis>(m, "ACHMMTrellis")
        .def(py::init<const achmm::ACHMMParams&>())
        .def("set_random_seed", &achmm::ACHMMTrellis::set_random_seed)
        .def("fit", [](achmm::ACHMMTrellis& self, py::array_t<int> symbols,
                       py::array_t<uint8_t> mask) {
            return self.fit(symbols_from_array(symbols), mask_from_array(mask), {}, {}, {});
        })
        .def("score", [](achmm::ACHMMTrellis& self, py::array_t<int> symbols,
                         py::array_t<uint8_t> mask) {
            return self.score(symbols_from_array(symbols), mask_from_array(mask));
        })
        .def("num_states", &achmm::ACHMMTrellis::num_states)
        .def("context_depth", &achmm::ACHMMTrellis::context_depth)
        .def_property_readonly(
            "pi",
            [](const achmm::ACHMMTrellis& self) { return self.pi(); })
        .def_property_readonly(
            "A",
            [](const achmm::ACHMMTrellis& self) { return self.A(); })
        .def_property_readonly(
            "B",
            [](const achmm::ACHMMTrellis& self) { return self.B(); });
}
