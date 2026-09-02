#include "optimal_shape.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_optimizer, m) {
  m.doc() = "C++ optimized version of get_optimal_shape function";

  // Bind the OptimalShapeResult struct
  py::class_<OptimalShapeResult>(m, "OptimalShapeResult")
      .def(py::init<>())
      .def_readwrite("best_gpu_hour", &OptimalShapeResult::best_gpu_hour)
      .def_readwrite("best_shape", &OptimalShapeResult::best_shape)
      .def_readwrite("best_total_time", &OptimalShapeResult::best_total_time)
      .def("__repr__", [](const OptimalShapeResult &r) {
        return "OptimalShapeResult(gpu_hour=" +
               std::to_string(r.best_gpu_hour) + ", shape=[" +
               std::to_string(r.best_shape[0]) + "," +
               std::to_string(r.best_shape[1]) + "," +
               std::to_string(r.best_shape[2]) + "," +
               std::to_string(r.best_shape[3]) +
               "], time=" + std::to_string(r.best_total_time) + ")";
      });

  // Bind the main optimization function
  m.def("find_optimal_shape", &find_optimal_shape,
        "Find optimal GPU allocation shape", py::arg("current_world_size"),
        py::arg("num_inference_steps"), py::arg("latency_threshold"),
        py::arg("time_per_step_1gpu"), py::arg("time_per_step_2gpu"),
        py::arg("time_per_step_4gpu"), py::arg("time_per_step_8gpu"));

  // Helper function to convert format for compatibility with existing Python
  // code
  m.def(
      "get_optimal_shape_for_scheduler",
      [](int current_world_size, int num_inference_steps,
         double latency_threshold, double time_per_step_1gpu,
         double time_per_step_2gpu, double time_per_step_4gpu,
         double time_per_step_8gpu) -> py::tuple {
        OptimalShapeResult result = find_optimal_shape(
            current_world_size, num_inference_steps, latency_threshold,
            time_per_step_1gpu, time_per_step_2gpu, time_per_step_4gpu,
            time_per_step_8gpu);

        // Convert [r1, r2, r4, r8] format to [(gpu_count, steps), ...] format
        std::vector<py::tuple> optimal_shape_list;
        std::vector<int> gpu_counts = {1, 2, 4, 8};

        for (int i = 0; i < 4; i++) {
          if (result.best_shape[i] > 0) {
            optimal_shape_list.push_back(
                py::make_tuple(gpu_counts[i], result.best_shape[i]));
          }
        }

        // Convert total GPU hours to seconds (divide by 3600 to match original
        // code)
        double total_gpu_hours = result.best_gpu_hour / 3600.0;

        return py::make_tuple(total_gpu_hours, optimal_shape_list,
                              result.best_total_time);
      },
      "Get optimal shape in format compatible with scheduler",
      py::arg("current_world_size"), py::arg("num_inference_steps"),
      py::arg("latency_threshold"), py::arg("time_per_step_1gpu"),
      py::arg("time_per_step_2gpu"), py::arg("time_per_step_4gpu"),
      py::arg("time_per_step_8gpu"));
}
