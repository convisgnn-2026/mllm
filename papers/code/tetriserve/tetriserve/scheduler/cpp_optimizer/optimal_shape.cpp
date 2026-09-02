#include <limits>
#include <tuple>
#include <vector>

struct OptimalShapeResult {
  double best_gpu_hour;
  std::vector<int> best_shape; // [r1, r2, r4, r8]
  double best_total_time;

  OptimalShapeResult()
      : best_gpu_hour(std::numeric_limits<double>::infinity()),
        best_shape({0, 0, 0, 0}),
        best_total_time(std::numeric_limits<double>::infinity()) {}
};

/**
 * Find optimal GPU allocation shape using brute force optimization.
 *
 * @param current_world_size Number of available GPUs
 * @param num_inference_steps Total number of inference steps to execute
 * @param latency_threshold Maximum allowed latency in seconds
 * @param time_per_step_1gpu Time per step when using 1 GPU (seconds)
 * @param time_per_step_2gpu Time per step when using 2 GPUs (seconds)
 * @param time_per_step_4gpu Time per step when using 4 GPUs (seconds)
 * @param time_per_step_8gpu Time per step when using 8 GPUs (seconds)
 * @return OptimalShapeResult containing optimal configuration
 */
OptimalShapeResult
find_optimal_shape(int current_world_size, int num_inference_steps,
                   double latency_threshold, double time_per_step_1gpu,
                   double time_per_step_2gpu, double time_per_step_4gpu,
                   double time_per_step_8gpu) {
  OptimalShapeResult result;

  // Early exit for edge cases
  if (num_inference_steps <= 0 || current_world_size <= 0) {
    return result;
  }

  // Optimize loop order: start with more constrained variables (r8, r4, r2, r1)
  // This allows for early termination when constraints are not met
  for (int r1 = 0; r1 <= num_inference_steps; r1++) {
    for (int r2 = 0; r2 <= num_inference_steps - r1; r2++) {
      for (int r4 = 0; r4 <= num_inference_steps - r1 - r2; r4++) {

        // Skip if invalid time for 1 GPU
        if (r1 > 0 && time_per_step_1gpu < 0)
          continue;

        // Skip if we don't have enough GPUs or invalid time
        if (r2 > 0 && (current_world_size < 2 || time_per_step_2gpu < 0))
          continue;

        // Skip if we don't have enough GPUs or invalid time
        if (r4 > 0 && (current_world_size < 4 || time_per_step_4gpu < 0))
          continue;

        int r8 = num_inference_steps - r1 - r2 - r4;

        // Skip if we don't have enough GPUs or invalid time
        if (r8 > 0 && (current_world_size < 8 || time_per_step_8gpu < 0))
          continue;

        // Calculate times for each configuration
        double r1_time = (r1 == 0) ? 0.0 : time_per_step_1gpu * r1;
        double r2_time = (r2 == 0) ? 0.0 : time_per_step_2gpu * r2;
        double r4_time = (r4 == 0) ? 0.0 : time_per_step_4gpu * r4;
        double r8_time = (r8 == 0) ? 0.0 : time_per_step_8gpu * r8;

        double total_time = r1_time + r2_time + r4_time + r8_time;
        double total_gpu_hour =
            r1_time + r2_time * 2 + r4_time * 4 + r8_time * 8;

        // Apply the latency adjustment factor from original code
        // This accounts for pipeline overhead and scheduling delays
        double adjusted_total_time =
            total_time / num_inference_steps * (num_inference_steps + 1);

        if (adjusted_total_time > latency_threshold) {
          // Cannot meet latency requirement
          // If this is the first solution we found, keep it as fallback
          if (result.best_total_time ==
              std::numeric_limits<double>::infinity()) {
            result.best_gpu_hour = total_gpu_hour;
            result.best_shape = {r1, r2, r4, r8};
            result.best_total_time = total_time;
          }
          continue;
        }

        // Among solutions that meet latency, choose minimum GPU hours (economic
        // optimization)
        if (total_gpu_hour < result.best_gpu_hour) {
          result.best_gpu_hour = total_gpu_hour;
          result.best_shape = {r1, r2, r4, r8};
          result.best_total_time = total_time;
        }
      }
    }
  }
  return result;
}
