// Exercise the actual RMCS policy loader without ROS or motor interfaces.
#include "policy/policy_model.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 5)
        return 1;
    try {
        rmcs_rl::PolicyConfig config;
        config.path = argv[1];
        config.input_name = "obs";
        config.output_name = "actions";
        config.model_type = "wheel_leg_v7";
        config.observation_size = std::stoul(argv[2]);
        config.action_size = 6;
        config.feature_size = config.observation_size;
        config.sequence_length = 1;
        config.expected_model_spec_sha256 = argv[3];
        config.expected_leg_winding_version = argv[4];
        config.obs_clip = 100.;
        rmcs_rl::PolicyModel model{config};
        std::array<double, 35> input{};
        std::array<double, 6> output{};
        while (std::cin >> input[0]) {
            for (std::size_t i = 1; i < input.size(); ++i)
                if (!(std::cin >> input[i]))
                    throw std::runtime_error("incomplete observation fixture");
            std::string error;
            if (!model.run(input, output, error))
                throw std::runtime_error(error);
            for (double value : output)
                std::cout << std::setprecision(17) << value << ' ';
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
