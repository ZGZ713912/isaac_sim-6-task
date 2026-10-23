#include "model_geometry107.hpp"
#include "standing_reference_v3.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main() {
    const auto& geometry = wheel_leg_policy::frozen_geometry107();
    wheel_leg_policy::StandingReferenceV3 reference(geometry);
    std::uint64_t tick;
    int reset, enabled;
    std::cout << std::setprecision(9);
    const auto read_float = [] {
        std::string token;
        std::cin >> token;
        return std::strtof(token.c_str(), nullptr);
    };
    while (std::cin >> tick >> reset >> enabled) {
        std::array<float, 6> position;
        std::array<float, 35> observation;
        for (auto& value : position)
            value = read_float();
        for (auto& value : observation)
            value = read_float();
        if (!std::cin)
            return 2;
        if (reset)
            reference.reset(tick);
        const wheel_leg_policy::Vector3 gravity{observation[7], observation[8], observation[9]};
        const auto coordinate = wheel_leg_policy::wheel_angle_coordinate(geometry, position, gravity);
        const auto output = reference.sample(tick, position, observation, enabled);
        const auto& state = reference.state();
        std::cout << coordinate.valid << ' ' << coordinate.coordinate_m[0] << ' ' << coordinate.coordinate_m[1]
                  << ' ' << state.active << ' ' << state.anchor_m[0] << ' ' << state.anchor_m[1]
                  << ' ' << state.filtered_m[0] << ' ' << state.filtered_m[1] << ' ' << state.severe_reset_count
                  << ' ' << output[0] << ' ' << output[1] << '\n';
    }
}
