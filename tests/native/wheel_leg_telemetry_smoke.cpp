// Standalone ROS smoke test. Supplies synthetic hardware feedback to the real
// state estimator and rmcs_rl mirror. Never creates hardware or CAN.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#include <rclcpp/executors/single_threaded_executor.hpp>

#include "controller/chassis/wheel_leg_state_estimator.cpp"
#include "visualization/wheel_leg_mirror.cpp"

std::string rmcs_executor::Component::initializing_component_name;

namespace rmcs_executor {
// Test-only friend: exercise the production interface binding functions.
class Executor {
public:
    static void pair(std::initializer_list<Component*> sources, Component& consumer) {
        for (auto& input : consumer.input_list_) {
            bool paired = false;
            for (auto* source : sources) {
                auto output = std::find_if(
                    source->output_list_.begin(), source->output_list_.end(),
                    [&](const auto& value) { return value.name == input.name; });
                if (output == source->output_list_.end())
                    continue;
                if (paired || output->type != input.type)
                    throw std::runtime_error("ambiguous/wrong input type: " + input.name);
                input.bind(input.binding, output->binding);
                paired = true;
            }
            if (!paired)
                throw std::runtime_error("unpaired input: " + input.name);
        }
    }
};
} // namespace rmcs_executor

class SyntheticFeedback final : public rmcs_executor::Component {
public:
    using Clock = std::chrono::steady_clock;
    using JointVector = rmcs_core::controller::chassis::JointVector;
    SyntheticFeedback() {
        register_output("/predefined/timestamp", timestamp, Clock::now());
        register_output("/chassis/reset_count", reset, std::size_t{0});
        register_output("/chassis/control_enable", control_enabled, false);
        register_output("/wheel_leg/dm_control_ready", dm_ready, false);
        for (std::size_t i = 0; i < 6; ++i) {
            const auto prefix =
                std::string{"/wheel_leg/"} + rmcs_core::controller::chassis::kMotorNames[i];
            register_output(prefix + "/angle", angle[i], static_cast<double>(i) + .25);
            register_output(prefix + "/velocity", velocity[i], static_cast<double>(i) + .5);
            register_output(prefix + "/feedback_steady_ns", feedback_ns[i], std::uint64_t{0});
        }
        register_output("/wheel_leg/imu/quaternion", orientation, Eigen::Quaterniond::Identity());
        register_output("/wheel_leg/imu/angular_velocity", gyro, Eigen::Vector3d{.1, -.2, .3});
        register_output("/wheel_leg/imu/last_steady_ns", imu_ns, std::uint64_t{0});
        register_output("/wheel_leg/feedback_fresh", fresh, true);
    }
    void update() override {
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                .count();
        *imu_ns = ns;
        for (auto& value : feedback_ns)
            *value = ns;
    }
    OutputInterface<Clock::time_point> timestamp;
    OutputInterface<std::size_t> reset;
    std::array<OutputInterface<double>, 6> angle, velocity;
    std::array<OutputInterface<std::uint64_t>, 6> feedback_ns;
    OutputInterface<Eigen::Quaterniond> orientation;
    OutputInterface<Eigen::Vector3d> gyro;
    OutputInterface<std::uint64_t> imu_ns;
    OutputInterface<bool> fresh, control_enabled, dm_ready;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rmcs_executor::Component::initializing_component_name = "synthetic_feedback";
    SyntheticFeedback feedback;
    rmcs_executor::Component::initializing_component_name = "wheel_leg_state_estimator";
    rmcs_core::controller::chassis::WheelLegStateEstimator estimator;
    rmcs_executor::Executor::pair({&feedback}, estimator);
    rmcs_executor::Component::initializing_component_name = "wheel_leg_mirror";
    auto publisher = std::make_shared<rmcs_rl::WheelLegMirrorComponent>();
    rmcs_executor::Executor::pair({&feedback, &estimator}, *publisher);
    std::string stream_pose;
    double stream_seconds = 10.;
    bool hand_moved = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string{argv[i]} == "--hand-moved")
            hand_moved = true;
        else if (std::string{argv[i]} == "--stream-pose" && i + 1 < argc)
            stream_pose = argv[++i];
        else if (std::string{argv[i]} == "--stream-seconds" && i + 1 < argc)
            stream_seconds = std::stod(argv[++i]);
    }
    if (!stream_pose.empty()) {
        std::ifstream input{stream_pose};
        for (auto& value : feedback.angle)
            input >> *value;
        if (!input || !std::isfinite(stream_seconds) || stream_seconds <= 0.)
            throw std::runtime_error("invalid synthetic stream fixture");
        for (auto& value : feedback.velocity)
            *value = 0.;
        SyntheticFeedback::JointVector initial;
        for (std::size_t i = 0; i < initial.size(); ++i)
            initial[i] = *feedback.angle[i];
        *feedback.gyro = Eigen::Vector3d::Zero();
        rclcpp::executors::SingleThreadedExecutor stream_executor;
        stream_executor.add_node(publisher);
        const auto start = SyntheticFeedback::Clock::now();
        std::cout << "SYNTHETIC ROS FIXTURE streaming; no hardware created" << std::endl;
        while (std::chrono::duration<double>(SyntheticFeedback::Clock::now() - start).count()
               < stream_seconds) {
            *feedback.timestamp = SyntheticFeedback::Clock::now();
            if (hand_moved) {
                const double elapsed =
                    std::chrono::duration<double>(*feedback.timestamp - start).count();
                for (std::size_t i = 0; i < 6; ++i) {
                    const double direction = i == 2 || i == 3 || i == 5 ? -1. : 1.;
                    *feedback.angle[i] = initial[i] + direction * .2 * std::sin(elapsed);
                    *feedback.velocity[i] = direction * .2 * std::cos(elapsed);
                }
                *feedback.orientation = Eigen::Quaterniond{
                    Eigen::AngleAxisd{.4 * elapsed, Eigen::Vector3d::UnitZ()}
                    * Eigen::AngleAxisd{.6 * std::sin(elapsed), Eigen::Vector3d::UnitY()}
                    * Eigen::AngleAxisd{2.7 * std::sin(.4 * elapsed), Eigen::Vector3d::UnitX()}};
                *feedback.gyro = Eigen::Vector3d{20., -30., 40.};
                // Individual sensor timestamps remain fresh: the estimator's
                // aggregate invalid flag must not hide the raw measured pose.
                *feedback.fresh = elapsed < .5 * stream_seconds;
            }
            feedback.update();
            estimator.update();
            publisher->update();
            stream_executor.spin_some();
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        std::cout << "{\"source\":\"SYNTHETIC ROS FIXTURE\",\"hardware_created\":false}\n";
        rclcpp::shutdown();
        return 0;
    }
    auto observer = std::make_shared<rclcpp::Node>("telemetry_smoke_observer");
    const auto qos = rclcpp::QoS{20}.best_effort();
    std::map<std::int64_t, unsigned> topics;
    unsigned invalid = 0, reset_frames = 0, disarmed = 0;
    std::vector<double> observed_angles, observed_rotations;
    const auto stamp = [&](const auto& message, unsigned bit) {
        assert(message.header.frame_id == "chassis_body");
        topics[rclcpp::Time{message.header.stamp}.nanoseconds()] |= bit;
    };
    auto joints = observer->create_subscription<sensor_msgs::msg::JointState>(
        "/wheel_leg/telemetry/joint_states", qos, [&](const sensor_msgs::msg::JointState& m) {
            stamp(m, 1);
            assert(m.name.size() == 6 && m.position.size() == 6 && m.velocity.size() == 6);
            for (std::size_t i = 0; i < 6; ++i) {
                assert(m.name[i] == rmcs_core::controller::chassis::kMotorNames[i]);
                assert(std::abs(m.position[i] - (static_cast<double>(i) + m.position[0])) < 1e-12);
            }
            observed_angles.push_back(m.position[0]);
        });
    auto imu = observer->create_subscription<sensor_msgs::msg::Imu>(
        "/wheel_leg/telemetry/imu_body", qos, [&](const sensor_msgs::msg::Imu& m) {
            stamp(m, 2);
            assert(m.angular_velocity.y == -.2);
            observed_rotations.push_back(m.orientation.x);
            assert(m.linear_acceleration_covariance[0] == -1.);
        });
    auto gravity = observer->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/wheel_leg/telemetry/rl_projected_gravity", qos,
        [&](const geometry_msgs::msg::Vector3Stamped& m) {
            stamp(m, 4);
            assert(std::isfinite(m.vector.z));
        });
    auto gyro = observer->create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/wheel_leg/telemetry/rl_angular_velocity", qos,
        [&](const geometry_msgs::msg::Vector3Stamped& m) {
            stamp(m, 8);
            assert(m.vector.x == .1);
        });
    auto diagnostics = observer->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        "/wheel_leg/telemetry/diagnostics", qos,
        [&](const diagnostic_msgs::msg::DiagnosticArray& m) {
            stamp(m, 16);
            assert(m.status.size() == 1);
            std::map<std::string, std::string> values;
            for (const auto& entry : m.status[0].values)
                values[entry.key] = entry.value;
            assert(values["schema"] == "wheel_leg_telemetry_v1");
            assert(values["control_enabled"] == "false" && values["dm_control_ready"] == "false");
            ++disarmed;
            assert(
                values["calibration_sha256"]
                == rmcs_core::controller::chassis::full_vehicle_calibration110::kCalibrationSha256);
            assert(
                values["calibration_model_spec_sha256"]
                == rmcs_core::controller::chassis::full_vehicle_calibration110::kModelSpecSha256);
            assert(std::stod(values["model_offset_rad/left_hip_joint"]) == 1.6);
            assert(std::stod(values["model_offset_rad/right_hip_joint"]) == -1.6);
            assert(std::stod(values["imu_age_s"]) >= 0.);
            assert(std::stod(values["imu_age_s"]) < .04);
            invalid += values["feedback_fresh"] == "false";
            reset_frames += values["reset_count"] == "1";
        });
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(observer);
    executor.add_node(publisher);
    for (int i = 0; i < 100; ++i) {
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    const auto start = SyntheticFeedback::Clock::now();
    for (int tick = 0; tick <= 40; ++tick) {
        *feedback.timestamp = start + std::chrono::milliseconds{tick};
        *feedback.fresh = tick < 20;
        *feedback.reset = tick >= 33 ? 1 : 0;
        for (std::size_t i = 0; i < feedback.angle.size(); ++i)
            *feedback.angle[i] = static_cast<double>(i) + .25 + .01 * tick;
        *feedback.orientation =
            Eigen::Quaterniond{Eigen::AngleAxisd{.01 * tick, Eigen::Vector3d::UnitX()}};
        feedback.update();
        estimator.update();
        publisher->update();
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    for (int i = 0; i < 20; ++i) {
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    assert(topics.size() == 5); // ticks 0,10,20,30,33; reset publishes immediately.
    for (const auto& [time, mask] : topics)
        assert(mask == 31);
    assert(invalid == 3 && reset_frames == 1);
    assert(disarmed == 5 && observed_angles.size() == 5 && observed_rotations.size() == 5);
    assert(observed_angles.back() > observed_angles.front());
    assert(observed_rotations.back() > observed_rotations.front());
    std::cout
        << "{\"complete_snapshots\":5,\"topics_per_snapshot\":5,\"invalid_feedback_frames\":"
           "3,\"reset_frames\":1,\"disarmed_moving_snapshots\":5,\"hardware_created\":false}\n";
    rclcpp::shutdown();
}
