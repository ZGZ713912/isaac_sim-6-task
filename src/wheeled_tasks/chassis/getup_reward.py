"""Potential shaping with correct terminal and discount semantics."""
import torch


def potential_reward(previous, current, terminated, gamma, gain):
    # A crash has zero continuation potential. It cannot retain the reward for
    # briefly throwing the body upward immediately before an impact failure.
    continuation = torch.where(terminated, 0., current)
    return gain * (gamma * continuation - previous)
