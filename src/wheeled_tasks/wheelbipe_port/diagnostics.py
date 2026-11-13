"""Grouped logging only; these reductions never compute rewards or actions.

CUDA atomic sums can differ by float32 roundoff. An explicitly deterministic
run retains masked reductions; no global determinism setting is changed.
"""
from __future__ import annotations
import torch


def grouped_sum(roles,values,count):
    if values.ndim == 1:
        values = values[:,None]
    values = values.detach().float()
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        return torch.stack([values[roles == i].sum(0) for i in range(count)])
    return values.new_zeros(count,values.shape[1]).index_add_(0,roles,values)


def grouped_max(roles,values,count):
    values = values.detach()
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        rows = [values[roles == i] for i in range(count)]
        return torch.stack([row.max() if row.numel() else values.new_zeros(()) for row in rows])
    result = values.new_full((count,),-torch.inf)
    result.scatter_reduce_(0,roles,values,reduce="amax",include_self=True)
    return torch.where(torch.isfinite(result),result,0.)


def level_statistics(roles,counts,roll,pitch,radius,radius_max,flight,age,difficulty,reset_grace):
    r,p = roll.abs(),pitch.abs()
    tilt = torch.maximum(r,p)
    ordinary = ~flight & (age >= reset_grace)
    rows = torch.stack((r,p,(tilt > 3.).float(),(tilt > 6.).float(),radius,
        ((tilt > 3.) & ordinary).float(),((tilt > 6.) & ordinary).float(),ordinary.float(),
        ((tilt > 6.) & flight).float(),flight.float(),difficulty),-1)
    sums = grouped_sum(roles,rows,len(counts))
    means = sums/counts[:,None]
    normal = sums[:,5:7]/sums[:,7,None].clamp_min(1.)
    flying = sums[:,8]/sums[:,9].clamp_min(1.)
    return dict(roll_abs_mean_deg=means[:,0],pitch_abs_mean_deg=means[:,1],
        tilt_over_3deg_fraction=means[:,2],tilt_over_6deg_fraction=means[:,3],
        radius_mean_m=means[:,4],radius_max_m=grouped_max(roles,radius_max,len(counts)),
        ordinary_tilt_over_3deg_fraction=normal[:,0],ordinary_tilt_over_6deg_fraction=normal[:,1],
        flight_landing_tilt_over_6deg_fraction=flying,difficulty_mean=means[:,10])
