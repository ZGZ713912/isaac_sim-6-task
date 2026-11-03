"""Allow getup translation; standing holds the actual completion position."""
from copy import deepcopy


CONTRACT = dict(
    schema="wheel-leg-getup-translation-v1",
    mode="free_until_completion",
    rising_position_cost="disabled",
    rising_excursion_abort="record_only",
    standing_reference="actual_completion_xy",
)


def validate(contract):
    if contract != CONTRACT:
        raise ValueError("Use the explicit getup translation contract")
    return deepcopy(contract)


def enabled(plan):
    return plan.get("getup_translation", {}).get("mode") == "free_until_completion"
