def reward_function(output, scenario, context=None):
    output = output.lower()
    context = context or {}
    reward = 0
    breakdown = {
        "priority_alignment": 0,
        "rescheduling_quality": 0,
        "tone_quality": 0,
        "policy_compliance": 0,
        "prediction_bonus": 0,
        "tool_execution_bonus": 0,
        "penalty": 0
    }

    # reschedule is best strategy
    if "reschedule" in output:
        reward += 20
        breakdown["rescheduling_quality"] += 20

    # correct priority
    if scenario["priority"] == "event1" and scenario["event1"].lower() in output:
        reward += 15
        breakdown["priority_alignment"] += 15
    elif scenario["priority"] == "event2" and scenario["event2"].lower() in output:
        reward += 15
        breakdown["priority_alignment"] += 15

    # politeness
    if any(word in output for word in ["sorry", "apologize", "thanks"]):
        reward += 5
        breakdown["tone_quality"] += 5

    policy = context.get("policy", {})
    if policy.get("quiet_hours") and any(word in output for word in ["reschedule", "tomorrow", "next slot"]):
        reward += 6
        breakdown["policy_compliance"] += 6

    prediction = context.get("prediction", {})
    if prediction.get("likely_conflict_next_48h") and "pre-empt" in output:
        reward += 4
        breakdown["prediction_bonus"] += 4

    tool_results = context.get("tool_results", {})
    if tool_results:
        success_count = sum(1 for value in tool_results.values() if value.get("ok"))
        tool_bonus = min(success_count * 2, 8)
        reward += tool_bonus
        breakdown["tool_execution_bonus"] += tool_bonus

    # penalty
    if reward == 0:
        reward = -10
        breakdown["penalty"] = -10

    return reward, breakdown