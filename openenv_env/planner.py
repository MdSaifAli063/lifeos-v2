def decompose_goal(scenario):
    event1 = scenario.get("event1", "Task A")
    event2 = scenario.get("event2", "Task B")
    return [
        f"Analyze conflict between {event1} and {event2}",
        "Select winning priority under current policy",
        "Delegate scheduling + communication",
        "Draft emotionally-aware response"
    ]


def replan_if_needed(plan, drift_event):
    if not drift_event:
        return plan
    adjusted = list(plan)
    adjusted.insert(1, f"Adapt to policy drift: {drift_event}")
    return adjusted
