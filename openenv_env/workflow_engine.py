def run_workflow(scenario, predicted_conflict=None, drift_policy=None):
    steps = []

    steps.append("1. Analyze the conflict between tasks")
    if predicted_conflict and predicted_conflict.get("likely_conflict_next_48h"):
        steps.append("1b. Trigger pre-emptive action before conflict materializes")
    steps.append("2. Identify which task has higher priority")
    if drift_policy:
        steps.append(f"2b. Apply current policy drift rules: {drift_policy.get('hard_limit')}")
    steps.append("3. Decide whether to attend or reschedule")
    steps.append("4. Generate a polite response")
    steps.append("5. Execute delegated tool actions and verify outcomes")

    return steps