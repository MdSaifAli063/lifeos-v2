def predict_future_conflict(scenario, memory_preferences):
    event1 = (scenario.get("event1") or "").lower()
    event2 = (scenario.get("event2") or "").lower()
    msg = (scenario.get("email") or "").lower()

    risk = 0.2
    reasons = []

    if any(word in event1 + " " + event2 for word in ["meeting", "deadline", "call", "investor"]):
        risk += 0.2
        reasons.append("high-stakes task detected")
    if any(word in msg for word in ["urgent", "delay", "asap", "complain"]):
        risk += 0.25
        reasons.append("urgency/stress language present")
    if memory_preferences.get("risk_tolerance") == "low":
        risk += 0.1
        reasons.append("user currently low risk tolerance")

    risk = min(risk, 0.95)
    return {
        "risk_score": round(risk, 2),
        "likely_conflict_next_48h": risk >= 0.45,
        "reasons": reasons or ["no strong risk indicators"]
    }
