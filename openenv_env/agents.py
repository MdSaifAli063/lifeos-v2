def conflict_resolution_agent(scenario, persona):
    priority = scenario.get("priority", "event1")
    return f"Resolve conflict by prioritizing {priority} with {persona['style']}."


def calendar_agent(scenario):
    return "Reschedule lower-priority event to next available slot."


def email_agent(scenario, emotion):
    if emotion in ("angry", "stressed"):
        return "Write calm, empathetic message acknowledging urgency."
    return "Write concise professional update with clear next step."


def negotiation_agent(scenario, constraints):
    hard = constraints.get("hard_limit", "no hard limits")
    return f"Negotiate trade-off while respecting constraint: {hard}."


def delegation_agent(scenario):
    event1 = scenario.get("event1", "task A")
    event2 = scenario.get("event2", "task B")
    return f"Delegate preparatory subtasks for {event1} and {event2} to reduce overload."


def memory_agent(persona):
    return f"Retrieve long-term preference priors for persona: {persona.get('name', 'unknown')}."


def emotion_detection_agent(emotion):
    return f"Detected emotional context: {emotion}. Calibrate tone risk and urgency."


def delegate_agents(scenario, persona, emotion, constraints):
    return {
        "conflict_resolution": conflict_resolution_agent(scenario, persona),
        "calendar": calendar_agent(scenario),
        "email": email_agent(scenario, emotion),
        "negotiation": negotiation_agent(scenario, constraints),
        "delegation": delegation_agent(scenario),
        "memory": memory_agent(persona),
        "emotion_detection": emotion_detection_agent(emotion)
    }
