def assistant_agent(scenario, persona):
    priority = scenario.get("priority", "event1")
    return f"Decision: prioritize {priority} with {persona['style']}."


def calendar_agent(scenario):
    return "Reschedule lower-priority event to next available slot."


def email_agent(scenario, emotion):
    if emotion in ("angry", "stressed"):
        return "Write calm, empathetic message acknowledging urgency."
    return "Write concise professional update with clear next step."


def negotiation_agent(scenario, constraints):
    hard = constraints.get("hard_limit", "no hard limits")
    return f"Negotiate trade-off while respecting constraint: {hard}."


def delegate_agents(scenario, persona, emotion, constraints):
    return {
        "assistant": assistant_agent(scenario, persona),
        "calendar": calendar_agent(scenario),
        "email": email_agent(scenario, emotion),
        "negotiation": negotiation_agent(scenario, constraints)
    }
