def calendar_tool(action_plan):
    if "reschedule" in action_plan.lower():
        return {"ok": True, "slot_found": "tomorrow 7:30 PM", "latency_ms": 120}
    return {"ok": True, "slot_found": "no change", "latency_ms": 65}


def email_tool(action_plan):
    polite = any(word in action_plan.lower() for word in ["sorry", "thanks", "appreciate"])
    return {"ok": True, "polite_score": 0.9 if polite else 0.45, "latency_ms": 80}


def rides_tool(action_plan):
    needed = any(word in action_plan.lower() for word in ["travel", "commute", "ride"])
    return {"ok": True, "booked": needed, "eta_min": 14 if needed else 0}


def shopping_tool(action_plan):
    needed = "buy" in action_plan.lower() or "purchase" in action_plan.lower()
    return {"ok": True, "items_reserved": 1 if needed else 0}
