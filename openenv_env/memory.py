class Memory:
    def __init__(self):
        self.history = []
        self.preferences = {
            "quiet_hours_start": 20,
            "family_priority_weight": 1.2,
            "work_priority_weight": 1.0,
            "preferred_tone": "calm",
            "risk_tolerance": "medium"
        }

    def add(self, scenario, decision, metadata=None):
        metadata = metadata or {}
        self.history.append({
            "scenario": scenario,
            "decision": decision,
            "metadata": metadata
        })
        self._update_preferences(scenario, decision, metadata)

    def _update_preferences(self, scenario, decision, metadata):
        text = (decision or "").lower()
        if "family" in text:
            self.preferences["family_priority_weight"] = min(
                self.preferences["family_priority_weight"] + 0.02, 1.8
            )
        if "client" in text or "meeting" in text or "investor" in text:
            self.preferences["work_priority_weight"] = min(
                self.preferences["work_priority_weight"] + 0.01, 1.7
            )
        if metadata.get("emotion") in ("angry", "stressed"):
            self.preferences["preferred_tone"] = "de-escalation"
        if metadata.get("reward", 0) < 0:
            self.preferences["risk_tolerance"] = "low"

    def get_context(self, k=3):
        return self.history[-k:] if len(self.history) >= k else self.history

    def get_preferences(self):
        return dict(self.preferences)