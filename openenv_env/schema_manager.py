import random

class SchemaManager:
    def __init__(self):
        self.schemas = [
            ["event1", "event2", "priority", "email"],
            ["taskA", "taskB", "importance", "message"],
            ["activity_x", "activity_y", "focus", "note"],
            ["workflow_item_primary", "workflow_item_secondary", "decision_weight", "customer_message"]
        ]
        self.policy_versions = [
            {
                "version": "v1",
                "hard_limit": "none",
                "quiet_hours": False,
                "api_contract": "calendar_v1",
                "terms_change": "none"
            },
            {
                "version": "v2",
                "hard_limit": "no work meetings after 8 PM",
                "quiet_hours": True,
                "api_contract": "calendar_v2_timezone_required",
                "terms_change": "late-night work scheduling restricted"
            },
            {
                "version": "v3",
                "hard_limit": "family event cannot be auto-cancelled",
                "quiet_hours": True,
                "api_contract": "calendar_v3_opt_in_reschedule",
                "terms_change": "family-priority policy mandatory"
            }
        ]

    def get_schema(self):
        return random.choice(self.schemas)

    def apply_schema(self, scenario):
        keys = self.get_schema()
        values = list(scenario.values())

        # map values into new schema
        mapped = dict(zip(keys, values))

        return mapped

    def policy_drift_event(self):
        return random.choice(self.policy_versions)