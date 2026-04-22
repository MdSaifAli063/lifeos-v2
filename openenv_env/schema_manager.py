import random

class SchemaManager:
    def __init__(self):
        self.schemas = [
            ["event1", "event2", "priority", "email"],
            ["taskA", "taskB", "importance", "message"],
            ["activity_x", "activity_y", "focus", "note"]
        ]

    def get_schema(self):
        return random.choice(self.schemas)

    def apply_schema(self, scenario):
        keys = self.get_schema()
        values = list(scenario.values())

        # map values into new schema
        mapped = dict(zip(keys, values))

        return mapped