import random
from .memory import Memory
from .reward import reward_function
from .schema_manager import SchemaManager
from .personas import PERSONAS
from .predictor import predict_future_conflict
from .workflow_engine import run_workflow
from .planner import decompose_goal, replan_if_needed
from .agents import delegate_agents
from .tools import calendar_tool, email_tool, rides_tool, shopping_tool

class LifeOSEnv:
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.memory = Memory()
        self.schema_manager = SchemaManager()
        self.current_persona = None
        self.current_policy = None
        self.prediction = None
        self.workflow = []
        self.plan = []

    def reset(self):
        self.current = random.choice(self.scenarios)
        self.current_persona = random.choice(PERSONAS)
        self.current_policy = self.schema_manager.policy_drift_event()
        self.prediction = predict_future_conflict(
            self.current,
            self.memory.get_preferences()
        )
        self.workflow = run_workflow(self.current, self.prediction, self.current_policy)
        self.plan = replan_if_needed(
            decompose_goal(self.current),
            self.current_policy.get("hard_limit")
        )
        self.state = self.schema_manager.apply_schema(self.current)
        self.state["persona"] = self.current_persona["name"]
        self.state["policy_version"] = self.current_policy["version"]
        self.state["predicted_conflict"] = self.prediction["likely_conflict_next_48h"]
        return self.state

    def step(self, action):
        action_text = str(action)
        lower_action = action_text.lower()
        emotion = "neutral"
        if any(word in lower_action for word in ["angry", "frustrated", "unfair", "stress"]):
            emotion = "stressed"

        delegated = delegate_agents(
            self.current,
            self.current_persona,
            emotion,
            self.current_policy
        )
        tool_results = {
            "calendar": calendar_tool(action_text),
            "email": email_tool(action_text),
            "rides": rides_tool(action_text),
            "shopping": shopping_tool(action_text)
        }
        reward, breakdown = reward_function(
            action_text,
            self.current,
            context={
                "policy": self.current_policy,
                "prediction": self.prediction,
                "tool_results": tool_results
            }
        )

        self.memory.add(
            self.current,
            action_text,
            metadata={"emotion": emotion, "reward": reward}
        )

        done = True
        info = {
            "scenario": self.current,
            "action": action_text,
            "persona": self.current_persona,
            "policy": self.current_policy,
            "prediction": self.prediction,
            "workflow": self.workflow,
            "plan": self.plan,
            "delegation": delegated,
            "tool_results": tool_results,
            "reward_breakdown": breakdown
        }

        return self.state, reward, done, info