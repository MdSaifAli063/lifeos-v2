def build_prompt(scenario, memory):
    return f"""
You are a professional AI personal assistant.

Past Decisions:
{memory}

Current Scenario:
Event1: {scenario.get('event1')}
Event2: {scenario.get('event2')}
Priority: {scenario.get('priority')}
Email: {scenario.get('email')}

Instructions:
1. Choose best action (event1/event2/reschedule)
2. Explain briefly
3. Write polite reply

Answer:
"""