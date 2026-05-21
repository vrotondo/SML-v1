from .base import BaseAgent
from .. import database as db


class ReminderAgent(BaseAgent):
    name = "ReminderAgent"
    system_prompt = """You are a legal calendar and deadline manager at a law firm.
Your job is to track all important dates: court hearings, filing deadlines, client meetings,
discovery deadlines, statute of limitations, and any other critical dates.

When adding reminders:
- Always confirm the exact date with the user
- Categorize each deadline clearly
- Flag anything within 14 days as urgent
- Suggest follow-up reminders (e.g. 30 days before a court date, 7 days before a filing deadline)

Reminder types: court_date, filing_deadline, discovery_deadline, client_meeting,
statute_of_limitations, deposition, mediation, appeal_deadline, other"""

    def tools(self):
        return [
            {
                "name": "get_case",
                "description": "Retrieve case details.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "add_reminder",
                "description": "Add a deadline or court date to a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id":       {"type": "integer"},
                        "reminder_type": {"type": "string", "description": "court_date | filing_deadline | discovery_deadline | client_meeting | statute_of_limitations | deposition | mediation | appeal_deadline | other"},
                        "description":   {"type": "string", "description": "Description of the deadline or event"},
                        "due_date":      {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    },
                    "required": ["case_id", "reminder_type", "description", "due_date"],
                },
            },
            {
                "name": "list_upcoming_reminders",
                "description": "List all upcoming reminders across all cases.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "days_ahead": {"type": "integer", "description": "Number of days to look ahead (default 30)"},
                    },
                },
            },
            {
                "name": "get_case_reminders",
                "description": "Get all reminders for a specific case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "complete_reminder",
                "description": "Mark a reminder as completed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reminder_id": {"type": "integer"},
                    },
                    "required": ["reminder_id"],
                },
            },
        ]

    def call_tool(self, name, inputs):
        if name == "get_case":
            case = db.get_case(inputs["case_id"])
            return case or {"error": "Case not found"}

        if name == "add_reminder":
            reminder_id = db.add_reminder(**inputs)
            db.log_action(self.name, "add_reminder", case_id=inputs["case_id"],
                          details=f"{inputs['reminder_type']} on {inputs['due_date']}")
            return {"success": True, "reminder_id": reminder_id,
                    "message": f"Reminder set for {inputs['due_date']}"}

        if name == "list_upcoming_reminders":
            days = inputs.get("days_ahead", 30)
            reminders = db.list_upcoming_reminders(days)
            return {"reminders": reminders, "count": len(reminders),
                    "window": f"Next {days} days"}

        if name == "get_case_reminders":
            reminders = db.get_case_reminders(inputs["case_id"])
            return {"reminders": reminders, "count": len(reminders)}

        if name == "complete_reminder":
            db.complete_reminder(inputs["reminder_id"])
            db.log_action(self.name, "complete_reminder", details=f"reminder_id={inputs['reminder_id']}")
            return {"success": True, "message": "Reminder marked complete"}

        raise ValueError(f"Unknown tool: {name}")
