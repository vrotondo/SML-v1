from .base import BaseAgent
from .. import database as db


class ResearchAgent(BaseAgent):
    name = "ResearchAgent"
    system_prompt = """You are a senior legal research attorney with expertise across all areas of law.
Your job is to conduct thorough legal research for a given case.

When researching, you should:
1. Identify the key legal issues and claims/defenses
2. Cite relevant statutes and code sections for the jurisdiction
3. Reference landmark and recent case precedents (use your knowledge of US law)
4. Identify strengths and weaknesses in the case
5. Note any statutes of limitations or procedural deadlines
6. Suggest legal strategies based on the facts

Save each distinct research topic as a separate note so the Brief agent can organize them effectively.
Be precise with citations — include case names, years, and jurisdiction where possible."""

    def tools(self):
        return [
            {
                "name": "get_case",
                "description": "Retrieve full case details including client info.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "save_research_note",
                "description": "Save a research finding, statute, or case precedent.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer"},
                        "title":   {"type": "string", "description": "Short title for this research note (e.g. 'Statute of Limitations', 'Key Precedents', 'Elements of Claim')"},
                        "content": {"type": "string", "description": "Detailed research content"},
                        "source":  {"type": "string", "description": "Source reference (e.g. '42 U.S.C. § 1983', 'CA Civil Code § 1714')"},
                    },
                    "required": ["case_id", "title", "content"],
                },
            },
            {
                "name": "get_research_notes",
                "description": "Retrieve all saved research notes for a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "add_reminder",
                "description": "Add a deadline discovered during research (e.g. statute of limitations expiry).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id":       {"type": "integer"},
                        "reminder_type": {"type": "string", "description": "e.g. statute_of_limitations, filing_deadline, discovery_deadline"},
                        "description":   {"type": "string"},
                        "due_date":      {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    },
                    "required": ["case_id", "reminder_type", "description", "due_date"],
                },
            },
        ]

    def call_tool(self, name, inputs):
        if name == "get_case":
            case = db.get_case(inputs["case_id"])
            return case or {"error": "Case not found"}

        if name == "save_research_note":
            note_id = db.save_research_note(**inputs)
            db.log_action(self.name, "save_research_note", case_id=inputs["case_id"], details=inputs["title"])
            return {"success": True, "note_id": note_id, "message": f"Research note saved (ID {note_id})"}

        if name == "get_research_notes":
            notes = db.get_research_notes(inputs["case_id"])
            return {"notes": notes, "count": len(notes)}

        if name == "add_reminder":
            reminder_id = db.add_reminder(**inputs)
            db.log_action(self.name, "add_reminder", case_id=inputs["case_id"], details=inputs["description"])
            return {"success": True, "reminder_id": reminder_id, "message": "Deadline added to reminders"}

        raise ValueError(f"Unknown tool: {name}")
