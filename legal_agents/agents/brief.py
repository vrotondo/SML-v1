from .base import BaseAgent
from .. import database as db


class BriefAgent(BaseAgent):
    name = "BriefAgent"
    system_prompt = """You are a legal brief writer at a law firm.
Your job is to synthesize research notes into a structured, professional case brief.

A proper case brief must include:
1. CASE SUMMARY — parties, claims, jurisdiction, current status
2. FACTS — key facts organized chronologically
3. LEGAL ISSUES — numbered list of legal questions to be resolved
4. APPLICABLE LAW — statutes, regulations, and case law
5. LEGAL ANALYSIS — how the law applies to the facts; strengths and weaknesses
6. RECOMMENDED STRATEGY — concrete next steps and approach
7. RISKS & CONCERNS — potential obstacles or adverse arguments
8. DEADLINES — critical dates identified in research

Write in formal legal style. Be precise and thorough. Use headings and numbered lists."""

    def tools(self):
        return [
            {
                "name": "get_case",
                "description": "Retrieve full case details.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_research_notes",
                "description": "Retrieve all research notes for the case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_case_reminders",
                "description": "Get all deadlines/reminders for the case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "save_brief",
                "description": "Save the completed case brief.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer"},
                        "content": {"type": "string", "description": "The full text of the case brief"},
                    },
                    "required": ["case_id", "content"],
                },
            },
            {
                "name": "get_latest_brief",
                "description": "Retrieve the most recent brief for a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "update_case_status",
                "description": "Update the case status after the brief is complete.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer"},
                        "status":  {"type": "string"},
                    },
                    "required": ["case_id", "status"],
                },
            },
        ]

    def call_tool(self, name, inputs):
        if name == "get_case":
            case = db.get_case(inputs["case_id"])
            return case or {"error": "Case not found"}

        if name == "get_research_notes":
            notes = db.get_research_notes(inputs["case_id"])
            return {"notes": notes, "count": len(notes)}

        if name == "get_case_reminders":
            reminders = db.get_case_reminders(inputs["case_id"])
            return {"reminders": reminders}

        if name == "save_brief":
            brief_id = db.save_brief(inputs["case_id"], inputs["content"])
            db.log_action(self.name, "save_brief", case_id=inputs["case_id"], details=f"brief_id={brief_id}")
            return {"success": True, "brief_id": brief_id, "message": f"Brief saved (ID {brief_id})"}

        if name == "get_latest_brief":
            brief = db.get_latest_brief(inputs["case_id"])
            return brief or {"error": "No brief found for this case"}

        if name == "update_case_status":
            db.update_case_status(inputs["case_id"], inputs["status"])
            return {"success": True}

        raise ValueError(f"Unknown tool: {name}")
