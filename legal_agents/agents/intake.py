from .base import BaseAgent
from .. import database as db


class IntakeAgent(BaseAgent):
    name = "IntakeAgent"
    system_prompt = """You are a legal client intake specialist at a law firm.
Your job is to collect all relevant information from a client, create their profile,
and open a case record. Ask clarifying questions to ensure completeness.
Always confirm the information back to the user before saving.
Be warm, professional, and thorough."""

    def tools(self):
        return [
            {
                "name": "create_client",
                "description": "Create a new client record in the system.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name":    {"type": "string", "description": "Full legal name"},
                        "email":   {"type": "string", "description": "Email address"},
                        "phone":   {"type": "string", "description": "Phone number"},
                        "address": {"type": "string", "description": "Mailing address"},
                        "notes":   {"type": "string", "description": "Additional notes about the client"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "create_case",
                "description": "Open a new case for an existing client.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id":       {"type": "integer", "description": "ID of the client"},
                        "case_type":       {"type": "string",  "description": "Type of case (e.g. personal injury, contract dispute, family law, criminal defense, real estate, employment)"},
                        "description":     {"type": "string",  "description": "Detailed description of the legal matter"},
                        "opposing_party":  {"type": "string",  "description": "Name of opposing party if known"},
                        "jurisdiction":    {"type": "string",  "description": "State/court jurisdiction"},
                    },
                    "required": ["client_id", "case_type", "description"],
                },
            },
            {
                "name": "search_clients",
                "description": "Search existing clients by name, email, or phone.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_client",
                "description": "Retrieve a client record by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "integer"},
                    },
                    "required": ["client_id"],
                },
            },
            {
                "name": "get_case",
                "description": "Retrieve a case record by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer"},
                    },
                    "required": ["case_id"],
                },
            },
            {
                "name": "list_cases",
                "description": "List all cases, optionally filtered by status.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Filter by status: intake, active, research, drafting, closed"},
                    },
                },
            },
            {
                "name": "update_case_status",
                "description": "Update the status of a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "integer"},
                        "status":  {"type": "string", "description": "New status: intake, active, research, drafting, closed"},
                    },
                    "required": ["case_id", "status"],
                },
            },
        ]

    def call_tool(self, name, inputs):
        if name == "create_client":
            client_id = db.create_client(**inputs)
            db.log_action(self.name, "create_client", details=f"id={client_id}, name={inputs['name']}")
            return {"success": True, "client_id": client_id, "message": f"Client created with ID {client_id}"}

        if name == "create_case":
            case_id = db.create_case(**inputs)
            db.log_action(self.name, "create_case", case_id=case_id)
            case = db.get_case(case_id)
            return {"success": True, "case_id": case_id, "case_number": case["case_number"],
                    "message": f"Case {case['case_number']} opened (ID {case_id})"}

        if name == "search_clients":
            results = db.search_clients(inputs["query"])
            return {"clients": results, "count": len(results)}

        if name == "get_client":
            client = db.get_client(inputs["client_id"])
            return client or {"error": "Client not found"}

        if name == "get_case":
            case = db.get_case(inputs["case_id"])
            return case or {"error": "Case not found"}

        if name == "list_cases":
            cases = db.list_cases(inputs.get("status"))
            return {"cases": cases, "count": len(cases)}

        if name == "update_case_status":
            db.update_case_status(inputs["case_id"], inputs["status"])
            db.log_action(self.name, "update_case_status", case_id=inputs["case_id"], details=inputs["status"])
            return {"success": True, "message": f"Case {inputs['case_id']} status updated to '{inputs['status']}'"}

        raise ValueError(f"Unknown tool: {name}")
