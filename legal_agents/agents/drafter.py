from .base import BaseAgent
from .. import database as db


DOCUMENT_TYPES = {
    "demand_letter":       "Formal demand letter to opposing party",
    "complaint":           "Initial civil complaint/petition",
    "motion_to_dismiss":   "Motion to dismiss filing",
    "motion_for_summary":  "Motion for summary judgment",
    "settlement_agreement":"Settlement agreement and release",
    "retainer_agreement":  "Attorney-client retainer agreement",
    "cease_and_desist":    "Cease and desist letter",
    "discovery_request":   "Interrogatories / Request for production",
    "deposition_outline":  "Deposition outline and questions",
    "client_letter":       "General client update letter",
    "contract_review":     "Contract review memorandum",
    "memo":                "Internal legal memorandum",
}


class DrafterAgent(BaseAgent):
    name = "DrafterAgent"
    system_prompt = """You are a skilled legal document drafter at a law firm.
Your job is to produce professional, precise legal documents based on case facts and research.

When drafting:
- Use proper legal formatting and headings
- Include all required legal elements for the document type
- Reference specific facts from the case record
- Cite applicable statutes and case law from the research brief
- Use formal legal language, but ensure clarity
- Include signature blocks, date lines, and proper captions where applicable
- Always check case type and jurisdiction to ensure correct legal standards

Available document types:
""" + "\n".join(f"- {k}: {v}" for k, v in DOCUMENT_TYPES.items())

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
                "name": "get_latest_brief",
                "description": "Get the latest case research brief to inform the draft.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_research_notes",
                "description": "Get all research notes for the case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "save_document",
                "description": "Save a drafted legal document.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id":  {"type": "integer"},
                        "doc_type": {"type": "string", "description": "One of: " + ", ".join(DOCUMENT_TYPES.keys())},
                        "title":    {"type": "string", "description": "Document title"},
                        "content":  {"type": "string", "description": "Full document text"},
                    },
                    "required": ["case_id", "doc_type", "title", "content"],
                },
            },
            {
                "name": "list_documents",
                "description": "List all documents drafted for a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_document",
                "description": "Retrieve a specific document by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {"doc_id": {"type": "integer"}},
                    "required": ["doc_id"],
                },
            },
            {
                "name": "update_case_status",
                "description": "Update case status.",
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

        if name == "get_latest_brief":
            brief = db.get_latest_brief(inputs["case_id"])
            return brief or {"message": "No brief available — drafting from case facts only"}

        if name == "get_research_notes":
            notes = db.get_research_notes(inputs["case_id"])
            return {"notes": notes, "count": len(notes)}

        if name == "save_document":
            doc_id = db.save_document(**inputs)
            db.log_action(self.name, "save_document", case_id=inputs["case_id"],
                          details=f"{inputs['doc_type']}: {inputs['title']}")
            return {"success": True, "doc_id": doc_id,
                    "message": f"Document saved (ID {doc_id}): {inputs['title']}"}

        if name == "list_documents":
            docs = db.list_documents(inputs["case_id"])
            return {"documents": docs, "count": len(docs)}

        if name == "get_document":
            doc = db.get_document(inputs["doc_id"])
            return doc or {"error": "Document not found"}

        if name == "update_case_status":
            db.update_case_status(inputs["case_id"], inputs["status"])
            return {"success": True}

        raise ValueError(f"Unknown tool: {name}")
