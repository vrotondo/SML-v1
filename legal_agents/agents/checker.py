from .base import BaseAgent
from .. import database as db


class CheckerAgent(BaseAgent):
    name = "CheckerAgent"
    system_prompt = """You are a senior reviewing attorney at a law firm.
Your job is to review drafted legal documents for quality, accuracy, and completeness.

When reviewing, check for:
1. LEGAL ACCURACY — are citations correct? Is the law stated accurately?
2. FACTUAL ACCURACY — do facts match the case record?
3. COMPLETENESS — are all required elements present for this document type?
4. PROFESSIONAL TONE — appropriate legal language, no informal phrasing
5. FORMATTING — proper headings, paragraphs, signature blocks
6. STRATEGIC SOUNDNESS — does the document serve the client's interests?
7. RISK EXPOSURE — does anything in the document create unnecessary liability?
8. JURISDICTION — is the document appropriate for the stated jurisdiction?

Return a clear APPROVE or NEEDS REVISION verdict with specific feedback.
If approving, also note any minor suggestions for improvement."""

    def tools(self):
        return [
            {
                "name": "get_document",
                "description": "Retrieve a document to review.",
                "input_schema": {
                    "type": "object",
                    "properties": {"doc_id": {"type": "integer"}},
                    "required": ["doc_id"],
                },
            },
            {
                "name": "list_documents",
                "description": "List all documents for a case.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_case",
                "description": "Get case details for cross-referencing.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "get_latest_brief",
                "description": "Get the case brief to cross-reference with the document.",
                "input_schema": {
                    "type": "object",
                    "properties": {"case_id": {"type": "integer"}},
                    "required": ["case_id"],
                },
            },
            {
                "name": "approve_document",
                "description": "Approve a document after review.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id":       {"type": "integer"},
                        "review_notes": {"type": "string", "description": "Approval notes and minor suggestions"},
                    },
                    "required": ["doc_id"],
                },
            },
            {
                "name": "flag_for_revision",
                "description": "Flag a document as needing revision with specific issues.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id":  {"type": "integer"},
                        "issues":  {"type": "string", "description": "Detailed list of issues that must be fixed"},
                    },
                    "required": ["doc_id", "issues"],
                },
            },
        ]

    def call_tool(self, name, inputs):
        if name == "get_document":
            doc = db.get_document(inputs["doc_id"])
            return doc or {"error": "Document not found"}

        if name == "list_documents":
            docs = db.list_documents(inputs["case_id"])
            return {"documents": docs, "count": len(docs)}

        if name == "get_case":
            case = db.get_case(inputs["case_id"])
            return case or {"error": "Case not found"}

        if name == "get_latest_brief":
            brief = db.get_latest_brief(inputs["case_id"])
            return brief or {"message": "No brief on file"}

        if name == "approve_document":
            db.update_document(inputs["doc_id"], "approved", inputs.get("review_notes"))
            db.log_action(self.name, "approve_document", details=f"doc_id={inputs['doc_id']}")
            return {"success": True, "message": f"Document {inputs['doc_id']} approved"}

        if name == "flag_for_revision":
            db.update_document(inputs["doc_id"], "needs_revision", inputs["issues"])
            db.log_action(self.name, "flag_for_revision", details=f"doc_id={inputs['doc_id']}")
            return {"success": True, "message": f"Document {inputs['doc_id']} flagged for revision"}

        raise ValueError(f"Unknown tool: {name}")
