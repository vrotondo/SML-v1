import anthropic
from .config import MODEL, MAX_TOKENS
from .agents.intake import IntakeAgent
from .agents.research import ResearchAgent
from .agents.brief import BriefAgent
from .agents.reminder import ReminderAgent
from .agents.drafter import DrafterAgent
from .agents.checker import CheckerAgent
from . import database as db


class LegalOrchestrator:
    """
    Coordinates all agents to handle a full case workflow.
    Decides which agents to invoke and in what sequence based on case state.
    """

    WORKFLOW = [
        ("intake",    "IntakeAgent",    "Collect and confirm all client/case information"),
        ("research",  "ResearchAgent",  "Conduct comprehensive legal research"),
        ("briefing",  "BriefAgent",     "Generate structured case brief from research"),
        ("reminders", "ReminderAgent",  "Set all deadlines and court dates"),
        ("drafting",  "DrafterAgent",   "Draft required legal documents"),
        ("review",    "CheckerAgent",   "Review all documents for quality and accuracy"),
    ]

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.agents = {
            "IntakeAgent":    IntakeAgent(self.client),
            "ResearchAgent":  ResearchAgent(self.client),
            "BriefAgent":     BriefAgent(self.client),
            "ReminderAgent":  ReminderAgent(self.client),
            "DrafterAgent":   DrafterAgent(self.client),
            "CheckerAgent":   CheckerAgent(self.client),
        }

    def get_agent(self, agent_name: str):
        return self.agents[agent_name]

    def run_intake(self, client_info: str) -> str:
        agent = self.get_agent("IntakeAgent")
        return agent.run(
            f"Please process this new client intake:\n\n{client_info}\n\n"
            "Create the client record and open a case. Confirm all details back to me."
        )

    def run_research(self, case_id: int, extra_context: str = "") -> str:
        agent = self.get_agent("ResearchAgent")
        task = (
            f"Please conduct comprehensive legal research for case ID {case_id}. "
            "Retrieve the case details first, then research all applicable laws, "
            "statutes, and precedents. Save multiple research notes covering: "
            "elements of the claim/defense, applicable statutes, key case precedents, "
            "statute of limitations, and recommended strategy."
        )
        if extra_context:
            task += f"\n\nAdditional context: {extra_context}"
        return agent.run(task)

    def run_brief(self, case_id: int) -> str:
        agent = self.get_agent("BriefAgent")
        return agent.run(
            f"Generate a comprehensive case brief for case ID {case_id}. "
            "Retrieve the case details and all research notes, then produce "
            "a structured brief covering all required sections."
        )

    def run_reminders(self, case_id: int, dates_info: str = "") -> str:
        agent = self.get_agent("ReminderAgent")
        task = (
            f"Review case ID {case_id} and set up all necessary deadline reminders. "
            "Check the case details and existing reminders."
        )
        if dates_info:
            task += f"\n\nDates to add: {dates_info}"
        return agent.run(task)

    def run_draft(self, case_id: int, doc_type: str, instructions: str = "") -> str:
        agent = self.get_agent("DrafterAgent")
        task = (
            f"Draft a '{doc_type}' for case ID {case_id}. "
            "Retrieve the case details, latest brief, and any relevant research notes first."
        )
        if instructions:
            task += f"\n\nSpecific instructions: {instructions}"
        return agent.run(task)

    def run_review(self, case_id: int, doc_id: int = None) -> str:
        agent = self.get_agent("CheckerAgent")
        if doc_id:
            task = (
                f"Review document ID {doc_id} for case ID {case_id}. "
                "Cross-reference with the case brief and provide approve/revise verdict."
            )
        else:
            task = (
                f"Review all draft documents for case ID {case_id}. "
                "Retrieve and review each document, providing verdict and feedback for each."
            )
        return agent.run(task)

    def run_full_workflow(self, case_id: int, console=None) -> dict:
        """Run research → brief → review workflow for an existing case."""
        results = {}

        steps = [
            ("Research",  lambda: self.run_research(case_id)),
            ("Brief",     lambda: self.run_brief(case_id)),
            ("Reminders", lambda: self.run_reminders(case_id)),
        ]

        for step_name, fn in steps:
            if console:
                console.print(f"[cyan]Running {step_name} agent...[/cyan]")
            result = fn()
            results[step_name] = result
            if console:
                console.print(f"[green]{step_name} complete.[/green]")

        return results
