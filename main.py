#!/usr/bin/env python3
"""
Legal AI Agent System — Command-line interface
"""
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box

from legal_agents import database as db
from legal_agents.orchestrator import LegalOrchestrator

console = Console()


def require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY environment variable is not set.")
        console.print("Set it with: [bold]export ANTHROPIC_API_KEY=your-key-here[/bold]")
        sys.exit(1)
    return key


def print_banner():
    console.print(Panel.fit(
        "[bold blue]Legal AI Agent System[/bold blue]\n"
        "[dim]Powered by Claude claude-sonnet-4-6 · 7-Agent Architecture[/dim]",
        border_style="blue",
    ))


def show_upcoming_deadlines():
    reminders = db.list_upcoming_reminders(days_ahead=14)
    if reminders:
        console.print(f"\n[bold yellow]⚠  {len(reminders)} deadline(s) in the next 14 days:[/bold yellow]")
        for r in reminders:
            days_left = (datetime.strptime(r["due_date"], "%Y-%m-%d").date() - datetime.now().date()).days
            urgency = "[red]URGENT[/red]" if days_left <= 3 else "[yellow]" + str(days_left) + " days[/yellow]"
            console.print(f"  • {r['case_number']} ({r['client_name']}) — {r['description']} [{urgency}]")


def print_cases_table(cases):
    if not cases:
        console.print("[dim]No cases found.[/dim]")
        return
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("ID",          style="dim",    width=6)
    table.add_column("Case #",      style="cyan",   width=18)
    table.add_column("Client",      width=22)
    table.add_column("Type",        width=20)
    table.add_column("Status",      width=12)
    table.add_column("Opened",      width=12)
    status_colors = {
        "intake": "yellow", "active": "green", "research": "blue",
        "drafting": "magenta", "closed": "dim",
    }
    for c in cases:
        color = status_colors.get(c.get("status", ""), "white")
        table.add_row(
            str(c["id"]),
            c.get("case_number", ""),
            c.get("client_name", ""),
            c.get("case_type", ""),
            f"[{color}]{c.get('status', '')}[/{color}]",
            (c.get("created_at") or "")[:10],
        )
    console.print(table)


def menu_intake(orch: LegalOrchestrator):
    console.rule("[bold]New Client Intake[/bold]")
    console.print("[dim]Describe the new client and their legal matter. The Intake Agent will ask clarifying questions.[/dim]\n")
    console.print("Enter client details (name, contact info, nature of legal matter):")
    console.print("[dim]Example: John Smith, 555-0100, john@example.com, 123 Main St. He was in a car accident on 3/15/2024 and suffered a broken arm. The other driver ran a red light.[/dim]\n")

    info = Prompt.ask("[bold green]Client info[/bold green]")
    if not info.strip():
        return

    console.print("\n[cyan]Running Intake Agent...[/cyan]")
    with console.status("Processing...", spinner="dots"):
        result = orch.run_intake(info)

    console.print(Panel(Markdown(result), title="[bold]Intake Agent Response[/bold]", border_style="green"))


def menu_research(orch: LegalOrchestrator):
    console.rule("[bold]Case Research[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("\nEnter Case ID to research")
    if not case_id.isdigit():
        return
    case_id = int(case_id)

    extra = Prompt.ask("Any additional context for the researcher? (press Enter to skip)", default="")

    console.print("\n[cyan]Running Research Agent — this may take a moment...[/cyan]")
    with console.status("Researching...", spinner="dots"):
        result = orch.run_research(case_id, extra)

    console.print(Panel(Markdown(result), title="[bold]Research Agent Report[/bold]", border_style="blue"))


def menu_brief(orch: LegalOrchestrator):
    console.rule("[bold]Generate Case Brief[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("\nEnter Case ID")
    if not case_id.isdigit():
        return
    case_id = int(case_id)

    console.print("\n[cyan]Running Brief Agent...[/cyan]")
    with console.status("Generating brief...", spinner="dots"):
        result = orch.run_brief(case_id)

    console.print(Panel(Markdown(result), title="[bold]Case Brief[/bold]", border_style="magenta"))


def menu_reminders(orch: LegalOrchestrator):
    console.rule("[bold]Manage Deadlines & Reminders[/bold]")

    choice = Prompt.ask(
        "Choose action",
        choices=["1", "2", "3"],
        default="1",
        show_choices=False,
    )
    console.print("[dim]1=View upcoming  2=Add reminders for case  3=Mark complete[/dim]")

    choice = Prompt.ask("Choice (1/2/3)", choices=["1", "2", "3"], default="1")

    if choice == "1":
        days = Prompt.ask("Days ahead to check", default="30")
        reminders = db.list_upcoming_reminders(int(days) if days.isdigit() else 30)
        if not reminders:
            console.print("[green]No upcoming reminders.[/green]")
            return
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold yellow")
        table.add_column("ID",       width=6)
        table.add_column("Case #",   width=18)
        table.add_column("Client",   width=22)
        table.add_column("Type",     width=22)
        table.add_column("Description", width=35)
        table.add_column("Due Date", width=12)
        for r in reminders:
            table.add_row(
                str(r["id"]), r.get("case_number", ""), r.get("client_name", ""),
                r.get("reminder_type", ""), r.get("description", ""), r.get("due_date", ""),
            )
        console.print(table)

    elif choice == "2":
        cases = db.list_cases()
        print_cases_table(cases)
        case_id = Prompt.ask("Enter Case ID")
        if not case_id.isdigit():
            return
        dates_info = Prompt.ask("Describe the dates/deadlines to add (or press Enter for AI to decide)")
        console.print("\n[cyan]Running Reminder Agent...[/cyan]")
        with console.status("Setting reminders...", spinner="dots"):
            result = orch.run_reminders(int(case_id), dates_info)
        console.print(Panel(Markdown(result), title="[bold]Reminder Agent[/bold]", border_style="yellow"))

    elif choice == "3":
        reminders = db.list_upcoming_reminders(365)
        if not reminders:
            console.print("[green]No open reminders.[/green]")
            return
        for r in reminders:
            console.print(f"[{r['id']}] {r.get('case_number')} — {r['description']} ({r['due_date']})")
        rid = Prompt.ask("Enter Reminder ID to mark complete")
        if rid.isdigit():
            db.complete_reminder(int(rid))
            console.print("[green]Reminder marked complete.[/green]")


def menu_draft(orch: LegalOrchestrator):
    console.rule("[bold]Draft Legal Document[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("Enter Case ID")
    if not case_id.isdigit():
        return
    case_id = int(case_id)

    doc_types = [
        "demand_letter", "complaint", "motion_to_dismiss", "motion_for_summary",
        "settlement_agreement", "retainer_agreement", "cease_and_desist",
        "discovery_request", "deposition_outline", "client_letter", "contract_review", "memo",
    ]
    console.print("\nDocument types: " + ", ".join(f"[cyan]{d}[/cyan]" for d in doc_types))
    doc_type = Prompt.ask("Document type")
    instructions = Prompt.ask("Specific instructions (optional)", default="")

    console.print("\n[cyan]Running Drafter Agent...[/cyan]")
    with console.status("Drafting...", spinner="dots"):
        result = orch.run_draft(case_id, doc_type, instructions)

    console.print(Panel(Markdown(result), title=f"[bold]Draft: {doc_type}[/bold]", border_style="magenta"))


def menu_review(orch: LegalOrchestrator):
    console.rule("[bold]Review Document[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("Enter Case ID")
    if not case_id.isdigit():
        return
    case_id = int(case_id)

    docs = db.list_documents(case_id)
    if not docs:
        console.print("[yellow]No documents found for this case.[/yellow]")
        return

    console.print("\nDocuments:")
    for d in docs:
        console.print(f"  [{d['id']}] {d['doc_type']} — {d['title']} [{d['status']}]")

    doc_id_input = Prompt.ask("Enter Document ID to review (or Enter for all)", default="")
    doc_id = int(doc_id_input) if doc_id_input.isdigit() else None

    console.print("\n[cyan]Running Checker Agent...[/cyan]")
    with console.status("Reviewing...", spinner="dots"):
        result = orch.run_review(case_id, doc_id)

    console.print(Panel(Markdown(result), title="[bold]Review Report[/bold]", border_style="red"))


def menu_full_workflow(orch: LegalOrchestrator):
    console.rule("[bold]Full Workflow — Research → Brief → Reminders[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("Enter Case ID to run full workflow")
    if not case_id.isdigit():
        return
    case_id = int(case_id)

    if not Confirm.ask(f"Run full Research → Brief → Reminders workflow for case {case_id}?"):
        return

    results = orch.run_full_workflow(case_id, console)

    for step, output in results.items():
        console.print(Panel(Markdown(output), title=f"[bold]{step}[/bold]", border_style="cyan"))


def menu_view_cases(orch: LegalOrchestrator = None):
    console.rule("[bold]All Cases[/bold]")
    cases = db.list_cases()
    print_cases_table(cases)

    if not cases:
        return

    case_id = Prompt.ask("\nEnter Case ID for details (or Enter to go back)", default="")
    if not case_id.isdigit():
        return

    case = db.get_case(int(case_id))
    if not case:
        console.print("[red]Case not found.[/red]")
        return

    console.print(Panel(
        f"[bold]Case:[/bold] {case['case_number']}\n"
        f"[bold]Client:[/bold] {case['client_name']}  |  {case.get('client_email', '')}  |  {case.get('client_phone', '')}\n"
        f"[bold]Type:[/bold] {case['case_type']}\n"
        f"[bold]Status:[/bold] {case['status']}\n"
        f"[bold]Jurisdiction:[/bold] {case.get('jurisdiction', 'N/A')}\n"
        f"[bold]Opposing Party:[/bold] {case.get('opposing_party', 'N/A')}\n\n"
        f"[bold]Description:[/bold]\n{case.get('description', '')}",
        title=f"Case Details — {case['case_number']}",
        border_style="cyan",
    ))

    notes = db.get_research_notes(int(case_id))
    brief = db.get_latest_brief(int(case_id))
    docs = db.list_documents(int(case_id))
    reminders = db.get_case_reminders(int(case_id))

    console.print(f"  Research notes: [cyan]{len(notes)}[/cyan]   "
                  f"Brief: [cyan]{'v' + str(brief['version']) if brief else 'None'}[/cyan]   "
                  f"Documents: [cyan]{len(docs)}[/cyan]   "
                  f"Reminders: [cyan]{len(reminders)}[/cyan]")


MENU = [
    ("1", "New Client Intake",                  menu_intake),
    ("2", "Case Research",                      menu_research),
    ("3", "Generate Case Brief",                menu_brief),
    ("4", "Manage Deadlines & Reminders",       menu_reminders),
    ("5", "Draft Legal Document",               menu_draft),
    ("6", "Review Document",                    menu_review),
    ("7", "Full Workflow (Research→Brief→Remind)", menu_full_workflow),
    ("8", "View All Cases",                     menu_view_cases),
    ("0", "Exit",                               None),
]


def main():
    db.init_db()
    key = require_api_key()
    orch = LegalOrchestrator(api_key=key)

    print_banner()
    show_upcoming_deadlines()

    while True:
        console.print("\n[bold]Main Menu[/bold]")
        for key_str, label, _ in MENU:
            console.print(f"  [cyan]{key_str}[/cyan]  {label}")

        choice = Prompt.ask("\nSelect option", default="8")

        for key_str, label, fn in MENU:
            if choice == key_str:
                if fn is None:
                    console.print("[dim]Goodbye.[/dim]")
                    sys.exit(0)
                try:
                    fn(orch)
                except KeyboardInterrupt:
                    console.print("\n[dim]Cancelled.[/dim]")
                except Exception as exc:
                    console.print(f"[red]Error:[/red] {exc}")
                break
        else:
            console.print("[dim]Invalid choice.[/dim]")


if __name__ == "__main__":
    main()
