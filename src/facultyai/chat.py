"""Interactive chat agent for FacultyAI configuration and queries."""

from __future__ import annotations

import json

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from .config import AppConfig
from .database import Database
from .schema import ColumnDef, Schema

console = Console()

_STYLE = Style.from_dict(
    {
        "prompt": "bold ansicyan",
    }
)


def _help_text() -> str:
    return """Available commands:
  /list          List configured universities
  /jobs          Show job statuses
  /schema        Show current schema columns
  /add-col NAME TYPE [hint HINT] [formula FORMULA] [value VALUE] [value_from FIELD]
                 Add a column to schema.json
  /export        Regenerate Excel from database
  /config        Show current config (secrets masked)
  /help          Show this help
  /exit, /quit   Exit chat"""


async def run_chat(
    config: AppConfig,
    schema: Schema,
    db: Database,
) -> None:
    """Start the interactive chat REPL."""
    console.print(
        Panel.fit(
            "[bold blue]FacultyAI Chat[/]\nType [cyan]/help[/] for available commands.",
            border_style="blue",
        )
    )

    session = PromptSession(
        history=FileHistory(".facultyai_chat_history"),
        style=_STYLE,
    )

    while True:
        try:
            line = await session.prompt_async("facultyai> ", style=_STYLE)
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            await _handle_command(line, config, schema, db)
        else:
            await _handle_natural_language(line, config, schema, db)


async def _handle_command(
    line: str,
    config: AppConfig,
    schema: Schema,
    db: Database,
) -> None:
    parts = line.split()
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        raise EOFError

    if cmd == "/help":
        console.print(_help_text())

    elif cmd == "/list":
        rows = await db.get_input_universities()
        if not rows:
            console.print("[yellow]No universities configured.[/]")
        else:
            for r in rows:
                dept = r["department"] or "[all departments]"
                console.print(f"  • {r['university']} — {dept}")

    elif cmd == "/jobs":
        jobs = await db.list_jobs()
        if not jobs:
            console.print("[yellow]No jobs yet.[/]")
        else:
            status_colors = {
                "completed": "green",
                "running": "yellow",
                "failed": "red",
                "pending": "dim",
            }
            for j in jobs:
                c = status_colors.get(j["status"], "white")
                dept = j["department"] or "All"
                console.print(f"  [{c}]●[/{c}] {j['university']}/{dept} [{c}]{j['status']}[/{c}]")

    elif cmd == "/schema":
        if not schema.columns:
            console.print("[yellow]No schema columns defined.[/]")
        else:
            for col in schema.columns:
                extra = ""
                if col.hint:
                    extra += f" hint={col.hint}"
                if col.formula:
                    extra += f" formula={col.formula}"
                if col.value:
                    extra += f" value={col.value}"
                if col.value_from:
                    extra += f" value_from={col.value_from}"
                console.print(f"  [{col.type}][/] {col.name}{extra}")

    elif cmd == "/add-col":
        await _add_column(parts[1:], schema, config)

    elif cmd == "/export":
        from .exporter import export_to_excel

        rows = await export_to_excel(db, schema, config.files.output_excel)
        console.print(f"[green]Exported {rows} rows to {config.files.output_excel}[/]")

    elif cmd == "/config":
        from .config import mask_secrets

        console.print_json(json.dumps(mask_secrets(config)))

    else:
        console.print(f"[red]Unknown command: {cmd}[/]. Type /help for available commands.")


async def _add_column(args: list[str], schema: Schema, config: AppConfig) -> None:
    if len(args) < 2:
        console.print(
            "[red]Usage: /add-col NAME TYPE [hint HINT] [formula FORMULA] [value VALUE] [value_from FIELD][/]"
        )
        return

    name = args[0]
    col_type = args[1]
    if col_type not in ("extracted", "formula", "static"):
        console.print(f"[red]Invalid type: {col_type}. Must be extracted, formula, or static.[/]")
        return

    hint = None
    formula = None
    value = None
    value_from = None

    i = 2
    while i < len(args):
        if args[i] == "hint" and i + 1 < len(args):
            hint = args[i + 1]
            i += 2
        elif args[i] == "formula" and i + 1 < len(args):
            formula = args[i + 1]
            i += 2
        elif args[i] == "value" and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        elif args[i] == "value_from" and i + 1 < len(args):
            value_from = args[i + 1]
            i += 2
        else:
            i += 1

    col = ColumnDef(
        name=name, type=col_type, hint=hint, formula=formula, value=value, value_from=value_from
    )
    schema.columns.append(col)

    from pathlib import Path

    path = Path(config.files.schema_file)
    path.write_text(
        schema.model_dump_json(indent=2),
        encoding="utf-8",
    )

    console.print(f"[green]Added column '{name}' ({col_type}) to {path}[/]")


async def _handle_natural_language(
    text: str,
    config: AppConfig,
    schema: Schema,
    db: Database,
) -> None:
    """Basic natural language command parsing."""
    text_lower = text.lower().strip()

    if "add university" in text_lower:
        import re

        match = re.search(r"add university\s+(.+)", text_lower)
        if match:
            rest = match.group(1).strip()
            parts = rest.split(",")
            uni = parts[0].strip()
            dept = parts[1].strip() if len(parts) > 1 else None
            await db.upsert_input_university(uni, dept)
            console.print(f"[green]Added university: {uni}{' (' + dept + ')' if dept else ''}[/]")

    elif "list universit" in text_lower:
        rows = await db.get_input_universities()
        for r in rows:
            dept = r["department"] or "[all]"
            console.print(f"  • {r['university']} — {dept}")

    elif "job status" in text_lower or "show jobs" in text_lower:
        jobs = await db.list_jobs()
        if not jobs:
            console.print("[yellow]No jobs yet.[/]")
        else:
            for j in jobs:
                dept = j["department"] or "All"
                console.print(f"  {j['university']}/{dept} [{j['status']}]")

    elif "show config" in text_lower:
        from .config import mask_secrets

        console.print_json(json.dumps(mask_secrets(config)))

    else:
        console.print(
            "[dim]I didn't understand that. Try using slash commands like /help for available options.[/]"
        )
