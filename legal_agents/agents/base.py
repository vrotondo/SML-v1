import json
import anthropic
from typing import Any
from ..config import MODEL, MAX_TOKENS, MAX_AGENT_ITERATIONS
from .. import database as db


class BaseAgent:
    name: str = "BaseAgent"
    system_prompt: str = "You are a specialized legal AI assistant. Be professional, precise, and thorough."

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def tools(self) -> list:
        """Return list of Anthropic tool definitions. Override in subclass."""
        return []

    def call_tool(self, name: str, inputs: dict) -> Any:
        """Dispatch a tool call. Override in subclass."""
        raise NotImplementedError(f"Tool '{name}' not implemented in {self.name}")

    def run(self, task: str, max_iterations: int = MAX_AGENT_ITERATIONS) -> str:
        """Run the agent loop until completion or iteration limit."""
        messages = [{"role": "user", "content": task}]
        tool_defs = self.tools()

        # Use ephemeral caching on the static system prompt
        system = [{"type": "text", "text": self.system_prompt,
                   "cache_control": {"type": "ephemeral"}}]

        for _ in range(max_iterations):
            kwargs = dict(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
            )
            if tool_defs:
                kwargs["tools"] = tool_defs

            response = self.client.messages.create(**kwargs)

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            result = self.call_tool(block.name, block.input)
                            content = json.dumps(result) if not isinstance(result, str) else result
                        except Exception as exc:
                            content = f"Error: {exc}"
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        })
                messages.append({"role": "user", "content": results})
                db.log_action(self.name, "tool_use", details=str([b.name for b in response.content if b.type == "tool_use"]))
            else:
                break

        return "Agent reached iteration limit before completing the task."

    @staticmethod
    def _extract_text(response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
