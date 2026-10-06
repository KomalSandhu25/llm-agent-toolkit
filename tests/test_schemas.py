"""Tests for agent_toolkit.schemas."""

from agent_toolkit.schemas import AgentStep, Message, Role, ToolCall, ToolResult


def test_message_constructors() -> None:
    assert Message.system("s").role is Role.SYSTEM
    assert Message.user("u").content == "u"
    call = ToolCall(name="calculator", arguments={"expression": "1+1"})
    assistant = Message.assistant("thinking", [call])
    assert assistant.tool_calls[0].name == "calculator"


def test_tool_message_links_back_to_call() -> None:
    call = ToolCall(name="echo", arguments={"text": "x"})
    result = ToolResult(call_id=call.id, name="echo", output="x")
    msg = Message.from_tool_result(result)
    assert msg.role is Role.TOOL
    assert msg.tool_call_id == call.id


def test_tool_call_ids_are_unique() -> None:
    assert ToolCall(name="a").id != ToolCall(name="a").id


def test_message_round_trips_through_json() -> None:
    original = Message.assistant("hi", [ToolCall(name="t", arguments={"k": 1})])
    restored = Message.model_validate_json(original.model_dump_json())
    assert restored == original


def test_agent_step_final_flag() -> None:
    assert not AgentStep(index=0, thought="hmm").is_final
    assert AgentStep(index=1, final_answer="42").is_final
