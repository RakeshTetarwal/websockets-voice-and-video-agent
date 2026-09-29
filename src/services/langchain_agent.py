from typing import AsyncGenerator, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from src.core.config import config
from src.tools.system_tools import get_current_time, get_weather, calculate, save_note, get_notes


# Wrap our functions into LangChain Tools
@tool
def tool_get_time() -> str:
    """Returns the current UTC date and time."""
    return get_current_time()

@tool
def tool_get_weather(city: str) -> str:
    """Gets the weather for a city."""
    return get_weather(city)

@tool
def tool_calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    return calculate(expression)

@tool
def tool_save_note(note: str) -> str:
    """Saves a note to session memory."""
    return save_note(note)

@tool
def tool_get_notes() -> str:
    """Retrieves all saved notes."""
    return get_notes()


LANGCHAIN_TOOLS = [
    tool_get_time,
    tool_get_weather,
    tool_calculate,
    tool_save_note,
    tool_get_notes,
]


class LangChainVoiceAgent:
    """LangChain-powered Conversational Agent with Google GenAI and Tool Calling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key or config.api_key
        self.system_prompt = system_prompt or (
            "You are a helpful, spoken conversational voice agent. "
            "Keep answers concise, warm, conversational, and direct. "
            "Use your tools when asked about weather, current time, notes, or math."
        )
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature,
        )
        self.llm_with_tools = self.llm.bind_tools(LANGCHAIN_TOOLS)
        self.history: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt)
        ]

    async def chat(self, user_input: str) -> str:
        """Sends a message and returns the completed text response, handling tool calls."""
        self.history.append(HumanMessage(content=user_input))
        response = await self.llm_with_tools.ainvoke(self.history)
        self.history.append(response)

        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                tool_instance = next((t for t in LANGCHAIN_TOOLS if t.name == name), None)
                if tool_instance:
                    tool_output = await tool_instance.ainvoke(args)
                else:
                    tool_output = f"Tool {name} not found."
                
                # Append ToolMessage and get follow-up answer
                from langchain_core.messages import ToolMessage
                self.history.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_output)))

            final_response = await self.llm.ainvoke(self.history)
            self.history.append(final_response)
            return str(final_response.content)

        return str(response.content)

    async def stream_tokens(self, user_input: str) -> AsyncGenerator[str, None]:
        """Streams text tokens asynchronously for real-time TTS synthesis."""
        self.history.append(HumanMessage(content=user_input))
        async for chunk in self.llm.astream(self.history):
            if chunk.content:
                yield str(chunk.content)
