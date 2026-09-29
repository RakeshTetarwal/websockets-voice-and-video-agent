import math
from datetime import datetime, timezone


# In-memory notepad for the agent session
_SESSION_NOTES = []


def get_current_time() -> str:
    """Returns the current UTC date and time."""
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")


def get_weather(city: str) -> str:
    """Gets the current weather condition and temperature for a city.
    
    Args:
        city: The name of the city or location (e.g. 'San Francisco', 'London', 'Tokyo').
    """
    # Deterministic mock weather for demonstration
    city_lower = city.lower()
    if "san francisco" in city_lower:
        return "San Francisco: 64°F (18°C), Partly Cloudy, Wind 12 mph NW, Humidity 68%."
    elif "tokyo" in city_lower:
        return "Tokyo: 72°F (22°C), Sunny, Wind 5 mph E, Humidity 45%."
    elif "london" in city_lower:
        return "London: 58°F (14°C), Light Rain, Wind 8 mph W, Humidity 82%."
    elif "bangalore" in city_lower or "bengaluru" in city_lower:
        return "Bengaluru: 77°F (25°C), Pleasant & Breezy, Wind 10 mph SW, Humidity 55%."
    return f"{city}: 70°F (21°C), Clear skies with mild breeze."


def calculate(expression: str) -> str:
    """Evaluates a mathematical expression safely.
    
    Args:
        expression: The math expression to evaluate, like '25 * 40' or 'sqrt(144)'.
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


def save_note(note: str) -> str:
    """Saves a user's note or reminder to the session memory.
    
    Args:
        note: The note text or reminder to remember.
    """
    _SESSION_NOTES.append(note)
    return f"Saved note: '{note}'. Total notes: {len(_SESSION_NOTES)}"


def get_notes() -> str:
    """Retrieves all saved notes from the current session."""
    if not _SESSION_NOTES:
        return "You have no saved notes yet."
    return "\n".join(f"{i+1}. {n}" for i, n in enumerate(_SESSION_NOTES))


# Registry of tools passed to the Gemini Live session
AVAILABLE_TOOLS = [
    get_current_time,
    get_weather,
    calculate,
    save_note,
    get_notes,
]
