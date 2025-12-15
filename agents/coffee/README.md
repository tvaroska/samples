# Coffee Barista Agent

A conversational AI barista agent built with Google's Agent Development Kit (ADK) that takes coffee orders and provides pricing information.

## Overview

This agent simulates a coffee shop ordering system where customers can:
- Browse the coffee menu
- Place orders for various coffee types and sizes
- Get instant pricing information
- Complete their coffee order transaction

## Features

- **Natural Language Ordering**: Interact with the barista using conversational language
- **Dynamic Pricing**: Automatic price calculation based on coffee type and size
- **Menu Validation**: Ensures orders match available menu items
- **Multi-size Support**: Different sizing options for different coffee types

## Menu

### Coffee Types

1. **Espresso**
   - Single: $1.00
   - Double: $1.50

2. **Capucino**
   - Small: $1.00
   - Medium: $1.50
   - Large: $2.00
   - XL: $3.00

3. **Americano**
   - Small: $1.00
   - Medium: $1.50
   - Large: $2.00
   - XL: $3.00

4. **Iced Americano**
   - Small: $1.00
   - Medium: $1.50
   - Large: $2.00
   - XL: $3.00

## Project Structure

```
coffee/
├── __init__.py       # Package initialization
├── agent.py          # Main agent definition and pricing tool
├── prompt.py         # Barista instruction prompt
└── README.md         # This file
```

## Files Description

### `agent.py`

Contains:
- **`pricing(type: str, size: str) -> float`**: Tool function that returns the price for a given coffee type and size
- **`barista`**: LlmAgent instance configured with the barista persona and pricing tool
- **`root_agent`**: Entry point for the agent application

### `prompt.py`

Defines the `BARISTA` instruction prompt that gives the agent its personality and role as a coffee shop ordering assistant.

## Usage

### Local Development

**Interactive Terminal Mode:**
```bash
adk run
```

**Web Interface:**
```bash
adk web
```

**API Server:**
```bash
adk api_server
```

### Example Interactions

```
User: Hi, I'd like to order a coffee
Barista: Hello! Welcome to our coffee shop. What would you like to order today?

User: I'll have a large cappuccino
Barista: Great choice! A large Capucino will be $2.00. Would you like to add anything else?

User: No, that's all
Barista: Perfect! Your total is $2.00. Enjoy your coffee!
```

## Technical Details

### Agent Configuration

- **Model**: `gemini-2.5-flash`
- **Type**: `LlmAgent`
- **Tools**:
  - `pricing`: Function tool for price calculation

### Pricing Function

The `pricing()` function uses a lookup table to determine prices based on coffee type and size combinations. It expects exact matches for:
- Coffee type (e.g., "Espresso", "Capucino", "Americano", "Iced Americano")
- Size (e.g., "Single", "Double", "Small", "Medium", "Large", "XL")

## Requirements

```bash
pip install google-adk
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# For Google AI Studio
GOOGLE_API_KEY=your_api_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# OR for Vertex AI
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

## Future Enhancements

Potential improvements:
- Add more coffee varieties (Latte, Mocha, Flat White, etc.)
- Implement customization options (milk type, sugar, syrups)
- Add order history tracking via session state
- Implement payment processing simulation
- Support for multiple items in a single order
- Order summary and receipt generation

## Notes

- Coffee type and size names are case-sensitive and must match exactly
- The agent is designed for educational and demonstration purposes
- Pricing is in USD

## License

This agent is part of a learning project for Google ADK development.
