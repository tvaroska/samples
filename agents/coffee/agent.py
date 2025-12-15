from typing import List
import asyncio

from pydantic import BaseModel

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

class Item(BaseModel):
    type: str
    size: str
    volume: int
    price: float

class Receipt(BaseModel):
    items: List[Item]
    subtotal: float
    tax: float
    total: float


BARISTA = """
You are a friendly and enthusiastic barista at a cozy coffee shop. Your goal is to make each customer feel
welcome while efficiently taking their order and processing payment. Be warm, conversational, and helpful!

Your workflow:
1. Greet the customer warmly and ask what they'd like today
2. Get their drink order (type and size) - ask clarifying questions if needed
3. IMMEDIATELY call pricing(type, size) for each drink as soon as you have complete information
4. Confirm what was added and let them know the cost of the dring and full cost of the order (total)
5. Ask "Would you like anything else?" or "Is that everything for you today?"
6. Repeat steps 2-5 until they confirm the order is complete
7. Share the total breakdown (subtotal, tax, total) and call payment() to process
8. Share their order number from the payment result and thank them warmly!

Our menu:
- Espresso (Single or Double)
- Cappuccino (Small, Medium, Large, XL)
- Americano (Small, Medium, Large, XL)
- Iced Americano (Small, Medium, Large, XL)

Current Order:
Items: {receipt.items?No items yet}
Subtotal: ${receipt.subtotal?0.00}
Tax: ${receipt.tax?0.00}
Total: ${receipt.total?0.00}

Important:
- Use EXACT menu names when calling pricing(): "Espresso", "Cappuccino", "Americano", "Iced Americano"
- Use EXACT size names: "Single", "Double", "Small", "Medium", "Large", "XL"
- Call pricing() immediately when you know type + size (don't wait!)
- Only call payment() after customer confirms order is complete
- If pricing() returns an error, politely ask them to respecify from the menu
- If customer is unsure, offer suggestions or describe popular items
"""

async def auto_save_session_to_memory_callback(callback_context):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session)


async def pricing(type: str, size: str, tool_context: ToolContext) -> str:
    """

        Coffee pricing function. Based on type and size will update the receipt and return if the transaction is OK

    """
    pricing_table = {
        ("Espresso", "Single"): 1.0,
        ("Espresso", "Double"): 1.5,
        ("Cappuccino", "Small"): 1.0,
        ("Cappuccino", "Medium"): 1.5,
        ("Cappuccino", "Large"): 2.0,
        ("Cappuccino", "XL"): 3.0,
        ("Americano", "Small"): 1.0,
        ("Americano", "Medium"): 1.5,
        ("Americano", "Large"): 2.0,
        ("Americano", "XL"): 3.0,
        ("Iced Americano", "Small"): 1.0,
        ("Iced Americano", "Medium"): 1.5,
        ("Iced Americano", "Large"): 2.0,
        ("Iced Americano", "XL"): 3.0,
    }

    # Volume mapping for different sizes
    volume_table = {
        "Single": 30,
        "Double": 60,
        "Small": 240,
        "Medium": 360,
        "Large": 480,
        "XL": 600,
    }

    try:
        price = pricing_table[(type, size)]
        volume = volume_table[size]

        # Initialize receipt if it doesn't exist
        if 'receipt' not in tool_context.state:
            tool_context.state['receipt'] = Receipt(
                items=[],
                subtotal=0.0,
                tax=0.0,
                total=0.0
            )

        # Get current receipt
        receipt = tool_context.state['receipt']

        # Create new item
        new_item = Item(
            type=type,
            size=size,
            volume=volume,
            price=price
        )

        # Add item to receipt
        receipt.items.append(new_item)

        # Recalculate totals
        receipt.subtotal = sum(item.price for item in receipt.items)
        receipt.tax = round(receipt.subtotal * 0.08, 2)  # 8% tax
        receipt.total = round(receipt.subtotal + receipt.tax, 2)

        # Update state
        tool_context.state['receipt'] = receipt

        return f"OK - Added {size} {type} for ${price:.2f}. Current subtotal: ${receipt.subtotal:.2f}"
    except KeyError:
        return "Error, unknown type or size"

async def payment(tool_context: ToolContext) -> str:
    """

        Payment and order confirmation. Returns order number with receipt details or Fail if payment failed

    """
    # Simulate payment processing
    await asyncio.sleep(2)

    # Get receipt from state
    receipt = tool_context.state.get('receipt')

    if not receipt or not receipt.items:
        return "Error: No items in order. Please add items before payment."

    # Generate order number (in production, this would be more sophisticated)
    import random
    order_number = random.randint(1, 999)

    # Clear the receipt after successful payment
    tool_context.state['receipt'] = Receipt(items=[], subtotal=0.0, tax=0.0, total=0.0)

    return f"Payment successful! Order #{order_number}. Total charged: ${receipt.total:.2f} (Subtotal: ${receipt.subtotal:.2f}, Tax: ${receipt.tax:.2f})"


barista = LlmAgent(
    name = 'barista',
    model = 'gemini-2.5-flash',
    description="Cofee ordering agent",
    instruction=BARISTA,
    tools=[pricing, payment, PreloadMemoryTool()],
    after_agent_callback=auto_save_session_to_memory_callback,

)

root_agent = barista