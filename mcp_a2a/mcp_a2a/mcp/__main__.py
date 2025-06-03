from typing import Dict

from fastmcp import FastMCP
from forex_python.converter import CurrencyRates

# Initialize currency converter
c = CurrencyRates()

# Create FastMCP server
mcp = FastMCP(
    title="Currency Conversion MCP Server",
    description="An MCP server that provides currency conversion tools and resources",
    host='0.0.0.0',
    port=8080
)

# List of supported currencies
SUPPORTED_CURRENCIES = {
        "USD": {"name": "US Dollar", "symbol": "$", "country": "United States"},
        "EUR": {"name": "Euro", "symbol": "€", "country": "Eurozone"},
        "GBP": {"name": "British Pound", "symbol": "£", "country": "United Kingdom"},
        "JPY": {"name": "Japanese Yen", "symbol": "¥", "country": "Japan"},
        "AUD": {"name": "Australian Dollar", "symbol": "A$", "country": "Australia"},
        "CAD": {"name": "Canadian Dollar", "symbol": "C$", "country": "Canada"},
        "CHF": {"name": "Swiss Franc", "symbol": "Fr", "country": "Switzerland"},
        "CNY": {"name": "Chinese Yuan", "symbol": "¥", "country": "China"},
        "SEK": {"name": "Swedish Krona", "symbol": "kr", "country": "Sweden"},
        "NZD": {"name": "New Zealand Dollar", "symbol": "NZ$", "country": "New Zealand"},
    }
 
@mcp.tool(
    name="get_conversion_rate",
    description="Get the conversion rate between two currencies",
)
async def get_conversion_rate(from_currency: str, to_currency: str) -> float:
    """Get the conversion rate between two currencies."""
   
    # Validate currencies
    if from_currency not in SUPPORTED_CURRENCIES.keys():
        raise ValueError("Unsupported to_currency: {}".format(from_currency))
    
    if to_currency not in SUPPORTED_CURRENCIES.keys():
        raise ValueError("Unsupported to_currency: {}".format(to_currency))
    
    # Get the conversion rate
    rate = c.get_rate(from_currency, to_currency)
        
    return rate
        

@mcp.resource(
    "currency://{currency}",
    description="Get information about a specific currency",
    mime_type="application/json", # Explicit MIME type
)
async def get_currency_info(currency: str) -> Dict:
    """Return information about a specific currency."""
    
    if currency not in SUPPORTED_CURRENCIES.keys():
        raise ValueError("Unsupported currency: {}.".format(currency))
    
    return SUPPORTED_CURRENCIES[currency]

mcp.run(transport="streamable-http")