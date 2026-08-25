import requests

def get_exchange_rates(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """
    Get the latest exchange rate between two currencies and calculate the converted amount.

    Args:
        from_currency: The three-letter currency code to convert from (e.g., "USD").
        to_currency: The three-letter currency code to convert to (e.g., "EUR").
        amount: The amount of money to convert. Defaults to 1.0.

    Returns:
        A string with the conversion result or an error message.
    """
    if not from_currency or not to_currency:
        return "Error: Both 'from_currency' and 'to_currency' must be provided."

    base_url = "https://api.frankfurter.app/latest"
    params = {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        if "rates" not in data or to_currency.upper() not in data["rates"]:
            return f"Could not find exchange rate for {from_currency} to {to_currency}."

        converted_amount = data["rates"][to_currency.upper()]
        # Calculate the rate for 1 unit if a different amount was provided
        rate = converted_amount / amount if amount else 0

        return (
            f"Exchange Rate ({from_currency.upper()} to {to_currency.upper()}):\n"
            f"- 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}\n"
            f"- {amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}"
        )

    except requests.Timeout:
        return "The currency conversion service request timed out. Please try again."
    except requests.HTTPError as e:
        # The API returns helpful JSON errors
        try:
            error_data = e.response.json()
            error_message = error_data.get("message", "Unknown API error")
            return f"Currency API returned an error: {error_message}"
        except ValueError:
             return f"Currency API returned an HTTP error: {e.response.status_code}"
    except requests.RequestException as e:
        return f"Could not connect to the currency conversion service: {e}"

if __name__ == "__main__":
    # Example usage:
    print(get_exchange_rates.invoke({"from_currency": "USD", "to_currency": "EUR", "amount": 100}))
    print("\n" + "=" * 80 + "\n")
    print(get_exchange_rates.invoke({"from_currency": "GBP", "to_currency": "JPY"}))
    print("\n" + "=" * 80 + "\n")
    # Example of an error from the API
    print(get_exchange_rates.invoke({"from_currency": "INVALID", "to_currency": "USD"}))
