import asyncio

from mcp import Client
from app.mcp.email_server import mcp


async def run_test(
    client,
    test_name: str,
    arguments: dict,
):
    result = await client.call_tool(
        "send_email",
        arguments,
    )

    print(f"\n{test_name}")

    if result.is_error:
        print("ERROR: Tool rejected the request.")
    else:
        print("SUCCESS:")
        print(result)


async def main():
    async with Client(mcp) as client:

        # -----------------------------------------------------
        # TEST 1: Valid email
        # -----------------------------------------------------

        await run_test(
            client,
            "TEST 1: Valid email",
            {
                "recipient": "test@example.com",
                "subject": "Movie Scene Breakdown",
                "body": (
                    "This is a test email containing "
                    "a movie scene breakdown."
                ),
            },
        )

        # -----------------------------------------------------
        # TEST 2: Empty recipient
        # -----------------------------------------------------

        await run_test(
            client,
            "TEST 2: Empty recipient",
            {
                "recipient": "",
                "subject": "Test",
                "body": "Test body",
            },
        )

        # -----------------------------------------------------
        # TEST 3: Invalid email
        # -----------------------------------------------------

        await run_test(
            client,
            "TEST 3: Invalid email",
            {
                "recipient": "not-an-email",
                "subject": "Test",
                "body": "Test body",
            },
        )

        # -----------------------------------------------------
        # TEST 4: Empty subject
        # -----------------------------------------------------

        await run_test(
            client,
            "TEST 4: Empty subject",
            {
                "recipient": "test@example.com",
                "subject": "",
                "body": "Test body",
            },
        )

        # -----------------------------------------------------
        # TEST 5: Empty body
        # -----------------------------------------------------

        await run_test(
            client,
            "TEST 5: Empty body",
            {
                "recipient": "test@example.com",
                "subject": "Test",
                "body": "",
            },
        )


if __name__ == "__main__":
    asyncio.run(main())