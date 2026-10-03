from mcp.server.mcpserver import MCPServer
from dotenv import load_dotenv
from app.mcp.email_sender import EmailSender

load_dotenv()

mcp = MCPServer("Movie Email Server")


@mcp.tool()
def send_email(
    recipient: str,
    subject: str,
    body: str,
) -> dict:
    """
    Send an email containing movie-related information.

    This development version validates the email request
    without actually sending an email.
    """

    if not recipient.strip():
        raise ValueError("Recipient cannot be empty.")

    if "@" not in recipient or "." not in recipient.split("@")[-1]:
        raise ValueError("Invalid email address.")

    if not subject.strip():
        raise ValueError("Subject cannot be empty.")

    if not body.strip():
        raise ValueError("Email body cannot be empty.")

    sender = EmailSender()

    return sender.send(
        recipient=recipient,
        subject=subject,
        body=body,
    )