class EmailSender:
    def __init__(self, mock: bool = True):
        self.mock = mock

    def send(self, recipient: str, subject: str, body: str) -> dict:
        if self.mock:
            return {
                "success": True,
                "status": "sent",
                "transport": "mock",
                "recipient": recipient,
                "subject": subject,
                "body": body,
            }

        raise NotImplementedError(
            "Real email transport is not implemented yet."
        )