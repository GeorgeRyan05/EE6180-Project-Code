import logging
class DuplicateFilter(logging.Filter):
    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._past_messages = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg in self._past_messages:
            return False
        self._past_messages.add(record.msg)
        return True