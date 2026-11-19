"""The one error type of the package."""


class MakerStdError(RuntimeError):
    """A pinned maker file is missing or differs from its pin, a file cannot be read as written, or the stored
    placement differs from the one derived from the pinned texts. The grader reports it as a task error."""
