class PrivacyError(RuntimeError):
    """A safe-to-display error; never include source text or provider response bodies."""
