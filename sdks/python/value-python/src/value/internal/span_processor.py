"""Custom span processor for propagating user context to child spans."""

from contextvars import ContextVar
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

_user_id_context: ContextVar[Optional[str]] = ContextVar("_user_id_context", default=None)
_anonymous_id_context: ContextVar[Optional[str]] = ContextVar("_anonymous_id_context", default=None)


class UserContextSpanProcessor(SpanProcessor):
    """
    Span processor that adds user_id and anonymous_id attributes to all spans
    created within an action_context.
    """

    def on_start(self, span: ReadableSpan, parent_context: Optional[Context] = None) -> None:
        """
        Called when a span is started. Adds user context attributes if available.

        Args:
            span: The span that was started
            parent_context: The parent context (optional)
        """
        user_id = _user_id_context.get()
        anonymous_id = _anonymous_id_context.get()

        if user_id and span.is_recording():
            span.set_attribute("value.action.user_id", user_id)
        if anonymous_id and span.is_recording():
            span.set_attribute("value.action.anonymous_id", anonymous_id)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended."""
        pass

    def shutdown(self) -> None:
        """Called when the processor is shut down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True


def set_user_context(user_id: Optional[str] = None, anonymous_id: Optional[str] = None):
    """
    Set user context for the current execution context.

    Args:
        user_id: User ID to set
        anonymous_id: Anonymous ID to set

    Returns:
        Tuple of tokens to reset the context later
    """
    user_token = _user_id_context.set(user_id) if user_id else None
    anon_token = _anonymous_id_context.set(anonymous_id) if anonymous_id else None
    return user_token, anon_token


def reset_user_context(user_token, anon_token):
    """
    Reset user context to previous values.

    Args:
        user_token: Token from set_user_context for user_id
        anon_token: Token from set_user_context for anonymous_id
    """
    if user_token:
        _user_id_context.reset(user_token)
    if anon_token:
        _anonymous_id_context.reset(anon_token)
