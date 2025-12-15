"""Action emitter for creating custom OpenTelemetry spans."""

import json
from contextvars import ContextVar
from typing import Any, Optional

from opentelemetry import trace

from .config import VALUE_ACTION_ATTRIBUTES
from .span_processor import reset_user_context, set_user_context

_current_action_context: ContextVar[Optional["ActionContext"]] = ContextVar("_current_action_context", default=None)


class ActionEmitter:
    """Emitter for creating custom actions (OpenTelemetry spans)."""

    def __init__(self, tracer: trace.Tracer):
        """
        Initialize the action emitter.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def send(
        self,
        action_name: str,
        anonymous_id: str,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Send an action immediately as an OpenTelemetry span.

        Creates a span, adds attributes, and immediately ends it (sends it).

        If called within a 'with actions.start(...)' context, user_id and anonymous_id
        are inherited from the context and should not be provided.

        Args:
            action_name: Name of the action
            anonymous_id: Anonymous ID (required if not in a start() context)
            user_id: User ID (optional)
            **kwargs: Additional attributes for the action
        """
        current_context = _current_action_context.get()

        if current_context:
            self._send_action(
                action_name=action_name,
                anonymous_id=current_context._anonymous_id,
                user_id=current_context._user_id,
                **kwargs,
            )
        else:
            self._send_action(
                action_name=action_name,
                anonymous_id=anonymous_id,
                user_id=user_id,
                **kwargs,
            )

    def _send_action(
        self,
        action_name: str,
        anonymous_id: str,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Internal method to send an action.

        Args:
            action_name: Name of the action
            anonymous_id: Anonymous ID for the action
            user_id: User ID for the action
            **kwargs: Additional attributes for the action
        """
        standard_attrs = {}
        non_standard_attrs = {}
        for key, value in kwargs.items():
            if key in VALUE_ACTION_ATTRIBUTES:
                standard_attrs[key] = value
            else:
                non_standard_attrs[key] = value

        if user_id:
            standard_attrs["value.action.user_id"] = user_id
        if anonymous_id:
            standard_attrs["value.action.anonymous_id"] = anonymous_id

        standard_attrs["value.action.name"] = action_name
        standard_attrs["value.action.user_attributes"] = json.dumps(non_standard_attrs)

        with self._tracer.start_as_current_span(name="value.action", attributes=standard_attrs):
            pass


class ActionContext:
    """Context manager for action contexts that allows sending multiple actions."""

    def __init__(
        self,
        emitter: ActionEmitter,
        anonymous_id: str,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize an action context.

        Args:
            emitter: The ActionEmitter instance
            anonymous_id: Anonymous ID for the action context
            user_id: User ID for the action context
            **kwargs: Additional attributes for the context
        """
        self._emitter = emitter
        self._user_id = user_id
        self._anonymous_id = anonymous_id
        self._attributes = kwargs
        self._token = None
        self._action_sent = False
        self._user_context_tokens = None

    def __enter__(self) -> Any:
        """Enter the context and set user context."""
        self._token = _current_action_context.set(self)
        self._user_context_tokens = set_user_context(user_id=self._user_id, anonymous_id=self._anonymous_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and reset user context."""
        if self._user_context_tokens:
            reset_user_context(*self._user_context_tokens)
        if self._token:
            _current_action_context.reset(self._token)
        return False

    def send(self, action_name: str, **kwargs: Any) -> None:
        """
        Send an action within this context.

        Args:
            action_name: Name of the action
            **kwargs: Additional attributes for the action
        """
        self._action_sent = True
        self._emitter.send(
            action_name=action_name,
            anonymous_id=self._anonymous_id,
            user_id=self._user_id,
            **kwargs,
        )
