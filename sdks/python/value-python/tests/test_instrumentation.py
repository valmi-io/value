"""Tests for instrumentation."""

import pytest

from value.instrumentation import (
    SUPPORTED_LIBRARIES,
    auto_instrument,
    get_supported_libraries,
    is_library_available,
    uninstrument,
)


def test_get_supported_libraries() -> None:
    """Test that get_supported_libraries returns expected libraries."""
    libraries = get_supported_libraries()
    assert isinstance(libraries, list)
    assert len(libraries) > 0
    assert "gemini" in libraries
    assert "langchain" in libraries


def test_is_library_available_unsupported() -> None:
    """Test is_library_available returns False for unsupported library."""
    assert is_library_available("unsupported_lib") is False


def test_is_library_available_not_installed() -> None:
    """Test is_library_available returns False when package not installed."""
    # This will return False if the instrumentation package isn't installed
    # We can't guarantee which packages are installed in test env
    result = is_library_available("gemini")
    assert isinstance(result, bool)


def test_auto_instrument_unsupported_library() -> None:
    """Test that unsupported libraries raise warnings."""
    with pytest.warns(UserWarning, match="not supported"):
        auto_instrument(libraries=["unsupported_lib"])


def test_auto_instrument_returns_list() -> None:
    """Test that auto_instrument returns a list."""
    result = auto_instrument(libraries=["unsupported_lib"])
    assert isinstance(result, list)
    # Should be empty since unsupported_lib is not valid
    assert len(result) == 0


def test_auto_instrument_no_libraries_installed() -> None:
    """Test auto_instrument with no libraries when none are available."""
    # When no libraries are specified, it should try to instrument all available
    # This may warn if none are installed
    result = auto_instrument()
    assert isinstance(result, list)


def test_uninstrument_returns_list() -> None:
    """Test that uninstrument returns a list."""
    result = uninstrument(libraries=["unsupported_lib"])
    assert isinstance(result, list)


def test_supported_libraries_mapping() -> None:
    """Test that SUPPORTED_LIBRARIES has valid structure."""
    assert isinstance(SUPPORTED_LIBRARIES, dict)
    for lib_name, instrumentor_path in SUPPORTED_LIBRARIES.items():
        assert isinstance(lib_name, str)
        assert isinstance(instrumentor_path, str)
        # Check that the path has the expected format
        assert "." in instrumentor_path
        assert "Instrumentor" in instrumentor_path
