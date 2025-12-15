"""Auto-instrumentation for supported libraries."""

import warnings
from typing import Optional

# Mapping of library names to their instrumentor classes
SUPPORTED_LIBRARIES = {
    "gemini": "opentelemetry.instrumentation.google_generativeai.GoogleGenerativeAiInstrumentor",
    "langchain": "opentelemetry.instrumentation.langchain.LangChainInstrumentor",
}

# Mapping of library names to their required extras
LIBRARY_EXTRAS = {
    "gemini": "genai",
    "langchain": "langchain",
}


def get_supported_libraries() -> list[str]:
    """Return a list of supported library names for auto-instrumentation."""
    return list(SUPPORTED_LIBRARIES.keys())


def is_library_available(library: str) -> bool:
    """
    Check if the instrumentation for a library is available.

    Args:
        library: Name of the library to check (e.g., "gemini", "langchain").

    Returns:
        True if the instrumentation package is installed, False otherwise.
    """
    if library not in SUPPORTED_LIBRARIES:
        return False

    try:
        module_path, class_name = SUPPORTED_LIBRARIES[library].rsplit(".", 1)
        __import__(module_path, fromlist=[class_name])
        return True
    except ImportError:
        return False


def auto_instrument(libraries: Optional[list[str]] = None) -> list[str]:
    """
    Enable auto-instrumentation for supported libraries.

    This function dynamically imports and enables OpenTelemetry instrumentors
    for the specified libraries. If no libraries are specified, it will attempt
    to instrument all supported libraries that are installed.

    Args:
        libraries: List of library names to instrument (e.g., ["langchain", "gemini"]).
                  If None, attempts to instrument all supported libraries that are available.

    Returns:
        List of successfully instrumented library names.

    Raises:
        ImportError: If a specified library's instrumentation package is not installed.

    Examples:
        # Instrument specific libraries
        >>> auto_instrument(["gemini", "langchain"])

        # Instrument all available libraries
        >>> auto_instrument()

        # Instrument only gemini
        >>> auto_instrument(["gemini"])
    """
    if libraries is None:
        # When no libraries specified, only instrument those that are available
        libraries_to_instrument = [lib for lib in SUPPORTED_LIBRARIES.keys() if is_library_available(lib)]
        if not libraries_to_instrument:
            warnings.warn(
                "No auto-instrumentation libraries are installed. "
                "Install extras with: pip install value-python[genai] "
                "or pip install value-python[all]"
            )
            return []
    else:
        libraries_to_instrument = libraries

    instrumented: list[str] = []

    for lib in libraries_to_instrument:
        if lib not in SUPPORTED_LIBRARIES:
            warnings.warn(
                f"Instrumentation for '{lib}' is not supported. "
                f"Supported libraries: {list(set(SUPPORTED_LIBRARIES.keys()))}"
            )
            continue

        try:
            # Dynamically import the instrumentor class
            module_path, class_name = SUPPORTED_LIBRARIES[lib].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            instrumentor_class = getattr(module, class_name)

            # Check if already instrumented
            instrumentor = instrumentor_class()
            if hasattr(instrumentor, "is_instrumented_by_opentelemetry"):
                if instrumentor.is_instrumented_by_opentelemetry:
                    # Already instrumented, skip
                    instrumented.append(lib)
                    continue

            # Instrument the library
            instrumentor.instrument()
            instrumented.append(lib)

        except ImportError as e:
            extra = LIBRARY_EXTRAS.get(lib, "all")
            raise ImportError(
                f"Could not import instrumentor for '{lib}'. "
                f"Install it with: pip install value-python[{extra}]\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            warnings.warn(f"Failed to instrument {lib}: {e}")

    return instrumented


def uninstrument(libraries: Optional[list[str]] = None) -> list[str]:
    """
    Disable auto-instrumentation for specified libraries.

    Args:
        libraries: List of library names to uninstrument. If None, uninstruments all.

    Returns:
        List of successfully uninstrumented library names.
    """
    if libraries is None:
        libraries_to_uninstrument = list(SUPPORTED_LIBRARIES.keys())
    else:
        libraries_to_uninstrument = libraries

    uninstrumented: list[str] = []

    for lib in libraries_to_uninstrument:
        if lib not in SUPPORTED_LIBRARIES:
            continue

        try:
            module_path, class_name = SUPPORTED_LIBRARIES[lib].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            instrumentor_class = getattr(module, class_name)
            instrumentor_class().uninstrument()
            uninstrumented.append(lib)
        except ImportError:
            # Not installed, nothing to uninstrument
            pass
        except Exception as e:
            warnings.warn(f"Failed to uninstrument {lib}: {e}")

    return uninstrumented
