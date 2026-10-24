"""Custom error hierarchy and CLI exit codes for DEICIDE."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Meaningful process exit codes for the DEICIDE CLI."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIGURATION_ERROR = 2
    REPOSITORY_ERROR = 3
    PROVIDER_ERROR = 4
    BUDGET_EXCEEDED = 5
    SECURITY_VIOLATION = 6


class DeicideError(Exception):
    """Base exception for all expected DEICIDE errors."""

    exit_code: ExitCode = ExitCode.GENERAL_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(DeicideError):
    """Raised when configuration or CLI arguments are invalid."""

    exit_code = ExitCode.CONFIGURATION_ERROR


class MissingCredentialsError(ConfigurationError):
    """Raised when required API credentials are missing or external code egress is not authorized."""

    exit_code = ExitCode.CONFIGURATION_ERROR


class UnsupportedProviderError(ConfigurationError):
    """Raised when an unsupported LLM provider is requested."""

    exit_code = ExitCode.CONFIGURATION_ERROR


class RepositoryError(DeicideError):
    """Raised when the target directory is not a valid Git repository or a Git operation fails."""

    exit_code = ExitCode.REPOSITORY_ERROR


class InvalidRevisionError(RepositoryError):
    """Raised when a specified Git revision does not exist or is invalid."""

    exit_code = ExitCode.REPOSITORY_ERROR


class SecurityViolationError(DeicideError):
    """Raised when path traversal, symlink escape, or another trust boundary violation is detected."""

    exit_code = ExitCode.SECURITY_VIOLATION


class ProviderError(DeicideError):
    """Raised when an AI provider request fails, times out, or returns an unrecoverable error."""

    exit_code = ExitCode.PROVIDER_ERROR


class RateLimitError(ProviderError):
    """Raised when an AI provider rate limit is exceeded after bounded retries."""

    exit_code = ExitCode.PROVIDER_ERROR


class BudgetExceededError(DeicideError):
    """Raised when an execution budget (tokens, requests, context bytes, or elapsed time) is exceeded."""

    exit_code = ExitCode.BUDGET_EXCEEDED


class QuotaExhaustedError(BudgetExceededError):
    """Raised when a persistent API quota (RPD, RPM, or TPM) is exhausted."""

    exit_code = ExitCode.BUDGET_EXCEEDED

    def __init__(self, message: str, *, limit_type: str = "quota") -> None:
        super().__init__(message)
        self.limit_type = limit_type
