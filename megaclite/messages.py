"""This module defines all messages, that can be sent between client and server."""
import enum
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClientInfo:
    """Used to send environment info of the client to the server."""

    python_version: str
    packages: Optional[list[str]] = field(default_factory=list)


@dataclass
class TrainingJob:
    """The TrainingJob message is sent by the client to server to train a new model."""

    cell: str
    model_name: str
    state: bytes
    client: ClientInfo
    mig_slices: int
    uuid: Optional[str] = None


@dataclass
class AbortJob:
    """Sent by the client to abort training of the specified job uuid."""

    uuid: str


@dataclass
class ShellJob:
    """Run the provided shell command on the server."""

    command: str
    client: ClientInfo
    uuid: Optional[str] = None


@dataclass
class StdOut:
    """Used to send stdout of jobs back to the client."""

    line: str

@dataclass
class StdErr:
    """Used to send stderr of jobs back to the client."""

    line: str

# pylint: disable=too-few-public-methods
class EOF:
    """Used as a poison pill to close the std out stream."""


class JobState(enum.Enum):
    """Represents the current state of a job on the server."""

    PENDING = 1
    STARTED = 2
    REJECTED = 3
    FINISHED = 4
    ABORTED = 5
    FAILED = 6


    @property
    def exited(self):
        """Return true if the job has exited, for any reason."""
        return self in set([JobState.REJECTED, JobState.FINISHED, JobState.ABORTED, JobState.FAILED])


@dataclass
class JobInfo:
    """Provides current state and number in queue for a job."""

    state: JobState
    no_in_queue: int
    uuid: str


@dataclass
class JobResult:
    """Used to return model weights to the client."""

    result: bytes
