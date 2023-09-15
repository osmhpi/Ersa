"""This module defines all messages, that can be sent between client and server."""
import enum
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClientInfo:
    """Used to send environment info of the client to the server."""

    python_version: str
    user_name: str
    packages: Optional[list[str]] = field(default_factory=list)


@dataclass
class TrainingJobRequest:
    """Sent by the client, to request resources for a training job."""

    client: ClientInfo
    mig_slices: int
    state_size: int


@dataclass
class TrainingJob:
    """The TrainingJob message is sent by the client to server to train a new model."""

    cell: str
    model_name: str
    state: bytes
    client: ClientInfo  # TODO: remove
    mig_slices: int  # TODO: remove
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


    # # the job request has been accepted and put into a queue
    # ACCEPTED = enum.auto()
    # # the server is preparing the environment for the job
    # PREPARING_ENVIRONMENT = enum.auto()
    # # the environment is ready
    # ENVIRONMENT_READY = enum.auto()
    # # the server is ready for the state data transfer from the client
    # READY4DATA = enum.auto()
    # # the job is ready to be executed
    # READY = enum.auto()
    # # the job is currently running
    # RUNNING = enum.auto()
    # # the job has been rejected for any reason
    # REJECTED = enum.auto()
    # # the job was completed successfully
    # FINISHED = enum.auto()
    # # the job was aborted (by the client) at any time
    # ABORTED = enum.auto()

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
