# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""youtube-agent - A Bindu Agent for YouTube video analysis and structured summaries."""

from youtube_agent.__version__ import __version__
from youtube_agent.main import (
    handler,
    initialize_agent,
    main,
    run_agent,
    cleanup,
    APIKeyError,
)

__all__ = [
    "__version__",
    "handler",
    "initialize_agent",
    "run_agent",
    "cleanup",
    "main",
    "APIKeyError",
]