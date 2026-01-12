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
    APIKeyError,
    cleanup,
    handler,
    initialize_agent,
    main,
    run_agent,
)

__all__ = [
    "APIKeyError",
    "__version__",
    "cleanup",
    "handler",
    "initialize_agent",
    "main",
    "run_agent",
]
