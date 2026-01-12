# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""youtube-agent - A Bindu Agent for YouTube video analysis and structured summaries."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.mem0 import Mem0Tools
from agno.tools.youtube import YouTubeTools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global instances
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


class APIKeyError(ValueError):
    """API key is missing."""


def load_config() -> dict:
    """Load agent configuration from project root."""
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {e}")
                continue

    # Default configuration
    return {
        "name": "youtube-agent",
        "description": "AI agent that analyzes YouTube videos and creates structured summaries with accurate timestamps. Extracts key insights from video content for easy navigation.",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {
                "key": "OPENROUTER_API_KEY",
                "description": "OpenRouter API key for LLM calls (required)",
                "required": True,
            },
            {
                "key": "MODEL_NAME",
                "description": "Model ID for OpenRouter (default: openai/gpt-4o)",
                "required": False,
            },
            {
                "key": "MEM0_API_KEY",
                "description": "Mem0 API key for conversation memory",
                "required": False,
            },
        ],
    }


def _get_api_keys() -> tuple[str | None, str | None, str]:
    """Get API keys and configuration from environment."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    mem0_api_key = os.getenv("MEM0_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")
    return openrouter_api_key, mem0_api_key, model_name


def _create_llm_model(openrouter_api_key: str, model_name: str) -> OpenRouter:
    """Create and return the OpenRouter model."""
    if not openrouter_api_key:
        error_msg = (
            "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.\n"
            "Get an API key from: https://openrouter.ai/keys"
        )
        raise APIKeyError(error_msg)

    return OpenRouter(
        id=model_name,
        api_key=openrouter_api_key,
        cache_response=True,
        supports_native_structured_outputs=True,
    )


def _setup_tools(mem0_api_key: str | None) -> list:
    """Set up all tools for the YouTube agent."""
    tools = []

    # YouTubeTools for video analysis
    try:
        youtube_tools = YouTubeTools()
        tools.append(youtube_tools)
        print("🎬 YouTube analysis enabled for video transcripts and metadata")
    except Exception as e:
        print(f"❌ Failed to initialize YouTubeTools: {e}")
        raise

    # Mem0 is optional for conversation memory
    if mem0_api_key:
        try:
            mem0_tools = Mem0Tools(api_key=mem0_api_key)
            tools.append(mem0_tools)
            print("🧠 Mem0 memory system enabled for conversation context")
        except Exception as e:
            print(f"⚠️  Mem0 initialization issue: {e}")

    return tools


async def initialize_agent() -> None:
    """Initialize the YouTube analysis agent."""
    global agent

    openrouter_api_key, mem0_api_key, model_name = _get_api_keys()

    # Validate required API keys
    if not openrouter_api_key:
        error_msg = (
            "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.\n"
            "Get an API key from: https://openrouter.ai/keys"
        )
        raise APIKeyError(error_msg)

    model = _create_llm_model(openrouter_api_key, model_name)
    tools = _setup_tools(mem0_api_key)

    # Create the YouTube analysis agent
    agent = Agent(
        name="YouTube Video Analyst",
        model=model,
        tools=tools,
        description=dedent("""\
            You are an expert YouTube content analyst with a keen eye for detail! 🎓
            
            You specialize in analyzing YouTube videos and creating structured summaries
            with accurate timestamps to make video content easily navigable and searchable.
        """),
        instructions=dedent("""\
            YOUTUBE ANALYSIS PROCESS:

            1. VIDEO OVERVIEW 📋
               - Extract video metadata: title, duration, upload date
               - Identify video type: tutorial, review, lecture, documentary, etc.
               - Determine target audience and content difficulty level
               - Note the presenter's style and approach

            2. CONTENT EXTRACTION 🎬
               - Fetch and analyze full video transcript
               - Identify main themes and recurring topics
               - Note key demonstrations, examples, and practical content
               - Extract important references, resources, or links mentioned

            3. TIMESTAMP CREATION ⏱️
               - Create precise, meaningful timestamps for major topic transitions
               - Focus on content that provides educational or practical value
               - Highlight key moments: demonstrations, code examples, important explanations
               - Format: [start_time, end_time, detailed_summary]
               - Ensure timestamp accuracy and avoid hallucination

            4. STRUCTURED ORGANIZATION 🏗️
               - Group related segments into logical sections
               - Identify main themes and track topic progression
               - Create hierarchical structure: Chapters → Sections → Key Points
               - Include content type indicators with relevant emojis:
                 📚 Educational | 💻 Technical | 🎮 Gaming | 📱 Tech Review | 🎨 Creative
                 🧪 Science | 📈 Business | 🎭 Entertainment | 🏋️ Fitness | 🍳 Cooking

            5. QUALITY ASSURANCE ✅
               - Verify timestamp accuracy against transcript
               - Ensure comprehensive coverage of video content
               - Maintain consistent detail level throughout analysis
               - Focus on valuable content markers and learning points
               - Include practical takeaways and actionable insights

            6. OUTPUT FORMATTING ✨
               - Begin with comprehensive video overview
               - Use clear, descriptive segment titles
               - Include timestamps in HH:MM:SS format
               - Highlight key learning points with bullet points
               - Note practical demonstrations and important references
               - Mark content difficulty and prerequisites when relevant

            SPECIALIZED ANALYSIS GUIDELINES:
            - For tutorials: Focus on step-by-step processes and code examples
            - For lectures: Emphasize theoretical concepts and key arguments
            - For reviews: Highlight product features, pros/cons, comparisons
            - For documentaries: Track chronological events and key facts
            - For vlogs/podcasts: Identify main discussion points and insights

            ALWAYS:
            - Respect content creators and provide accurate representations
            - Note video length to help users plan their viewing
            - Include content warnings if video contains sensitive material
            - Acknowledge limitations when transcripts are incomplete
        """),
        add_datetime_to_context=True,
        markdown=True,
    )

    print(f"✅ YouTube analysis agent initialized using {model_name}")
    print("🎬 YouTube video analysis enabled")
    if mem0_api_key:
        print("🧠 Memory system enabled for conversation context")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent

    if not agent:
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing YouTube Analysis Agent...")
            await initialize_agent()
            _initialized = True

    return await run_agent(messages)


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up YouTube Analysis Agent resources...")


def _setup_environment_variables(args: argparse.Namespace) -> None:
    """Set environment variables from command line arguments."""
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model


def _display_configuration_info() -> None:
    """Display configuration information to the user."""
    print("=" * 60)
    print("🎬 YOUTUBE VIDEO ANALYSIS AGENT")
    print("=" * 60)
    print("📺 Purpose: Analyze YouTube videos and create structured summaries")
    print("🔧 Powered by: YouTube transcript extraction + AI analysis")

    config_info = []
    if os.getenv("OPENROUTER_API_KEY"):
        model = os.getenv("MODEL_NAME", "openai/gpt-4o")
        config_info.append(f"🤖 Model: {model}")
    config_info.append("🎬 YouTube: Video analysis and transcripts")
    if os.getenv("MEM0_API_KEY"):
        config_info.append("🧠 Memory: Conversation context")

    for info in config_info:
        print(info)

    print("=" * 60)
    print("Example queries:")
    print("• 'Analyze this video: https://www.youtube.com/watch?v=zjkBMFhNj_g'")
    print("• 'Create a study guide from this lecture video'")
    print("• 'Extract key points from this tutorial with timestamps'")
    print("• 'Summarize this documentary with chapter breakdowns'")
    print("• 'Analyze this product review and list all features mentioned'")
    print("=" * 60)


def main() -> None:
    """Run the main entry point for the YouTube Analysis Agent."""
    parser = argparse.ArgumentParser(
        description="YouTube Analysis Agent - Create structured summaries from YouTube videos"
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key for conversation memory (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )

    args = parser.parse_args()

    _setup_environment_variables(args)
    _display_configuration_info()

    config = load_config()

    try:
        print("\n🚀 Starting YouTube Analysis Agent server...")
        print(f"🌐 Access at: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 YouTube Analysis Agent stopped")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()