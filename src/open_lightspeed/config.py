"""Centralized configuration loaded from environment variables.

All scripts import from here instead of reading os.environ directly.
Values can be set via .env file or shell environment.
"""

import os


# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Models
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "openai/gpt-5.2")

# Service URLs
LIGHTSPEED_URL = os.environ.get("LIGHTSPEED_URL", "http://localhost:8080/v1/query")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8085/mcp")

# Scenario Generation
NUM_VARIANTS = int(os.environ.get("NUM_VARIANTS", "4"))

# Distillation
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))
TOOL_K = int(os.environ.get("TOOL_K", "3"))
