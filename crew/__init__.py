"""Multi-agent market research crew."""
import os
import warnings

# CrewAI shows an interactive "view execution traces?" prompt that blocks for
# several seconds and would pollute the benchmark timings. Opt back in with
# CREWAI_TRACING_ENABLED=true in your .env if you want CrewAI's own tracing.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

# Harmless CrewAI/pydantic notice about closures used as task callbacks.
warnings.filterwarnings("ignore", message="A function callback cannot be saved")
