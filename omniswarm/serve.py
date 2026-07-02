"""Console entry point for running the OmniSwarm HTTP server.

`omniswarm-serve` (after `pip install .`) starts the dashboard + API on 0.0.0.0:8100.
Override with OMNISWARM_HOST / OMNISWARM_PORT.
"""
import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "omniswarm.app:app",
        host=os.environ.get("OMNISWARM_HOST", "0.0.0.0"),
        port=int(os.environ.get("OMNISWARM_PORT", "8100")),
    )


if __name__ == "__main__":
    main()
