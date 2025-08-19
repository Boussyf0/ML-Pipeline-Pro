#!/usr/bin/env python3
"""Script to start the FastAPI server with proper configuration."""
import asyncio
import logging
import sys
import os
from pathlib import Path
import click
import uvicorn
from multiprocessing import cpu_count

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Calculate optimal number of workers."""
    workers = min(cpu_count(), 8)  # Cap at 8 workers
    logger.info(f"Using {workers} workers (detected {cpu_count()} CPU cores)")
    return workers


@click.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=0, help='Number of worker processes (0 = auto)')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', default='info', help='Log level')
@click.option('--config-path', default='config/config.yaml', help='Path to configuration file')
def main(host: str, port: int, workers: int, reload: bool, log_level: str, config_path: str):
    """Start the FastAPI server."""
    try:
        # Validate configuration file exists
        if not Path(config_path).exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Server will start with default configuration")
        
        # Set environment variables
        os.environ['CONFIG_PATH'] = config_path
        
        # Determine worker count
        if workers == 0:
            workers = 1 if reload else get_optimal_workers()
        
        logger.info(f"Starting FastAPI server on {host}:{port}")
        logger.info(f"Workers: {workers}, Reload: {reload}, Log Level: {log_level}")
        
        # Configure uvicorn
        config = uvicorn.Config(
            "src.api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,  # Workers not compatible with reload
            reload=reload,
            log_level=log_level,
            access_log=True,
            server_header=False,  # Don't expose server info
            date_header=False,    # Don't expose date header
        )
        
        # Start server
        server = uvicorn.Server(config)
        
        if reload:
            logger.info("🔄 Development mode: auto-reload enabled")
        else:
            logger.info("🚀 Production mode: multi-worker setup")
            
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


@click.command()
@click.option('--host', default='127.0.0.1', help='Host to test')
@click.option('--port', default=8000, help='Port to test')
@click.option('--timeout', default=10, help='Timeout in seconds')
def healthcheck(host: str, port: int, timeout: int):
    """Check if the API server is healthy."""
    import requests
    
    try:
        response = requests.get(
            f"http://{host}:{port}/health",
            timeout=timeout
        )
        response.raise_for_status()
        
        health_data = response.json()
        status = health_data.get("status", "unknown")
        
        if status == "healthy":
            logger.info("✅ API server is healthy")
            print(f"Status: {status}")
            print(f"Details: {health_data.get('details', {})}")
            sys.exit(0)
        else:
            logger.error(f"❌ API server is unhealthy: {status}")
            print(f"Status: {status}")
            print(f"Details: {health_data.get('details', {})}")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Cannot connect to API server at {host}:{port}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.error(f"❌ Health check timed out after {timeout} seconds")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        sys.exit(1)


@click.group()
def cli():
    """MLOps API server management."""
    pass


cli.add_command(main, name="start")
cli.add_command(healthcheck, name="health")


if __name__ == "__main__":
    cli()