"""
Main CLI entry point for ORB trading system.

Combines all command modules into a single CLI interface.
"""

import typer
from typing import Optional
from pathlib import Path

from ..utils.logging import setup_logging

# Create main CLI app
app = typer.Typer(
    name="orb",
    help="ORB Intraday Meta-Model Trading System",
    add_completion=False
)

# Import training commands directly
from .train import (
    prepare_data,
    tune_hyperparameters, 
    run_walkforward,
    train_full_pipeline
)

# Add training commands to main app
app.command("prepare-data")(prepare_data)
app.command("tune")(tune_hyperparameters)
app.command("walkforward")(run_walkforward)
app.command("train")(train_full_pipeline)


@app.command()
def version():
    """Show version information."""
    try:
        from .. import __version__
        typer.echo(f"ORB Trading System v{__version__}")
    except ImportError:
        typer.echo("ORB Trading System (version unknown)")


@app.command()
def info():
    """Show system information and configuration."""
    typer.echo("ORB Intraday Meta-Model Trading System")
    typer.echo("=" * 40)
    typer.echo("Components:")
    typer.echo("  - Data ingestion (Polygon API)")
    typer.echo("  - Feature engineering (ORB features)")
    typer.echo("  - Model training (LightGBM + Optuna)")
    typer.echo("  - MLflow experiment tracking")
    typer.echo("  - Walk-forward validation")
    typer.echo("  - Daily signal generation")
    typer.echo()
    
    # Check directory structure
    cwd = Path.cwd()
    typer.echo(f"Working directory: {cwd}")
    
    # Check for config files
    config_dir = cwd / "config"
    if config_dir.exists():
        typer.echo(f"✓ Config directory found: {config_dir}")
        for config_file in ["core.yaml", "train.yaml", "assets.yaml"]:
            if (config_dir / config_file).exists():
                typer.echo(f"  ✓ {config_file}")
            else:
                typer.echo(f"  ✗ {config_file} (missing)")
    else:
        typer.echo(f"✗ Config directory not found: {config_dir}")
    
    # Check data directories
    data_dirs = ["data/raw", "data/minute", "data/feat", "models", "mlruns"]
    for dir_name in data_dirs:
        dir_path = cwd / dir_name
        if dir_path.exists():
            typer.echo(f"✓ {dir_name}")
        else:
            typer.echo(f"✗ {dir_name} (missing)")


@app.command()
def setup(
    create_dirs: bool = typer.Option(True, "--create-dirs/--no-create-dirs", 
                                    help="Create missing directories"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Set up the ORB system directory structure."""
    if verbose:
        setup_logging(log_level="DEBUG")
    
    cwd = Path.cwd()
    typer.echo(f"Setting up ORB system in: {cwd}")
    
    # Create directories if requested
    if create_dirs:
        directories = [
            "data/raw",
            "data/minute", 
            "data/feat",
            "models",
            "mlruns",
            "blotters",
            "scripts",
            "notebooks",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = cwd / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                typer.echo(f"✓ Created {dir_name}")
            else:
                typer.echo(f"  {dir_name} already exists")
    
    # Check configuration
    config_dir = cwd / "config"
    if not config_dir.exists():
        typer.echo(f"⚠ Config directory not found: {config_dir}")
        typer.echo("Please ensure config files are present before running commands.")
    else:
        typer.echo(f"✓ Config directory found: {config_dir}")
    
    typer.echo()
    typer.echo("Setup complete! Next steps:")
    typer.echo("1. Set environment variables: POLYGON_API_KEY, BARCHART_API_KEY")
    typer.echo("2. Download data: orb data download --symbol AAPL")
    typer.echo("3. Train model: orb train --symbols AAPL MSFT")
    typer.echo("4. Generate signals: orb predict --date 2024-01-15")


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log to file")
):
    """
    ORB Intraday Meta-Model Trading System
    
    An automated system for Opening-Range-Breakout trading strategy with
    ML-based signal generation, feature engineering, and walk-forward validation.
    """
    # Set up logging
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    
    setup_logging(log_level=log_level, log_file=log_file)


if __name__ == "__main__":
    app() 