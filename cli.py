#!/usr/bin/env python3

import json
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from src.config import ConfigManager
from src.types.config import Config
from src.services.scan_service import ScanService
from src.commands.learn import learn_from_report
from src.helpers.fs import ensure_directory, list_files
from src.helpers.string import slugify

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Sorty Sorty - Organize photos using face recognition and OCR."""
    pass


@cli.command()
@click.option('--dir', 'directory', type=click.Path(), default=".", help="Directory to initialize")
def init(directory: str):
    """Initialize a new sorty sorty project."""
    base_dir = Path(directory).resolve()
    
    console.print(f"[bold green]Initializing sorty sorty project in {base_dir}[/bold green]")
    
    # Create directory structure
    profiles_dir = base_dir / "profiles"
    photos_dir = base_dir / "photos"
    output_dir = base_dir / "output"
    
    ensure_directory(profiles_dir)
    ensure_directory(photos_dir)
    ensure_directory(output_dir)
    
    # Create .gitkeep files
    (profiles_dir / ".gitkeep").touch()
    (photos_dir / ".gitkeep").touch()
    
    # Create default config
    config = ConfigManager.create_default(base_dir)
    config_path = base_dir / "sorter-sorter.config.json"
    ConfigManager.save(config, config_path)
    
    console.print(f"✓ Created directories:")
    console.print(f"  - {profiles_dir} (for profile photos)")
    console.print(f"  - {photos_dir} (for input photos)")
    console.print(f"  - {output_dir} (for sorted output)")
    console.print(f"✓ Created config file: {config_path}")
    console.print("\n[bold yellow]Next steps:[/bold yellow]")
    console.print("1. Add profile photos to profiles/person-name/")
    console.print("2. Add photos to sort to photos/")
    console.print("3. Run: sorty-sorty scan --input photos --output output")


@cli.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True), help="Input directory containing photos")
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(), help="Output directory for sorted photos")
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), help="Path to config file")
@click.option('--threshold', '-t', type=float, help="Face matching threshold (0.0-1.0)")
@click.option('--profiles', '-p', type=click.Path(exists=True), help="Profiles directory")
@click.option('--verbose', '-v', is_flag=True, help="Show detailed timing information")
@click.option('--parallel', is_flag=True, help="Enable parallel processing with multiple workers")
@click.option('--concurrency', type=int, help="Number of parallel workers (default: 3)")
@click.option('--max-dimension', type=int, help="Maximum image dimension in pixels (default: 1600)")
@click.option('--delete-duplicates', is_flag=True, help="Delete duplicate photos instead of skipping")
@click.option('--rename-timestamp', is_flag=True, help="Rename photos with timestamp")
@click.option('--no-clustering', is_flag=True, help="Disable clustering of unknown faces")
def scan(input_dir: str, output_dir: str, config_path: str | None = None, threshold: float | None = None, 
         profiles: str | None = None, verbose: bool = False, parallel: bool = False, 
         concurrency: int | None = None, max_dimension: int | None = None, 
         delete_duplicates: bool = False, rename_timestamp: bool = False, no_clustering: bool = False):
    """Scan and organize photos from input directory."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Load or create config
    if config_path:
        config = ConfigManager.load(Path(config_path))
    else:
        # Try to find config in input directory or current directory
        search_paths = [
            input_path / "sorty-sorty.config.json",
            Path.cwd() / "sorty-sorty.config.json"
        ]
        
        config = None
        for path in search_paths:
            if path.exists():
                config = ConfigManager.load(path)
                break
        
        if config is None:
            config = Config()
            console.print("[yellow]No config file found, using defaults[/yellow]")
    
    # Override with CLI options
    if threshold is not None:
        config.threshold = threshold
    
    if profiles:
        config.profiles_dir = Path(profiles).resolve()
    
    if verbose:
        config.verbose = True
    
    if concurrency is not None:
        config.concurrency = concurrency
    elif not parallel:
        config.concurrency = 1
    
    if max_dimension is not None:
        config.max_dimension = max_dimension
    
    if delete_duplicates:
        config.delete_duplicates = True
    
    if rename_timestamp:
        config.rename_with_timestamp = True
    
    if no_clustering:
        config.store_unknown_clusters = False
    
    # Ensure profiles dir is absolute
    if not config.profiles_dir.is_absolute():
        config.profiles_dir = input_path.parent / config.profiles_dir
    
    console.print(f"[bold]Scanning photos...[/bold]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    console.print(f"Profiles: {config.profiles_dir}")
    console.print(f"Threshold: {config.threshold}\n")
    
    # Run scan
    scan_service = ScanService(config)
    report = scan_service.scan(input_path, output_path)
    
    # Save report
    report_path = output_path / "report.json"
    ensure_directory(output_path)
    
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    # Display summary
    console.print("\n[bold green]Scan Complete![/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Total Processed", str(report.processed))
    table.add_row("Copied", str(report.copied))
    table.add_row("Duplicates Skipped", str(report.duplicates))
    table.add_row("Unknown Faces", str(report.unknown_faces))
    table.add_row("Ambiguous Matches", str(report.ambiguous))
    table.add_row("Errors", str(report.errors))
    
    if report.clusters:
        table.add_row("Face Clusters Found", str(len(report.clusters)))
    
    console.print(table)
    console.print(f"\n✓ Report saved to: {report_path}")
    
    if report.review_entries:
        console.print(f"\n[yellow]⚠ {len(report.review_entries)} photo(s) need review[/yellow]")
    
    # Check if there are learnable photos and ask user
    learnable_count = _count_learnable_photos(report_path, config.threshold)
    
    if learnable_count > 0:
        console.print(f"\n[bold cyan]📚 {learnable_count} photo(s) can be learned (similarity >= {config.threshold:.0%})[/bold cyan]")
        
        # Flush stdout to ensure prompt appears immediately
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        if click.confirm("Would you like to add these photos to profiles for future learning?", default=True):
            console.print()
            learn_count = learn_from_report(
                report_path,
                config.profiles_dir,
                min_similarity=config.threshold,
                dry_run=False
            )
            
            if learn_count > 0:
                console.print(f"\n[bold green]✓ Learned {learn_count} photo(s)![/bold green]")
                console.print("These photos have been added to profile directories.")
            else:
                console.print("\n[yellow]No photos were learned.[/yellow]")
    else:
        console.print(f"\n[dim]No photos meet the learning threshold (>= {config.threshold:.0%})[/dim]")


def _count_learnable_photos(report_path: Path, min_similarity: float) -> int:
    """Count how many photos can be learned from a report."""
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        count = 0
        for file_data in report.get('processedFiles', []):
            if file_data.get('action') == 'copied':
                matched_person = file_data.get('matchedPerson')
                similarity = file_data.get('similarity')
                
                if matched_person and matched_person != 'unknown' and similarity and similarity >= min_similarity:
                    count += 1
        
        return count
    except Exception:
        return 0


@cli.command()
@click.option('--report', '-r', 'report_path', required=True, type=click.Path(exists=True), help="Path to scan report JSON")
@click.option('--min-similarity', '-s', type=float, default=0.6, help="Minimum similarity threshold for learning")
@click.option('--profiles', '-p', type=click.Path(exists=True), help="Profiles directory")
@click.option('--dry-run', is_flag=True, help="Show what would be learned without copying files")
def learn(report_path: str, min_similarity: float, profiles: str | None = None, dry_run: bool = False):
    """Learn from high-confidence matches in a scan report."""
    report_file = Path(report_path).resolve()
    
    # Determine profiles directory
    if profiles:
        profiles_dir = Path(profiles).resolve()
    else:
        # Try to find profiles directory
        profiles_dir = report_file.parent.parent / "profiles"
        
        if not profiles_dir.exists():
            console.print("[red]Error: Could not find profiles directory[/red]")
            console.print("Please specify with --profiles option")
            sys.exit(1)
    
    console.print(f"[bold]Learning from report...[/bold]")
    console.print(f"Report: {report_file}")
    console.print(f"Profiles: {profiles_dir}")
    console.print(f"Min Similarity: {min_similarity}")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be copied[/yellow]\n")
    
    # Learn from report
    learned_count = learn_from_report(
        report_file,
        profiles_dir,
        min_similarity=min_similarity,
        dry_run=dry_run
    )
    
    if learned_count > 0:
        if dry_run:
            console.print(f"\n[green]Would learn {learned_count} photo(s)[/green]")
        else:
            console.print(f"\n[bold green]✓ Learned {learned_count} photo(s)![/bold green]")
            console.print("These photos have been added to profile directories for future scans.")
    else:
        console.print("\n[yellow]No photos met the learning criteria[/yellow]")


@cli.command()
@click.option('--name', '-n', required=True, help="Person's name")
@click.option('--images', '-i', multiple=True, required=True, type=click.Path(exists=True), help="Image files to add (can specify multiple)")
@click.option('--profiles', '-p', type=click.Path(exists=True), help="Profiles directory")
def add_person(name: str, images: tuple, profiles: str | None = None):
    """Add a new person profile with reference images."""
    # Determine profiles directory
    if profiles:
        profiles_dir = Path(profiles).resolve()
    else:
        profiles_dir = Path.cwd() / "profiles"
    
    # Create profile directory
    profile_name = slugify(name)
    profile_dir = profiles_dir / profile_name
    ensure_directory(profile_dir)
    
    # Copy images
    from src.helpers.fs import copy_file
    
    copied = 0
    for img_path in images:
        src = Path(img_path).resolve()
        dst = profile_dir / src.name
        
        if copy_file(src, dst):
            copied += 1
            console.print(f"✓ Added: {src.name}")
        else:
            console.print(f"[yellow]⚠ Skipped: {src.name} (already exists)[/yellow]")
    
    console.print(f"\n[bold green]✓ Profile '{profile_name}' created with {copied} image(s)[/bold green]")
    console.print(f"Location: {profile_dir}")


@cli.command()
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(exists=True), help="Output directory with report")
def stats(output_dir: str):
    """Show statistics from a scan report."""
    output_path = Path(output_dir).resolve()
    report_path = output_path / "report.json"
    
    if not report_path.exists():
        console.print(f"[red]Error: No report.json found in {output_path}[/red]")
        sys.exit(1)
    
    # Load report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Display statistics
    console.print(f"\n[bold]Scan Report Statistics[/bold]")
    console.print(f"Timestamp: {report.get('timestamp', 'Unknown')}\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Count", justify="right", style="green", width=10)
    
    table.add_row("Total Processed", str(report.get('processed', 0)))
    table.add_row("Successfully Copied", str(report.get('copied', 0)))
    table.add_row("Duplicates Skipped", str(report.get('duplicates', 0)))
    table.add_row("Unknown Faces", str(report.get('unknownFaces', 0)))
    table.add_row("Ambiguous Matches", str(report.get('ambiguous', 0)))
    table.add_row("Errors", str(report.get('errors', 0)))
    table.add_row("Items for Review", str(len(report.get('reviewEntries', []))))
    table.add_row("Face Clusters", str(len(report.get('clusters', []))))
    
    console.print(table)
    
    # Show person breakdown
    person_counts = {}
    for file_data in report.get('processedFiles', []):
        person = file_data.get('matchedPerson')
        if person:
            person_counts[person] = person_counts.get(person, 0) + 1
    
    if person_counts:
        console.print("\n[bold]Photos per Person:[/bold]")
        person_table = Table(show_header=True, header_style="bold magenta")
        person_table.add_column("Person", style="cyan")
        person_table.add_column("Photos", justify="right", style="green")
        
        for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
            person_table.add_row(person, str(count))
        
        console.print(person_table)


if __name__ == "__main__":
    cli()
