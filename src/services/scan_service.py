from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import numpy.typing as npt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..types.config import Config
from ..types.scan import ScanReport, ProcessedFile, ReviewEntry, FaceCluster
from ..services.face_service import FaceService
from ..services.ocr_service import OcrService
from ..services.file_service import FileService
from ..services.cluster_service import ClusterService
from ..helpers.fs import ensure_directory

# Create console for colored output
console = Console()


# Worker function for multiprocessing (must be at module level to be pickled)
def _process_image_worker(args: Tuple[Path, Dict[str, Any]]) -> Dict[str, Any]:
    image_path, config_dict = args
    
    # Recreate services in worker process
    from ..types.config import Config
    from ..services.face_service import FaceService
    
    config = Config(**config_dict)
    face_service = FaceService(config.threshold, config.low_confidence_threshold)
    
    # Load profiles from cache only (already computed in main process)
    if not hasattr(_process_image_worker, '_profiles_loaded'):
        if not face_service.load_from_cache_only(config.profiles_dir):
            # Cache should exist since main process creates it
            raise RuntimeError("Profile cache not found - main process should have created it")
        _process_image_worker._profiles_loaded = True
        _process_image_worker._face_service = face_service
    else:
        face_service = _process_image_worker._face_service
    
    # Extract face embedding
    verbose = config_dict.get('verbose', False)
    embedding = face_service.extract_embedding(image_path, verbose=verbose)
    
    result = {
        'image_path': str(image_path),
        'embedding': embedding.tolist() if embedding is not None else None,
        'has_face': embedding is not None
    }
    
    if embedding is not None:
        match_result = face_service.match_face(embedding)
        result['match_result'] = {
            'has_match': match_result.has_match,
            'is_ambiguous': match_result.is_ambiguous,
            'best_match': {
                'name': match_result.best_match.name,
                'similarity': match_result.best_match.similarity
            } if match_result.best_match else None,
            'alternatives': [
                {'name': alt.name, 'similarity': alt.similarity}
                for alt in match_result.alternatives[:3]
            ]
        }
    
    return result


class ScanService:
    
    def __init__(self, config: Config):
        self.config = config
        self.face_service = FaceService(
            threshold=config.threshold,
            low_confidence_threshold=config.low_confidence_threshold
        )
        self.ocr_service = OcrService(config.ocr)
        self.file_service = FileService()
        self.cluster_service = ClusterService()
        
        self.unknown_embeddings: List[npt.NDArray[np.float64]] = []
    
    def scan(self, input_dir: Path, output_dir: Path) -> ScanReport:
        import time
        
        # Initialize report
        report = ScanReport()
        
        # Reset file service for new scan
        self.file_service.reset()
        self.unknown_embeddings = []
        
        # Load face profiles FIRST (before deciding on parallel vs sequential)
        # This ensures cache is computed once in the main process
        t_start = time.perf_counter()
        console.print(f"Loading face profiles from [cyan]{self.config.profiles_dir}[/cyan]...")
        self.face_service.load_profiles(self.config.profiles_dir, verbose=self.config.verbose)
        t_profiles = time.perf_counter() - t_start
        profile_count = len(self.face_service.profile_embeddings)
        
        if self.config.verbose:
            console.print(f"Loaded [bold green]{profile_count}[/bold green] profile(s) in [yellow]{t_profiles:.2f}s[/yellow]\n")
        else:
            console.print(f"Loaded [bold green]{profile_count}[/bold green] profile(s)\n")
        
        # Get all image files
        image_files = self.file_service.get_image_files(input_dir)
        console.print(f"Found [bold cyan]{len(image_files)}[/bold cyan] image(s) to process")
        
        if not image_files:
            return report
        
        # Use multiprocessing for CPU-bound face recognition
        use_multiprocessing = self.config.concurrency > 1 and len(image_files) > 1
        
        if use_multiprocessing:
            console.print(f"Using [bold magenta]{self.config.concurrency}[/bold magenta] worker processes for parallel processing...")
            self._scan_parallel(image_files, output_dir, report)
        else:
            console.print("[yellow]Processing sequentially...[/yellow]")
            self._scan_sequential(image_files, output_dir, report)
        
        # Cluster unknown faces if enabled
        if self.config.store_unknown_clusters and self.unknown_embeddings:
            console.print(f"\n[cyan]Clustering {len(self.unknown_embeddings)} unknown face(s)...[/cyan]")
            clusters = self.cluster_service.cluster_faces(self.unknown_embeddings)
            report.clusters = clusters
            console.print(f"Found [bold green]{len(clusters)}[/bold green] cluster(s)")
        
        return report
    
    def _scan_sequential(self, image_files: List[Path], output_dir: Path, report: ScanReport) -> None:
        import time
        
        # Profiles already loaded in main scan() method
        # Process images with progress bar
        t_process_start = time.perf_counter()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing photos...", total=len(image_files))
            
            for image_path in image_files:
                try:
                    processed = self._process_image(image_path, output_dir, report)
                    report.processed_files.append(processed)
                    report.processed += 1
                    
                    if processed.action == "copied":
                        report.copied += 1
                    elif processed.action == "skipped-duplicate":
                        report.duplicates += 1
                    elif processed.action == "error":
                        report.errors += 1
                    
                except Exception as e:
                    console.print(f"\n[red]Error processing {image_path}: {e}[/red]")
                    report.processed_files.append(ProcessedFile(
                        source=image_path,
                        destination=None,
                        action="error",
                        error=str(e)
                    ))
                    report.errors += 1
                
                progress.update(task, advance=1)
        
        if self.config.verbose:
            t_process_total = time.perf_counter() - t_process_start
            console.print(f"\n[bold green]Total processing time:[/bold green] [yellow]{t_process_total:.2f}s[/yellow] ([dim]{t_process_total/len(image_files):.2f}s per image[/dim])")
    
    def _scan_parallel(self, image_files: List[Path], output_dir: Path, report: ScanReport) -> None:
        import time
        
        # Convert config to dict for pickling
        config_dict = {
            'threshold': self.config.threshold,
            'low_confidence_threshold': self.config.low_confidence_threshold,
            'profiles_dir': self.config.profiles_dir,
            'concurrency': self.config.concurrency,
            'rename_with_timestamp': self.config.rename_with_timestamp,
            'output_structure': self.config.output_structure,
            'ocr': self.config.ocr,
            'store_unknown_clusters': self.config.store_unknown_clusters,
            'verbose': self.config.verbose,
        }
        
        # Prepare arguments for workers
        worker_args = [(img_path, config_dict) for img_path in image_files]
        
        # Process in parallel
        results_map: Dict[str, Dict[str, Any]] = {}
        
        t_process_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.config.concurrency) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing photos...", total=len(image_files))
                
                # Submit all tasks
                future_to_path = {
                    executor.submit(_process_image_worker, args): args[0]
                    for args in worker_args
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    image_path = future_to_path[future]
                    try:
                        result = future.result()
                        results_map[result['image_path']] = result
                    except Exception as e:
                        console.print(f"\n[red]Error processing {image_path}: {e}[/red]")
                        results_map[str(image_path)] = {
                            'image_path': str(image_path),
                            'error': str(e),
                            'has_face': False
                        }
                    
                    progress.update(task, advance=1)
        
        t_process_total = time.perf_counter() - t_process_start
        
        if self.config.verbose:
            console.print(f"\n[bold green]Parallel processing time:[/bold green] [yellow]{t_process_total:.2f}s[/yellow] ([dim]{t_process_total/len(image_files):.2f}s per image avg[/dim])")
        
        # Now process results in original order and perform file operations
        # (File operations must be done sequentially to avoid conflicts)
        for image_path in image_files:
            result = results_map.get(str(image_path))
            if not result:
                continue
            
            try:
                # Check for duplicates
                if self.file_service.is_duplicate(image_path):
                    processed = ProcessedFile(
                        source=image_path,
                        destination=None,
                        action="skipped-duplicate"
                    )
                    report.processed_files.append(processed)
                    report.processed += 1
                    report.duplicates += 1
                    continue
                
                # Handle errors from worker
                if 'error' in result and not result.get('has_face'):
                    processed = ProcessedFile(
                        source=image_path,
                        destination=None,
                        action="error",
                        error=result.get('error')
                    )
                    report.processed_files.append(processed)
                    report.processed += 1
                    report.errors += 1
                    continue
                
                # Process face recognition result
                if result.get('has_face') and result.get('embedding'):
                    embedding_list = result['embedding']
                    embedding = np.array(embedding_list, dtype=np.float64)
                    match_info = result.get('match_result', {})
                    
                    if match_info.get('has_match') and match_info.get('best_match'):
                        # Matched a known person
                        person_name = match_info['best_match']['name']
                        similarity = match_info['best_match']['similarity']
                        
                        dest_dir = output_dir / self.config.output_structure.people / person_name
                        destination = self.file_service.copy_photo(
                            image_path,
                            dest_dir,
                            rename_with_timestamp=self.config.rename_with_timestamp
                        )
                        
                        # Check if ambiguous
                        if match_info.get('is_ambiguous'):
                            report.ambiguous += 1
                            report.review_entries.append(ReviewEntry(
                                image_path=image_path,
                                reason="ambiguous-face",
                                candidates=match_info.get('alternatives', [])[:3]
                            ))
                        
                        processed = ProcessedFile(
                            source=image_path,
                            destination=destination,
                            action="copied",
                            matched_person=person_name,
                            similarity=similarity
                        )
                        report.processed_files.append(processed)
                        report.processed += 1
                        report.copied += 1
                    else:
                        # Unknown face
                        report.unknown_faces += 1
                        self.unknown_embeddings.append(embedding)
                        
                        dest_dir = output_dir / self.config.output_structure.unknown_people
                        destination = self.file_service.copy_photo(
                            image_path,
                            dest_dir,
                            rename_with_timestamp=self.config.rename_with_timestamp
                        )
                        
                        # Add to review if we have candidates
                        alternatives = match_info.get('alternatives', [])
                        if alternatives:
                            report.review_entries.append(ReviewEntry(
                                image_path=image_path,
                                reason="low-confidence",
                                candidates=alternatives
                            ))
                        
                        processed = ProcessedFile(
                            source=image_path,
                            destination=destination,
                            action="copied",
                            matched_person="unknown"
                        )
                        report.processed_files.append(processed)
                        report.processed += 1
                        report.copied += 1
                else:
                    # No face detected - handle as generic photo
                    dest_dir = output_dir / self.config.output_structure.others
                    destination = self.file_service.copy_photo(
                        image_path,
                        dest_dir,
                        rename_with_timestamp=self.config.rename_with_timestamp
                    )
                    
                    processed = ProcessedFile(
                        source=image_path,
                        destination=destination,
                        action="copied"
                    )
                    report.processed_files.append(processed)
                    report.processed += 1
                    report.copied += 1
                    
            except Exception as e:
                console.print(f"\n[red]Error finalizing {image_path}: {e}[/red]")
                report.processed_files.append(ProcessedFile(
                    source=image_path,
                    destination=None,
                    action="error",
                    error=str(e)
                ))
                report.errors += 1
                report.processed += 1
    
    def _process_image(self, image_path: Path, output_dir: Path, report: ScanReport) -> ProcessedFile:
        # Check for duplicates
        if self.file_service.is_duplicate(image_path):
            return ProcessedFile(
                source=image_path,
                destination=None,
                action="skipped-duplicate"
            )
        
        # Try face recognition first
        embedding = self.face_service.extract_embedding(image_path, verbose=self.config.verbose)
        
        if embedding is not None:
            match_result = self.face_service.match_face(embedding)
            
            if match_result.has_match:
                # Matched a known person
                person_name = match_result.best_match.name
                dest_dir = output_dir / self.config.output_structure.people / person_name
                
                destination = self.file_service.copy_photo(
                    image_path,
                    dest_dir,
                    rename_with_timestamp=self.config.rename_with_timestamp
                )
                
                # Check if ambiguous
                if match_result.is_ambiguous:
                    report.ambiguous += 1
                    report.review_entries.append(ReviewEntry(
                        image_path=image_path,
                        reason="ambiguous-face",
                        candidates=[
                            {"name": c.name, "similarity": c.similarity}
                            for c in match_result.alternatives[:3]
                        ]
                    ))
                
                return ProcessedFile(
                    source=image_path,
                    destination=destination,
                    action="copied",
                    matched_person=person_name,
                    similarity=match_result.best_match.similarity
                )
            else:
                # Unknown face
                report.unknown_faces += 1
                self.unknown_embeddings.append(embedding)
                
                dest_dir = output_dir / self.config.output_structure.unknown_people
                destination = self.file_service.copy_photo(
                    image_path,
                    dest_dir,
                    rename_with_timestamp=self.config.rename_with_timestamp
                )
                
                # Add to review if we have candidates
                if match_result.alternatives:
                    report.review_entries.append(ReviewEntry(
                        image_path=image_path,
                        reason="low-confidence",
                        candidates=[
                            {"name": c.name, "similarity": c.similarity}
                            for c in match_result.alternatives
                        ]
                    ))
                
                return ProcessedFile(
                    source=image_path,
                    destination=destination,
                    action="copied",
                    matched_person="unknown"
                )
        
        # No face detected - try OCR
        if not self.config.ocr.enable_on_faces:
            ocr_result = self.ocr_service.extract_text(image_path)
            
            if ocr_result.is_chat:
                # Chat screenshot
                dest_dir = output_dir / self.config.output_structure.chats
            elif ocr_result.is_screenshot:
                # Regular screenshot
                dest_dir = output_dir / self.config.output_structure.screenshots
            else:
                # Other
                dest_dir = output_dir / self.config.output_structure.others
        else:
            # Skip OCR, categorize as other
            dest_dir = output_dir / self.config.output_structure.others
        
        destination = self.file_service.copy_photo(
            image_path,
            dest_dir,
            rename_with_timestamp=self.config.rename_with_timestamp
        )
        
        return ProcessedFile(
            source=image_path,
            destination=destination,
            action="copied"
        )
