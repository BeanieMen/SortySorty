import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..types.learn import LearnDatabase, LearnEntry
from ..helpers.fs import copy_file, ensure_directory
from ..helpers.string import slugify


def load_learn_database(profiles_dir: Path) -> LearnDatabase:
    db_path = profiles_dir / "learn-database.json"
    
    if not db_path.exists():
        return LearnDatabase()
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    return LearnDatabase.from_dict(data)


def save_learn_database(database: LearnDatabase, profiles_dir: Path) -> None:
    db_path = profiles_dir / "learn-database.json"
    ensure_directory(profiles_dir)
    
    with open(db_path, 'w') as f:
        json.dump(database.to_dict(), f, indent=2)


def learn_from_report(
    report_path: Path,
    profiles_dir: Path,
    min_similarity: float = 0.6,
    dry_run: bool = False
) -> int:
    # Load report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Load learn database
    database = load_learn_database(profiles_dir)
    
    # Track existing learned photos to avoid duplicates
    learned_sources = {str(entry.source_photo) for entry in database.entries}
    
    learned_count = 0
    
    # Process each file in report
    for file_data in report.get('processedFiles', []):
        # Skip if not copied or no match
        if file_data.get('action') != 'copied':
            continue
        
        matched_person = file_data.get('matchedPerson')
        similarity = file_data.get('similarity')
        source = file_data.get('source')
        
        # Skip if doesn't meet criteria
        if not matched_person or matched_person == 'unknown':
            continue
        
        if similarity is None or similarity < min_similarity:
            continue
        
        # Skip if already learned
        if source in learned_sources:
            continue
        
        # Determine profile directory
        profile_dir = profiles_dir / matched_person
        
        if not dry_run:
            ensure_directory(profile_dir)
            
            # Copy photo to profile directory
            source_path = Path(source)
            dest_path = profile_dir / source_path.name
            
            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                dest_path = profile_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file
            if copy_file(source_path, dest_path):
                # Add to database
                entry = LearnEntry(
                    source_photo=source_path,
                    profile_name=matched_person,
                    similarity=similarity,
                    learned_at=datetime.now(),
                    profile_photo_path=dest_path
                )
                database.add_entry(entry)
                learned_count += 1
        else:
            # Dry run - just count
            print(f"Would learn: {source} -> {matched_person} (similarity: {similarity:.3f})")
            learned_count += 1
    
    # Save database
    if not dry_run and learned_count > 0:
        save_learn_database(database, profiles_dir)
    
    return learned_count
