# TODO: Multi-Face Support

## Core Detection & Matching

- [ ] **Multiple faces per image**
  - Update `extract_embedding()` to return list of embeddings
  - Handle multiple faces in both profile photos and scanned photos
  - File: `src/services/face_service.py`

- [ ] **Match all faces**
  - Change `match_face()` to compare all detected faces against profiles
  - Return list of matches instead of single match
  - File: `src/services/face_service.py`

## Data Types

- [ ] **FaceMatchResult for multiple people**
  - Add `matched_faces: List[FaceMatch]` field
  - Include bounding box coordinates for each face
  - File: `src/types/face.py`

- [ ] **ProcessedFile multi-person metadata**
  - Change to `matched_persons: List[str]`
  - Change to `similarities: List[float]`
  - Update `to_dict()` serialization
  - File: `src/types/scan.py`

## Configuration

- [ ] **Add multi-face settings**
  - `multiFactDetection: bool` - enable/disable
  - `primaryMatchStrategy: 'strongest' | 'first' | 'all'` - how to pick primary person
  - `minFacesRequired: int` - minimum faces to detect
  - File: `src/types/config.py`

- [ ] **Multi-person output folder**
  - Add `multiPerson: "people/multi-person"` to outputStructure
  - Update copy logic to use this folder
  - Files: `src/types/config.py`, `src/services/file_service.py`

## Scan Logic

- [ ] **Process multiple matches**
  - Update `_process_single_image()` to handle list of matches
  - Implement primary match strategy (strongest/first/all)
  - Decide destination folder based on strategy
  - File: `src/services/scan_service.py`

- [ ] **Update embedding cache**
  - Store list of embeddings per profile photo
  - Update save/load serialization
  - File: `src/services/embedding_cache.py`

## User Interface

- [ ] **CLI output for multiple matches**
  - Show multi-person photo count in summary table
  - Verbose mode: list all detected faces with similarities
  - File: `cli.py`

## Testing & Docs

- [ ] **Test cases**
  - Group photo with multiple known people
  - Photo with known + unknown people
  - Profile photo with multiple faces (should warn)
  - Location: `tests/`

- [ ] **Update README**
  - Explain multi-face detection behavior
  - Document configuration options
  - Show example of group photo organization
  - File: `README.md`
