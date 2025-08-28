#!/usr/bin/env python3
"""
Artifact Versioning System for BEM Pipeline

Provides comprehensive versioning and provenance tracking for all pipeline
artifacts including datasets, models, results, and papers. Ensures full
reproducibility through cryptographic hashing and dependency tracking.

Classes:
    ArtifactHash: Manages cryptographic hashing of artifacts
    ProvenanceTracker: Tracks provenance and dependency chains  
    VersionRegistry: Central registry for all versioned artifacts
    ArtifactVersioner: Main interface for artifact versioning
    ReproducibilityValidator: Validates reproducibility of results

Usage:
    versioner = ArtifactVersioner(registry_dir="artifacts")
    
    # Version a dataset
    dataset_hash = versioner.version_artifact(
        artifact_path="data/train.json",
        artifact_type="dataset", 
        metadata={"split": "train", "size": 10000}
    )
    
    # Version results with dependencies
    results_hash = versioner.version_artifact(
        artifact_path="results/evaluation.json",
        artifact_type="results",
        dependencies=[dataset_hash, model_hash]
    )
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Comprehensive metadata for versioned artifacts."""
    artifact_id: str
    artifact_type: str
    creation_timestamp: datetime
    file_path: str
    file_size: int
    content_hash: str
    dependencies: List[str]
    tags: List[str]
    description: str
    creator: str
    git_commit: Optional[str]
    environment_hash: str
    custom_metadata: Dict[str, Any]


@dataclass 
class ProvenanceRecord:
    """Provenance record for tracking artifact creation."""
    artifact_id: str
    creation_method: str
    input_artifacts: List[str]
    parameters: Dict[str, Any]
    execution_environment: Dict[str, Any]
    execution_duration: Optional[float]
    success: bool
    error_message: Optional[str]
    resource_usage: Optional[Dict[str, Any]]


class ArtifactHash:
    """Manages cryptographic hashing of artifacts with multiple algorithms."""
    
    def __init__(self):
        """Initialize hash manager."""
        self.supported_algorithms = ['sha256', 'sha1', 'md5', 'blake2b']
        self.default_algorithm = 'sha256'
    
    def hash_file(
        self, 
        file_path: Union[str, Path], 
        algorithm: str = None
    ) -> str:
        """Calculate hash of a single file.
        
        Args:
            file_path: Path to file to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm is None:
            algorithm = self.default_algorithm
            
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_func = getattr(hashlib, algorithm)()
        
        try:
            with open(file_path, 'rb') as f:
                # Process file in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            raise
    
    def hash_directory(
        self, 
        dir_path: Union[str, Path], 
        algorithm: str = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> str:
        """Calculate hash of directory contents.
        
        Args:
            dir_path: Path to directory to hash
            algorithm: Hash algorithm to use
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            
        Returns:
            Hexadecimal hash string representing directory contents
        """
        if algorithm is None:
            algorithm = self.default_algorithm
            
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        # Collect all relevant files
        files_to_hash = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                # Apply include/exclude patterns if specified
                relative_path = file_path.relative_to(dir_path)
                
                if include_patterns:
                    if not any(relative_path.match(pattern) for pattern in include_patterns):
                        continue
                
                if exclude_patterns:
                    if any(relative_path.match(pattern) for pattern in exclude_patterns):
                        continue
                
                files_to_hash.append(file_path)
        
        # Create composite hash
        hash_func = getattr(hashlib, algorithm)()
        
        for file_path in files_to_hash:
            # Include relative path in hash for structure
            relative_path = file_path.relative_to(dir_path)
            hash_func.update(str(relative_path).encode())
            
            # Include file content in hash
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def hash_string(self, content: str, algorithm: str = None) -> str:
        """Calculate hash of string content.
        
        Args:
            content: String content to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm is None:
            algorithm = self.default_algorithm
            
        hash_func = getattr(hashlib, algorithm)()
        hash_func.update(content.encode('utf-8'))
        return hash_func.hexdigest()
    
    def hash_object(self, obj: Any, algorithm: str = None) -> str:
        """Calculate hash of arbitrary Python object.
        
        Args:
            obj: Object to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        # Convert to JSON for consistent representation
        json_str = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)
        return self.hash_string(json_str, algorithm)


class ProvenanceTracker:
    """Tracks provenance and dependency chains for artifacts."""
    
    def __init__(self, provenance_dir: Union[str, Path]):
        """Initialize provenance tracker.
        
        Args:
            provenance_dir: Directory for storing provenance records
        """
        self.provenance_dir = Path(provenance_dir)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dependency graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}
        
        logger.info(f"Initialized ProvenanceTracker at: {self.provenance_dir}")
    
    def record_provenance(
        self,
        artifact_id: str,
        creation_method: str,
        input_artifacts: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        execution_environment: Optional[Dict[str, Any]] = None,
        execution_duration: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        resource_usage: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record provenance information for an artifact.
        
        Args:
            artifact_id: Unique identifier for artifact
            creation_method: Method used to create artifact
            input_artifacts: List of input artifact IDs
            parameters: Parameters used in creation
            execution_environment: Environment information
            execution_duration: Time taken for creation
            success: Whether creation was successful
            error_message: Error message if creation failed
            resource_usage: Resource usage statistics
            
        Returns:
            Path to saved provenance record
        """
        if input_artifacts is None:
            input_artifacts = []
        if parameters is None:
            parameters = {}
        if execution_environment is None:
            execution_environment = self._get_execution_environment()
        
        # Create provenance record
        record = ProvenanceRecord(
            artifact_id=artifact_id,
            creation_method=creation_method,
            input_artifacts=input_artifacts,
            parameters=parameters,
            execution_environment=execution_environment,
            execution_duration=execution_duration,
            success=success,
            error_message=error_message,
            resource_usage=resource_usage
        )
        
        # Update dependency graphs
        self.dependency_graph[artifact_id] = set(input_artifacts)
        for input_artifact in input_artifacts:
            if input_artifact not in self.reverse_dependency_graph:
                self.reverse_dependency_graph[input_artifact] = set()
            self.reverse_dependency_graph[input_artifact].add(artifact_id)
        
        # Save provenance record
        record_path = self.provenance_dir / f"{artifact_id}.json"
        with open(record_path, 'w') as f:
            json.dump(asdict(record), f, indent=2, default=str)
        
        logger.info(f"Recorded provenance for {artifact_id}")
        return str(record_path)
    
    def get_provenance(self, artifact_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record for an artifact.
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            Provenance record or None if not found
        """
        record_path = self.provenance_dir / f"{artifact_id}.json"
        
        if not record_path.exists():
            return None
        
        try:
            with open(record_path, 'r') as f:
                data = json.load(f)
            
            # Convert timestamp strings back to datetime if present
            if 'execution_environment' in data and 'timestamp' in data['execution_environment']:
                try:
                    data['execution_environment']['timestamp'] = datetime.fromisoformat(
                        data['execution_environment']['timestamp']
                    )
                except:
                    pass
            
            return ProvenanceRecord(**data)
        except Exception as e:
            logger.error(f"Error loading provenance for {artifact_id}: {e}")
            return None
    
    def get_dependency_chain(
        self, 
        artifact_id: str, 
        include_descendants: bool = False
    ) -> Dict[str, Any]:
        """Get complete dependency chain for an artifact.
        
        Args:
            artifact_id: Artifact identifier
            include_descendants: Whether to include dependent artifacts
            
        Returns:
            Dependency chain information
        """
        def get_ancestors(aid: str, visited: Set[str]) -> Set[str]:
            """Recursively get all ancestor artifacts."""
            if aid in visited:
                return set()  # Avoid cycles
            
            visited.add(aid)
            ancestors = set()
            
            if aid in self.dependency_graph:
                for dep in self.dependency_graph[aid]:
                    ancestors.add(dep)
                    ancestors.update(get_ancestors(dep, visited.copy()))
            
            return ancestors
        
        def get_descendants(aid: str, visited: Set[str]) -> Set[str]:
            """Recursively get all descendant artifacts."""
            if aid in visited:
                return set()  # Avoid cycles
            
            visited.add(aid)
            descendants = set()
            
            if aid in self.reverse_dependency_graph:
                for dep in self.reverse_dependency_graph[aid]:
                    descendants.add(dep)
                    descendants.update(get_descendants(dep, visited.copy()))
            
            return descendants
        
        # Get dependency information
        ancestors = get_ancestors(artifact_id, set())
        direct_dependencies = self.dependency_graph.get(artifact_id, set())
        
        chain_info = {
            'artifact_id': artifact_id,
            'direct_dependencies': list(direct_dependencies),
            'all_ancestors': list(ancestors),
            'dependency_depth': len(ancestors),
            'direct_dependency_count': len(direct_dependencies)
        }
        
        if include_descendants:
            descendants = get_descendants(artifact_id, set())
            direct_dependents = self.reverse_dependency_graph.get(artifact_id, set())
            
            chain_info.update({
                'direct_dependents': list(direct_dependents),
                'all_descendants': list(descendants),
                'dependent_count': len(descendants)
            })
        
        return chain_info
    
    def _get_execution_environment(self) -> Dict[str, Any]:
        """Get current execution environment information.
        
        Returns:
            Environment information dictionary
        """
        import platform
        import sys
        
        env_info = {
            'timestamp': datetime.now(),
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'working_directory': os.getcwd()
        }
        
        # Add git information if available
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_commit'] = git_commit
            
            git_dirty = subprocess.check_output(
                ['git', 'status', '--porcelain'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_dirty'] = len(git_dirty) > 0
        except:
            env_info['git_commit'] = None
            env_info['git_dirty'] = None
        
        # Add environment variables
        important_env_vars = [
            'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'PATH'
        ]
        env_info['environment_variables'] = {
            var: os.environ.get(var) for var in important_env_vars
        }
        
        return env_info


class VersionRegistry:
    """Central registry for all versioned artifacts."""
    
    def __init__(self, registry_dir: Union[str, Path]):
        """Initialize version registry.
        
        Args:
            registry_dir: Directory for registry storage
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "artifact_registry.json"
        self.metadata_dir = self.registry_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
        
        logger.info(f"Initialized VersionRegistry at: {self.registry_dir}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk.
        
        Returns:
            Registry dictionary
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry: {e}")
        
        # Return empty registry
        return {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'artifacts': {},
            'statistics': {
                'total_artifacts': 0,
                'total_size_bytes': 0,
                'artifact_types': {}
            }
        }
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            self.registry['last_updated'] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
            logger.info("Registry saved successfully")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_artifact(self, metadata: ArtifactMetadata) -> str:
        """Register a new artifact in the registry.
        
        Args:
            metadata: Artifact metadata
            
        Returns:
            Artifact ID
        """
        artifact_id = metadata.artifact_id
        
        # Save detailed metadata
        metadata_file = self.metadata_dir / f"{artifact_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Update registry with summary
        self.registry['artifacts'][artifact_id] = {
            'artifact_type': metadata.artifact_type,
            'creation_timestamp': metadata.creation_timestamp.isoformat(),
            'file_path': metadata.file_path,
            'file_size': metadata.file_size,
            'content_hash': metadata.content_hash,
            'dependencies': metadata.dependencies,
            'tags': metadata.tags,
            'metadata_file': str(metadata_file)
        }
        
        # Update statistics
        stats = self.registry['statistics']
        stats['total_artifacts'] += 1
        stats['total_size_bytes'] += metadata.file_size
        
        if metadata.artifact_type not in stats['artifact_types']:
            stats['artifact_types'][metadata.artifact_type] = 0
        stats['artifact_types'][metadata.artifact_type] += 1
        
        self._save_registry()
        logger.info(f"Registered artifact: {artifact_id}")
        
        return artifact_id
    
    def get_artifact_metadata(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Get complete metadata for an artifact.
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            Artifact metadata or None if not found
        """
        if artifact_id not in self.registry['artifacts']:
            return None
        
        metadata_file = self.metadata_dir / f"{artifact_id}.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file missing for {artifact_id}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Convert timestamp string back to datetime
            data['creation_timestamp'] = datetime.fromisoformat(data['creation_timestamp'])
            
            return ArtifactMetadata(**data)
        except Exception as e:
            logger.error(f"Error loading metadata for {artifact_id}: {e}")
            return None
    
    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """List artifacts matching criteria.
        
        Args:
            artifact_type: Filter by artifact type
            tags: Filter by tags (all must match)
            created_after: Filter by creation time
            created_before: Filter by creation time
            
        Returns:
            List of artifact information
        """
        results = []
        
        for artifact_id, summary in self.registry['artifacts'].items():
            # Apply filters
            if artifact_type and summary['artifact_type'] != artifact_type:
                continue
            
            if tags:
                if not all(tag in summary.get('tags', []) for tag in tags):
                    continue
            
            if created_after or created_before:
                creation_time = datetime.fromisoformat(summary['creation_timestamp'])
                if created_after and creation_time < created_after:
                    continue
                if created_before and creation_time > created_before:
                    continue
            
            # Add to results
            result = summary.copy()
            result['artifact_id'] = artifact_id
            results.append(result)
        
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x['creation_timestamp'], reverse=True)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.registry['statistics'].copy()


class ArtifactVersioner:
    """Main interface for artifact versioning and management."""
    
    def __init__(
        self,
        registry_dir: Union[str, Path] = "artifact_registry",
        storage_dir: Union[str, Path] = "artifact_storage"
    ):
        """Initialize artifact versioner.
        
        Args:
            registry_dir: Directory for registry files
            storage_dir: Directory for storing versioned artifacts
        """
        self.registry_dir = Path(registry_dir)
        self.storage_dir = Path(storage_dir)
        
        # Create directories
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hasher = ArtifactHash()
        self.provenance_tracker = ProvenanceTracker(self.registry_dir / "provenance")
        self.registry = VersionRegistry(self.registry_dir)
        
        logger.info(f"Initialized ArtifactVersioner")
        logger.info(f"  Registry: {self.registry_dir}")
        logger.info(f"  Storage: {self.storage_dir}")
    
    def version_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        copy_to_storage: bool = True
    ) -> str:
        """Version an artifact with complete metadata.
        
        Args:
            artifact_path: Path to artifact to version
            artifact_type: Type of artifact (e.g., 'dataset', 'model', 'results')
            description: Human-readable description
            tags: Optional tags for categorization
            dependencies: List of dependent artifact IDs
            custom_metadata: Additional metadata
            copy_to_storage: Whether to copy artifact to managed storage
            
        Returns:
            Unique artifact identifier
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        # Calculate content hash
        if artifact_path.is_file():
            content_hash = self.hasher.hash_file(artifact_path)
            file_size = artifact_path.stat().st_size
        else:
            content_hash = self.hasher.hash_directory(artifact_path)
            file_size = sum(f.stat().st_size for f in artifact_path.rglob('*') if f.is_file())
        
        # Generate artifact ID
        timestamp = datetime.now()
        artifact_id = f"{artifact_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{content_hash[:8]}"
        
        # Handle storage
        if copy_to_storage:
            storage_path = self._copy_to_storage(artifact_path, artifact_id)
        else:
            storage_path = str(artifact_path)
        
        # Create metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            creation_timestamp=timestamp,
            file_path=storage_path,
            file_size=file_size,
            content_hash=content_hash,
            dependencies=dependencies or [],
            tags=tags or [],
            description=description,
            creator=os.environ.get('USER', 'unknown'),
            git_commit=self._get_git_commit(),
            environment_hash=self.hasher.hash_object(self._get_environment_info()),
            custom_metadata=custom_metadata or {}
        )
        
        # Register artifact
        self.registry.register_artifact(metadata)
        
        # Record provenance
        self.provenance_tracker.record_provenance(
            artifact_id=artifact_id,
            creation_method="manual_versioning",
            input_artifacts=dependencies or [],
            parameters={'copy_to_storage': copy_to_storage},
            success=True
        )
        
        logger.info(f"Versioned artifact {artifact_id}: {artifact_path}")
        return artifact_id
    
    def _copy_to_storage(self, source_path: Path, artifact_id: str) -> str:
        """Copy artifact to managed storage.
        
        Args:
            source_path: Source path
            artifact_id: Artifact identifier
            
        Returns:
            Path to stored artifact
        """
        # Create storage subdirectory based on artifact type and date
        date_str = datetime.now().strftime("%Y/%m")
        storage_subdir = self.storage_dir / date_str
        storage_subdir.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            # Copy file
            storage_path = storage_subdir / f"{artifact_id}{source_path.suffix}"
            shutil.copy2(source_path, storage_path)
        else:
            # Archive directory
            storage_path = storage_subdir / f"{artifact_id}.zip"
            with zipfile.ZipFile(storage_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)
        
        logger.info(f"Copied artifact to storage: {storage_path}")
        return str(storage_path)
    
    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get complete information about an artifact.
        
        Args:
            artifact_id: Artifact identifier
            
        Returns:
            Complete artifact information
        """
        metadata = self.registry.get_artifact_metadata(artifact_id)
        if metadata is None:
            return None
        
        provenance = self.provenance_tracker.get_provenance(artifact_id)
        dependency_chain = self.provenance_tracker.get_dependency_chain(
            artifact_id, include_descendants=True
        )
        
        return {
            'metadata': asdict(metadata),
            'provenance': asdict(provenance) if provenance else None,
            'dependency_chain': dependency_chain
        }
    
    def list_artifacts(self, **kwargs) -> List[Dict[str, Any]]:
        """List artifacts matching criteria.
        
        Args:
            **kwargs: Filtering criteria (passed to registry.list_artifacts)
            
        Returns:
            List of artifact information
        """
        return self.registry.list_artifacts(**kwargs)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash.
        
        Returns:
            Git commit hash or None if not in a git repository
        """
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            return None
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for hashing.
        
        Returns:
            Environment information dictionary
        """
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'working_directory': os.getcwd(),
            'environment_variables': {
                var: os.environ.get(var) 
                for var in ['PYTHONPATH', 'CUDA_VISIBLE_DEVICES']
            }
        }


class ReproducibilityValidator:
    """Validates reproducibility of results using versioned artifacts."""
    
    def __init__(self, versioner: ArtifactVersioner):
        """Initialize reproducibility validator.
        
        Args:
            versioner: Artifact versioner instance
        """
        self.versioner = versioner
        
    def validate_reproducibility(
        self,
        results_artifact_id: str,
        rerun_results_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Validate that results can be reproduced.
        
        Args:
            results_artifact_id: Original results artifact ID
            rerun_results_path: Path to rerun results
            
        Returns:
            Reproducibility validation report
        """
        # Get original results metadata
        original_artifact = self.versioner.get_artifact(results_artifact_id)
        if original_artifact is None:
            raise ValueError(f"Original artifact not found: {results_artifact_id}")
        
        original_metadata = original_artifact['metadata']
        
        # Calculate hash of rerun results
        rerun_path = Path(rerun_results_path)
        if rerun_path.is_file():
            rerun_hash = self.versioner.hasher.hash_file(rerun_path)
        else:
            rerun_hash = self.versioner.hasher.hash_directory(rerun_path)
        
        # Compare hashes
        hashes_match = original_metadata['content_hash'] == rerun_hash
        
        # Prepare validation report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'original_artifact_id': results_artifact_id,
            'rerun_results_path': str(rerun_results_path),
            'hashes_match': hashes_match,
            'original_hash': original_metadata['content_hash'],
            'rerun_hash': rerun_hash,
            'original_creation_time': original_metadata['creation_timestamp'],
            'reproducible': hashes_match
        }
        
        # Add dependency validation
        dependency_chain = original_artifact['dependency_chain']
        report['dependency_validation'] = self._validate_dependencies(dependency_chain)
        
        # Overall reproducibility status
        report['fully_reproducible'] = (
            hashes_match and 
            report['dependency_validation']['all_dependencies_available']
        )
        
        return report
    
    def _validate_dependencies(
        self, 
        dependency_chain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that all dependencies are still available.
        
        Args:
            dependency_chain: Dependency chain information
            
        Returns:
            Dependency validation results
        """
        all_dependencies = (
            dependency_chain['direct_dependencies'] + 
            dependency_chain['all_ancestors']
        )
        
        missing_dependencies = []
        available_dependencies = []
        
        for dep_id in all_dependencies:
            if self.versioner.get_artifact(dep_id) is None:
                missing_dependencies.append(dep_id)
            else:
                available_dependencies.append(dep_id)
        
        return {
            'total_dependencies': len(all_dependencies),
            'available_dependencies': len(available_dependencies),
            'missing_dependencies': len(missing_dependencies),
            'missing_dependency_ids': missing_dependencies,
            'all_dependencies_available': len(missing_dependencies) == 0,
            'dependency_availability_rate': len(available_dependencies) / max(1, len(all_dependencies))
        }
    
    def create_reproducibility_package(
        self,
        results_artifact_id: str,
        output_dir: Union[str, Path],
        include_dependencies: bool = True
    ) -> str:
        """Create a complete reproducibility package.
        
        Args:
            results_artifact_id: Results artifact to package
            output_dir: Output directory for package
            include_dependencies: Whether to include all dependencies
            
        Returns:
            Path to created package
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get artifact information
        artifact_info = self.versioner.get_artifact(results_artifact_id)
        if artifact_info is None:
            raise ValueError(f"Artifact not found: {results_artifact_id}")
        
        # Create package directory
        package_name = f"repro_package_{results_artifact_id}"
        package_dir = output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Copy main artifact
        main_metadata = artifact_info['metadata']
        main_artifact_path = Path(main_metadata['file_path'])
        if main_artifact_path.exists():
            shutil.copy2(main_artifact_path, package_dir / main_artifact_path.name)
        
        # Copy dependencies if requested
        if include_dependencies:
            deps_dir = package_dir / "dependencies"
            deps_dir.mkdir(exist_ok=True)
            
            dependency_chain = artifact_info['dependency_chain']
            all_deps = (
                dependency_chain['direct_dependencies'] + 
                dependency_chain['all_ancestors']
            )
            
            for dep_id in all_deps:
                dep_info = self.versioner.get_artifact(dep_id)
                if dep_info:
                    dep_metadata = dep_info['metadata']
                    dep_path = Path(dep_metadata['file_path'])
                    if dep_path.exists():
                        shutil.copy2(dep_path, deps_dir / dep_path.name)
        
        # Create metadata file
        package_metadata = {
            'package_creation_time': datetime.now().isoformat(),
            'main_artifact': artifact_info,
            'reproducibility_instructions': self._create_repro_instructions(artifact_info)
        }
        
        with open(package_dir / "package_metadata.json", 'w') as f:
            json.dump(package_metadata, f, indent=2, default=str)
        
        # Create ZIP archive
        archive_path = output_dir / f"{package_name}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up directory
        shutil.rmtree(package_dir)
        
        logger.info(f"Created reproducibility package: {archive_path}")
        return str(archive_path)
    
    def _create_repro_instructions(self, artifact_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create reproducibility instructions.
        
        Args:
            artifact_info: Artifact information
            
        Returns:
            Reproducibility instructions
        """
        metadata = artifact_info['metadata']
        provenance = artifact_info.get('provenance', {})
        
        instructions = {
            'overview': f"Instructions for reproducing {metadata['artifact_id']}",
            'artifact_type': metadata['artifact_type'],
            'original_creation_time': metadata['creation_timestamp'],
            'dependencies_required': len(metadata['dependencies']),
            'environment_requirements': {
                'git_commit': metadata.get('git_commit'),
                'python_environment': 'See provenance for detailed environment info'
            }
        }
        
        if provenance:
            instructions['creation_method'] = provenance.get('creation_method', 'unknown')
            instructions['parameters'] = provenance.get('parameters', {})
            instructions['execution_environment'] = provenance.get('execution_environment', {})
        
        return instructions


# Example usage and testing
if __name__ == "__main__":
    # Initialize versioner
    versioner = ArtifactVersioner(
        registry_dir="test_registry",
        storage_dir="test_storage"
    )
    
    # Create test file
    test_file = Path("test_artifact.txt")
    with open(test_file, 'w') as f:
        f.write("This is a test artifact for versioning.")
    
    try:
        # Version the test artifact
        artifact_id = versioner.version_artifact(
            artifact_path=test_file,
            artifact_type="test_data",
            description="Test artifact for versioning system",
            tags=["test", "example"],
            custom_metadata={"purpose": "system_validation"}
        )
        
        print(f"Created artifact: {artifact_id}")
        
        # Get artifact information
        artifact_info = versioner.get_artifact(artifact_id)
        print("Artifact info:")
        print(json.dumps(artifact_info, indent=2, default=str))
        
        # List artifacts
        artifacts = versioner.list_artifacts(artifact_type="test_data")
        print(f"Found {len(artifacts)} test_data artifacts")
        
        # Create reproducibility validator
        validator = ReproducibilityValidator(versioner)
        
        # Test reproducibility validation
        validation_report = validator.validate_reproducibility(
            results_artifact_id=artifact_id,
            rerun_results_path=test_file  # Same file for testing
        )
        
        print("Reproducibility validation:")
        print(json.dumps(validation_report, indent=2, default=str))
        
    finally:
        # Clean up
        test_file.unlink(missing_ok=True)
        import shutil
        shutil.rmtree("test_registry", ignore_errors=True)
        shutil.rmtree("test_storage", ignore_errors=True)