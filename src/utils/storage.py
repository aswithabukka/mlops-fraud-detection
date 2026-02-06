"""
Storage abstraction layer for local and cloud storage.

Provides unified interface for:
- Local filesystem storage
- AWS S3 storage
- MinIO storage (S3-compatible)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
import json
import pickle

import pandas as pd
import boto3
from botocore.exceptions import ClientError

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, data: Any, path: str, **kwargs) -> bool:
        """Save data to storage."""
        pass

    @abstractmethod
    def load(self, path: str, **kwargs) -> Any:
        """Load data from storage."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete path from storage."""
        pass

    @abstractmethod
    def list_files(self, prefix: str) -> list:
        """List files with given prefix."""
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Example:
        >>> storage = LocalStorage(base_path="data")
        >>> storage.save(df, "raw/transactions.csv")
        >>> df = storage.load("raw/transactions.csv")
    """

    def __init__(self, base_path: Union[str, Path] = "."):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalStorage at {self.base_path}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path."""
        return self.base_path / path

    def save(self, data: Any, path: str, **kwargs) -> bool:
        """
        Save data to local file.

        Supports:
        - DataFrame: CSV, Parquet, JSON
        - dict/list: JSON
        - bytes: Binary
        - Any object: Pickle

        Args:
            data: Data to save
            path: Relative path within base_path
            **kwargs: Additional arguments (e.g., index=False for DataFrames)

        Returns:
            True if successful
        """
        try:
            filepath = self._resolve_path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Determine file type and save accordingly
            if isinstance(data, pd.DataFrame):
                if path.endswith('.csv'):
                    data.to_csv(filepath, index=kwargs.get('index', False))
                elif path.endswith('.parquet'):
                    data.to_parquet(filepath, index=kwargs.get('index', False))
                elif path.endswith('.json'):
                    data.to_json(filepath, orient=kwargs.get('orient', 'records'))
                else:
                    raise ValueError(f"Unsupported DataFrame format: {path}")

            elif isinstance(data, (dict, list)):
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

            elif isinstance(data, bytes):
                with open(filepath, 'wb') as f:
                    f.write(data)

            else:
                # Default to pickle for other objects
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)

            logger.info(f"Saved data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save to {path}: {e}")
            return False

    def load(self, path: str, **kwargs) -> Any:
        """
        Load data from local file.

        Args:
            path: Relative path within base_path
            **kwargs: Additional arguments for loading

        Returns:
            Loaded data
        """
        try:
            filepath = self._resolve_path(path)

            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            # Determine file type and load accordingly
            if path.endswith('.csv'):
                return pd.read_csv(filepath, **kwargs)
            elif path.endswith('.parquet'):
                return pd.read_parquet(filepath, **kwargs)
            elif path.endswith('.json'):
                if kwargs.get('as_dataframe', False):
                    return pd.read_json(filepath, **kwargs)
                else:
                    with open(filepath, 'r') as f:
                        return json.load(f)
            else:
                # Default to pickle
                with open(filepath, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
            raise

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return self._resolve_path(path).exists()

    def delete(self, path: str) -> bool:
        """Delete file or directory."""
        try:
            filepath = self._resolve_path(path)
            if filepath.is_file():
                filepath.unlink()
            elif filepath.is_dir():
                import shutil
                shutil.rmtree(filepath)
            logger.info(f"Deleted {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """List files with given prefix."""
        base = self._resolve_path(prefix) if prefix else self.base_path
        if base.is_dir():
            return [str(p.relative_to(self.base_path)) for p in base.rglob("*") if p.is_file()]
        return []


class S3Storage(StorageBackend):
    """
    AWS S3 storage backend.

    Supports both AWS S3 and MinIO (S3-compatible).

    Example:
        >>> storage = S3Storage(bucket="my-bucket")
        >>> storage.save(df, "raw/transactions.csv")
        >>> df = storage.load("raw/transactions.csv")
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name (default from settings)
            aws_access_key_id: AWS access key (default from settings)
            aws_secret_access_key: AWS secret key (default from settings)
            endpoint_url: Custom endpoint URL for MinIO (default from settings)
            region: AWS region (default from settings)
        """
        self.bucket = bucket or settings.s3_bucket_name
        self.endpoint_url = endpoint_url or settings.aws_endpoint_url
        self.region = region or settings.aws_region

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id or settings.aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key or settings.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region,
        )

        # Create bucket if it doesn't exist (MinIO)
        self._ensure_bucket_exists()

        logger.info(f"Initialized S3Storage with bucket '{self.bucket}'")

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                if self.endpoint_url:
                    # MinIO or custom endpoint
                    self.s3_client.create_bucket(Bucket=self.bucket)
                else:
                    # AWS S3
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                logger.info(f"Created bucket '{self.bucket}'")
            except Exception as e:
                logger.warning(f"Could not create bucket: {e}")

    def save(self, data: Any, path: str, **kwargs) -> bool:
        """
        Save data to S3.

        Args:
            data: Data to save
            path: S3 key (path within bucket)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            # Save to temporary local file first
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=Path(path).suffix) as tmp:
                tmp_path = tmp.name

                # Save data to temp file
                if isinstance(data, pd.DataFrame):
                    if path.endswith('.csv'):
                        data.to_csv(tmp_path, index=False)
                    elif path.endswith('.parquet'):
                        data.to_parquet(tmp_path, index=False)
                elif isinstance(data, (dict, list)):
                    with open(tmp_path, 'w') as f:
                        json.dump(data, f)
                elif isinstance(data, bytes):
                    with open(tmp_path, 'wb') as f:
                        f.write(data)
                else:
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(data, f)

                # Upload to S3
                self.s3_client.upload_file(tmp_path, self.bucket, path)

                # Clean up temp file
                Path(tmp_path).unlink()

            logger.info(f"Saved data to s3://{self.bucket}/{path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save to S3 {path}: {e}")
            return False

    def load(self, path: str, **kwargs) -> Any:
        """
        Load data from S3.

        Args:
            path: S3 key (path within bucket)
            **kwargs: Additional arguments

        Returns:
            Loaded data
        """
        try:
            # Download to temporary local file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=Path(path).suffix) as tmp:
                tmp_path = tmp.name
                self.s3_client.download_file(self.bucket, path, tmp_path)

                # Load from temp file
                if path.endswith('.csv'):
                    data = pd.read_csv(tmp_path, **kwargs)
                elif path.endswith('.parquet'):
                    data = pd.read_parquet(tmp_path, **kwargs)
                elif path.endswith('.json'):
                    if kwargs.get('as_dataframe', False):
                        data = pd.read_json(tmp_path)
                    else:
                        with open(tmp_path, 'r') as f:
                            data = json.load(f)
                else:
                    with open(tmp_path, 'rb') as f:
                        data = pickle.load(f)

                # Clean up
                Path(tmp_path).unlink()

            logger.info(f"Loaded data from s3://{self.bucket}/{path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load from S3 {path}: {e}")
            raise

    def exists(self, path: str) -> bool:
        """Check if S3 object exists."""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=path)
            return True
        except ClientError:
            return False

    def delete(self, path: str) -> bool:
        """Delete S3 object."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=path)
            logger.info(f"Deleted s3://{self.bucket}/{path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3 {path}: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """List S3 objects with given prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []


def get_storage(storage_type: str = "local", **kwargs) -> StorageBackend:
    """
    Factory function to get storage backend.

    Args:
        storage_type: Type of storage ('local' or 's3')
        **kwargs: Additional arguments for storage initialization

    Returns:
        StorageBackend instance
    """
    if storage_type == "local":
        return LocalStorage(**kwargs)
    elif storage_type == "s3":
        return S3Storage(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


if __name__ == "__main__":
    """Test storage backends."""
    import numpy as np

    # Test data
    test_df = pd.DataFrame({
        'a': np.random.rand(100),
        'b': np.random.randint(0, 10, 100),
        'c': ['cat'] * 50 + ['dog'] * 50,
    })

    print("Testing LocalStorage...")
    local_storage = LocalStorage(base_path="test_storage")

    # Save and load DataFrame
    local_storage.save(test_df, "test/data.csv")
    loaded_df = local_storage.load("test/data.csv")
    print(f"✅ CSV: Original shape {test_df.shape}, Loaded shape {loaded_df.shape}")

    # Save and load Parquet
    local_storage.save(test_df, "test/data.parquet")
    loaded_df = local_storage.load("test/data.parquet")
    print(f"✅ Parquet: Loaded shape {loaded_df.shape}")

    # Save and load JSON
    test_dict = {'key': 'value', 'list': [1, 2, 3]}
    local_storage.save(test_dict, "test/data.json")
    loaded_dict = local_storage.load("test/data.json")
    print(f"✅ JSON: {loaded_dict}")

    # List files
    files = local_storage.list_files("test")
    print(f"✅ Listed {len(files)} files: {files}")

    # Clean up
    local_storage.delete("test")
    print("✅ LocalStorage tests passed!")

    print("\nNote: S3Storage requires AWS/MinIO credentials to test")
