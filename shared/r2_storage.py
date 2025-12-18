"""Cloudflare R2 storage utilities."""
import boto3
import os
import logging
from typing import Optional, BinaryIO, Tuple
from botocore.exceptions import ClientError
import uuid
from datetime import datetime
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class R2Storage:
    """Cloudflare R2 storage client (S3-compatible)."""
    
    def __init__(self, connection_url: str):
        """
        Initialize R2 storage client from connection URL.
        
        Args:
            connection_url: R2 connection string
                Format: r2://access_key_id:secret_access_key@account_id/bucket_name?public_url=https://pub-xxxxx.r2.dev
                Example: r2://key:secret@abc123/bucket-name?public_url=https://pub-xxxxx.r2.dev
        """
        # Parse connection URL
        parsed = urlparse(connection_url)
        
        if parsed.scheme != 'r2':
            raise ValueError(f"Invalid R2 connection URL scheme: {parsed.scheme}. Expected 'r2://'")
        
        # Extract credentials from netloc (user:pass@host)
        if '@' not in parsed.netloc:
            raise ValueError("Invalid R2 connection URL format. Expected: r2://access_key_id:secret_access_key@account_id/bucket_name")
        
        auth_part, account_id = parsed.netloc.rsplit('@', 1)
        
        # Extract access key and secret
        if ':' not in auth_part:
            raise ValueError("Invalid R2 credentials format. Expected: access_key_id:secret_access_key")
        
        access_key_id, secret_access_key = auth_part.split(':', 1)
        
        # Extract bucket name from path
        bucket_name = parsed.path.lstrip('/')
        if not bucket_name:
            raise ValueError("Bucket name is required in R2 connection URL")
        
        # Extract optional parameters from query string
        query_params = parse_qs(parsed.query)
        public_url = query_params.get('public_url', [None])[0]
        
        self.account_id = account_id
        self.bucket_name = bucket_name
        self.public_url = public_url
        
        # Auto-generate endpoint URL from account_id
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        # Create S3-compatible client for R2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'  # R2 doesn't use regions
        )
    
    def generate_file_key(self, original_filename: str, prefix: str = "uploads") -> str:
        """Generate a unique file key for R2 storage."""
        # Create a unique key with timestamp and UUID
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(original_filename)[1] or ""
        return f"{prefix}/{timestamp}/{file_id}{file_ext}"
    
    async def upload_file(
        self,
        file_path: str,
        file_key: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Upload a file to R2.
        
        Args:
            file_path: Local file path to upload
            file_key: R2 object key (optional, will generate if not provided)
            content_type: MIME type of the file
            metadata: Additional metadata to store
            
        Returns:
            dict with 'key', 'url', 'size', 'etag'
        """
        try:
            import aiofiles
            import asyncio
            
            # Generate key if not provided
            if not file_key:
                original_filename = os.path.basename(file_path)
                file_key = self.generate_file_key(original_filename)
            
            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Prepare metadata
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if metadata:
                extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
            
            # Upload to R2 (using sync boto3 in async context)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=file_key,
                    Body=file_content,
                    **extra_args
                )
            )
            
            # Generate URL for private bucket access
            if self.public_url:
                # Public bucket - use public URL
                file_url = f"{self.public_url}/{file_key}"
            else:
                # Private bucket - no public URL available
                # Return None for URL to indicate private access only
                file_url = None

            logger.info(f"File uploaded to R2: {file_key}")
            if file_url:
                logger.info(f"Public URL available: {file_url}")
            else:
                logger.info("Private bucket - file accessible via signed URLs or API only")

            return {
                'key': file_key,
                'url': file_url,  # None for private buckets
                'size': len(file_content),
                'bucket': self.bucket_name,
                'is_private': file_url is None
            }
            
        except ClientError as e:
            logger.error(f"R2 upload error: {e}")
            raise Exception(f"Failed to upload file to R2: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {e}")
            raise
    
    async def delete_file(self, file_key: str) -> bool:
        """Delete a file from R2."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=file_key
                )
            )
            logger.info(f"File deleted from R2: {file_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file from R2: {e}")
            return False
    
    def get_file_url(self, file_key: str) -> Optional[str]:
        """Get public URL for a file key (None for private buckets)."""
        if self.public_url:
            return f"{self.public_url}/{file_key}"
        # Private bucket - no public URL available
        return None

    def generate_signed_url(self, file_key: str, expiration: int = 3600) -> str:
        """
        Generate a signed URL for private bucket access.

        Args:
            file_key: The R2 object key
            expiration: URL expiration time in seconds (default 1 hour)

        Returns:
            Signed URL string
        """
        try:
            import datetime
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key
                },
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {file_key}: {e}")
            raise

