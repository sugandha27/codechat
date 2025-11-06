"""
S3 to Pinecone Vector Restoration Script

This script provides functionality to restore vector archives from S3 back into Pinecone.
It enables rollback to previous commit states and recovery from data loss.
"""

import json
import boto3
import os
import logging
import hashlib
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the core modules to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from lib.pine import PineconeClient
except ImportError:
    sys.path.append('/opt/python')
    from pine import PineconeClient

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'codechat-main')
S3_ARCHIVE_BUCKET = os.getenv('S3_ARCHIVE_BUCKET', 'codechat-vectors')
BATCH_SIZE = int(os.getenv('RESTORE_BATCH_SIZE', '100'))

class VectorRestorer:
    """Handles restoration of vector archives from S3 to Pinecone."""
    
    def __init__(self, pinecone_namespace: str, s3_bucket: str):
        self.pinecone_client = PineconeClient(namespace=pinecone_namespace)
        self.s3_bucket = s3_bucket
        logger.info(f"VectorRestorer initialized for Pinecone namespace '{pinecone_namespace}' and S3 bucket '{s3_bucket}'")

    def list_archives(self, repo_name: str) -> List[Dict[str, str]]:
        """List available vector archives for a repository."""
        try:
            prefix = f"vectors/{repo_name}/"
            response = s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix, Delimiter='/')
            
            archives = []
            for common_prefix in response.get('CommonPrefixes', []):
                commit_sha = common_prefix.get('Prefix').split('/')[-2]
                archives.append({
                    'repo_name': repo_name,
                    'commit_sha': commit_sha,
                    's3_prefix': common_prefix.get('Prefix')
                })
            
            logger.info(f"Found {len(archives)} archives for repository '{repo_name}'")
            return sorted(archives, key=lambda x: x['commit_sha'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list archives for {repo_name}: {e}")
            return []

    def restore_archive(self, repo_name: str, commit_sha: str, set_live: bool = False) -> Dict[str, Any]:
        """
        Restore a specific vector archive to Pinecone.
        
        Args:
            repo_name: The name of the repository.
            commit_sha: The commit SHA of the archive to restore.
            set_live: If True, sets the restored vectors to live:true and deactivates others.
            
        Returns:
            A dictionary with the restoration results.
        """
        try:
            logger.info(f"Starting restoration for {repo_name}@{commit_sha}")
            
            # 1. Download and validate the archive from S3
            archive_data = self._download_and_validate_archive(repo_name, commit_sha)
            if not archive_data:
                return {'success': False, 'error': 'Archive download or validation failed'}
            
            vectors = archive_data['vectors']
            logger.info(f"Successfully downloaded and validated archive with {len(vectors)} vectors")
            
            # 2. Batch upsert vectors to Pinecone
            upsert_result = self._batch_upsert_to_pinecone(vectors)
            if not upsert_result['success']:
                return {'success': False, 'error': f"Pinecone upsert failed: {upsert_result['error']}"}
            
            logger.info(f"Successfully upserted {upsert_result['upserted_count']} vectors to Pinecone")
            
            # 3. Optionally, activate the restored vectors
            activation_result = None
            if set_live:
                logger.info("Activating restored vectors...")
                activation_result = self._activate_restored_vectors(repo_name, commit_sha)
                if not activation_result['success']:
                    logger.warning(f"Failed to activate vectors: {activation_result['error']}")
            
            return {
                'success': True,
                'repo_name': repo_name,
                'commit_sha': commit_sha,
                'vectors_restored': upsert_result['upserted_count'],
                'activation_result': activation_result
            }
            
        except Exception as e:
            logger.error(f"Restoration failed for {repo_name}@{commit_sha}: {e}")
            return {'success': False, 'error': str(e)}

    def _download_and_validate_archive(self, repo_name: str, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Download and validate an archive from S3."""
        try:
            s3_prefix = f"vectors/{repo_name}/{commit_sha}"
            archive_key = f"{s3_prefix}/archive.json"
            checksum_key = f"{s3_prefix}/checksums.json"
            
            # Download archive file
            archive_obj = s3_client.get_object(Bucket=self.s3_bucket, Key=archive_key)
            archive_content = archive_obj['Body'].read()
            
            # Download checksum file
            checksum_obj = s3_client.get_object(Bucket=self.s3_bucket, Key=checksum_key)
            checksum_data = json.loads(checksum_obj['Body'].read())
            
            # Validate checksum
            expected_checksum = checksum_data['archive_sha256']
            actual_checksum = hashlib.sha256(archive_content).hexdigest()
            
            if actual_checksum != expected_checksum:
                logger.error("Archive checksum mismatch! The archive may be corrupt.")
                return None
            
            logger.info("Archive checksum validated successfully")
            return json.loads(archive_content)
            
        except Exception as e:
            logger.error(f"Failed to download or validate archive: {e}")
            return None

    def _batch_upsert_to_pinecone(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch upsert vectors to Pinecone."""
        try:
            total_upserted = 0
            
            for i in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[i:i + BATCH_SIZE]
                
                # Prepare vectors for Pinecone upsert
                pinecone_vectors = []
                for vector in batch:
                    # Ensure metadata is correctly formatted
                    metadata = vector.get('metadata', {})
                    metadata['restored_at'] = datetime.utcnow().isoformat()
                    
                    pinecone_vectors.append({
                        'id': vector['id'],
                        'values': vector['values'],
                        'metadata': metadata
                    })
                
                upsert_result = self.pinecone_client.upsert(pinecone_vectors)
                
                upserted_count = upsert_result.get('upserted_count', 0)
                total_upserted += upserted_count
                
                logger.info(f"Upserted batch {i//BATCH_SIZE + 1}: {upserted_count} vectors")
            
            return {'success': True, 'upserted_count': total_upserted}
            
        except Exception as e:
            logger.error(f"Pinecone batch upsert failed: {e}")
            return {'success': False, 'error': str(e)}

    def _activate_restored_vectors(self, repo_name: str, commit_sha: str) -> Dict[str, Any]:
        """Activate restored vectors and deactivate others for the repo."""
        try:
            # Deactivate all other live vectors for this repo
            deactivation_result = self.pinecone_client.update_metadata_by_filter(
                filter_dict={'repo': repo_name, 'live': True},
                update_dict={'live': False}
            )
            logger.info(f"Deactivated {deactivation_result.get('updated_count', 0)} old vectors for {repo_name}")
            
            # Activate the restored vectors
            activation_result = self.pinecone_client.update_metadata_by_filter(
                filter_dict={'repo': repo_name, 'commit_sha': commit_sha},
                update_dict={'live': True}
            )
            logger.info(f"Activated {activation_result.get('updated_count', 0)} restored vectors for {commit_sha}")
            
            return {
                'success': True,
                'deactivated_count': deactivation_result.get('updated_count', 0),
                'activated_count': activation_result.get('updated_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to activate restored vectors: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Command-line interface for the vector restoration script."""
    parser = argparse.ArgumentParser(description="Restore vector archives from S3 to Pinecone.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 'list' command
    list_parser = subparsers.add_parser('list', help="List available archives for a repository.")
    list_parser.add_argument('repo_name', type=str, help="Repository name (e.g., 'owner/repo').")
    
    # 'restore' command
    restore_parser = subparsers.add_parser('restore', help="Restore a specific archive.")
    restore_parser.add_argument('repo_name', type=str, help="Repository name.")
    restore_parser.add_argument('commit_sha', type=str, help="Commit SHA of the archive to restore.")
    restore_parser.add_argument('--set-live', action='store_true', help="Set the restored vectors as live.")
    
    args = parser.parse_args()
    
    restorer = VectorRestorer(
        pinecone_namespace=PINECONE_NAMESPACE,
        s3_bucket=S3_ARCHIVE_BUCKET
    )
    
    if args.command == 'list':
        archives = restorer.list_archives(args.repo_name)
        if archives:
            print(f"Available archives for {args.repo_name}:")
            for archive in archives:
                print(f"  - Commit: {archive['commit_sha']}")
        else:
            print(f"No archives found for {args.repo_name}.")
            
    elif args.command == 'restore':
        result = restorer.restore_archive(args.repo_name, args.commit_sha, args.set_live)
        
        if result['success']:
            logger.info("Restoration completed successfully.")
            print(json.dumps(result, indent=2))
        else:
            logger.error(f"Restoration failed: {result['error']}")
            print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
