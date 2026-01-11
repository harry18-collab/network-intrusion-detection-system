import os
import subprocess
import logging

class S3Sync:
    def __init__(self):
        self.aws_available = self._check_aws_cli()
    
    def _check_aws_cli(self):
        """Check if AWS CLI is available"""
        try:
            subprocess.run(["aws", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.warning("AWS CLI not found. S3 sync operations will be skipped.")
            return False
    
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        if not self.aws_available:
            logging.info(f"Skipping S3 sync to {aws_bucket_url} - AWS CLI not available")
            return
        
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Successfully synced {folder} to S3")
            else:
                logging.error(f"S3 sync failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Error during S3 sync: {str(e)}")
    
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        if not self.aws_available:
            logging.info(f"Skipping S3 sync from {aws_bucket_url} - AWS CLI not available")
            return
        
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Successfully synced from S3 to {folder}")
            else:
                logging.error(f"S3 sync failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Error during S3 sync: {str(e)}")
