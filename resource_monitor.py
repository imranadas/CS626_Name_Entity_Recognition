# resource_monitor.py

import time
import psutil
import threading
from logger_config import setup_logger

logger = setup_logger('resource_monitor')

class ResourceMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.thread = None
    
    def _monitor(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.Process().memory_info()
            memory_percent = psutil.virtual_memory().percent
            
            logger.info(f"Resource Usage - "
                       f"CPU: {cpu_percent}%, "
                       f"Memory: {memory.rss / 1024 / 1024:.1f} MB "
                       f"({memory_percent}%)")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring resources"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring resources"""
        self.running = False
        if self.thread:
            self.thread.join()
    