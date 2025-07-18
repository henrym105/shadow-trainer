#!/bin/bash

# CloudWatch Custom Metrics Publishing for Shadow Trainer
# Publishes GPU utilization, Celery queue metrics, and application performance data

import boto3
import psutil
import redis
import time
import json
import subprocess
from datetime import datetime

class ShadowTrainerMetrics:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch', region_name='us-east-2')
        self.redis_client = redis.Redis(host='shadow-trainer-redis.cache.amazonaws.com', port=6379, db=0)
        self.namespace = 'ShadowTrainer'
        
    def get_gpu_utilization(self):
        """Get GPU utilization from nvidia-smi"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) == 3:
                        utilization = float(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        memory_percent = (memory_used / memory_total) * 100
                        gpu_data.append({
                            'gpu_id': i,
                            'utilization': utilization,
                            'memory_percent': memory_percent,
                            'memory_used_mb': memory_used,
                            'memory_total_mb': memory_total
                        })
                return gpu_data
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return []
        
    def get_celery_queue_metrics(self):
        """Get Celery queue depth and worker metrics"""
        try:
            # Get queue lengths
            video_queue_length = self.redis_client.llen('celery')
            
            # Get worker stats (requires celery inspect)
            worker_stats = {}
            try:
                from celery import Celery
                app = Celery('shadow_trainer')
                app.config_from_object('api_backend.celery_config')
                
                inspect = app.control.inspect()
                active_tasks = inspect.active()
                stats = inspect.stats()
                
                if active_tasks:
                    total_active = sum(len(tasks) for tasks in active_tasks.values())
                    worker_stats['active_tasks'] = total_active
                    worker_stats['worker_count'] = len(active_tasks)
                else:
                    worker_stats['active_tasks'] = 0
                    worker_stats['worker_count'] = 0
                    
            except Exception as e:
                print(f"Error getting worker stats: {e}")
                worker_stats = {'active_tasks': 0, 'worker_count': 0}
            
            return {
                'queue_depth': video_queue_length,
                'active_tasks': worker_stats['active_tasks'],
                'worker_count': worker_stats['worker_count']
            }
        except Exception as e:
            print(f"Error getting Celery metrics: {e}")
            return {'queue_depth': 0, 'active_tasks': 0, 'worker_count': 0}
    
    def get_system_metrics(self):
        """Get system CPU, memory, and disk metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0]  # 1-minute load average
        }
    
    def publish_metrics(self):
        """Publish all metrics to CloudWatch"""
        timestamp = datetime.utcnow()
        
        # GPU Metrics
        gpu_data = self.get_gpu_utilization()
        for gpu in gpu_data:
            self.cloudwatch.put_metric_data(
                Namespace=f'{self.namespace}/GPU',
                MetricData=[
                    {
                        'MetricName': 'GPUUtilization',
                        'Value': gpu['utilization'],
                        'Unit': 'Percent',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'ServiceName', 'Value': 'shadow-trainer-api'},
                            {'Name': 'GPUId', 'Value': str(gpu['gpu_id'])}
                        ]
                    },
                    {
                        'MetricName': 'GPUMemoryUtilization',
                        'Value': gpu['memory_percent'],
                        'Unit': 'Percent',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'ServiceName', 'Value': 'shadow-trainer-api'},
                            {'Name': 'GPUId', 'Value': str(gpu['gpu_id'])}
                        ]
                    }
                ]
            )
        
        # Celery Metrics
        celery_metrics = self.get_celery_queue_metrics()
        self.cloudwatch.put_metric_data(
            Namespace=f'{self.namespace}/Celery',
            MetricData=[
                {
                    'MetricName': 'CeleryQueueDepth',
                    'Value': celery_metrics['queue_depth'],
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'QueueName', 'Value': 'video_processing'}]
                },
                {
                    'MetricName': 'ActiveTasks',
                    'Value': celery_metrics['active_tasks'],
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ServiceName', 'Value': 'shadow-trainer-api'}]
                },
                {
                    'MetricName': 'WorkerCount',
                    'Value': celery_metrics['worker_count'],
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ServiceName', 'Value': 'shadow-trainer-api'}]
                }
            ]
        )
        
        # System Metrics
        system_metrics = self.get_system_metrics()
        self.cloudwatch.put_metric_data(
            Namespace=f'{self.namespace}/System',
            MetricData=[
                {
                    'MetricName': 'CPUUtilization',
                    'Value': system_metrics['cpu_percent'],
                    'Unit': 'Percent',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ServiceName', 'Value': 'shadow-trainer-api'}]
                },
                {
                    'MetricName': 'MemoryUtilization',
                    'Value': system_metrics['memory_percent'],
                    'Unit': 'Percent',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ServiceName', 'Value': 'shadow-trainer-api'}]
                },
                {
                    'MetricName': 'DiskUtilization',
                    'Value': system_metrics['disk_percent'],
                    'Unit': 'Percent',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'ServiceName', 'Value': 'shadow-trainer-api'}]
                }
            ]
        )
        
        print(f"Metrics published at {timestamp}")

if __name__ == "__main__":
    metrics = ShadowTrainerMetrics()
    
    # Run continuously
    while True:
        try:
            metrics.publish_metrics()
            time.sleep(60)  # Publish every minute
        except Exception as e:
            print(f"Error publishing metrics: {e}")
            time.sleep(60)
