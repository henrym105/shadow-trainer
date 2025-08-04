import boto3
import schedule
import time
from datetime import datetime
import pytz

# Initialize the EC2 client
ec2_client = boto3.client('ec2')

INSTANCE_ID = 'i-054e2e406c626e3ea'

def start_instance():
    print(f"Starting EC2 instance: {INSTANCE_ID}")
    ec2_client.start_instances(InstanceIds=[INSTANCE_ID])
    print("Instance started.")

def stop_instance():
    print(f"Stopping EC2 instance: {INSTANCE_ID}")
    ec2_client.stop_instances(InstanceIds=[INSTANCE_ID])
    print("Instance stopped.")

def schedule_tasks():
    # Set the timezone to local time
    local_tz = pytz.timezone('America/New_York')

    # Schedule the start and stop times
    schedule.every().day.at("09:00").do(start_instance)
    schedule.every().day.at("22:00").do(stop_instance)

    while True:
        # Get the current time in the local timezone
        now = datetime.now(local_tz)
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_tasks()