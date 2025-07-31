# EC2 Scheduler

This project provides a simple solution for scheduling the start and stop of an AWS EC2 instance using Python. The scheduling is done using the AWS SDK for Python (Boto3) and is designed to run on a separate EC2 instance within the AWS Free Tier.

## Project Structure

```
ec2-scheduler
├── src
│   ├── scheduler.py      # Main logic for scheduling EC2 instance start and stop
│   └── utils.py          # Utility functions for time calculations and AWS configurations
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd ec2-scheduler
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Configure AWS Credentials**:
   Ensure that your AWS credentials are configured. You can do this by running:
   ```
   aws configure
   ```
   Provide your AWS Access Key, Secret Key, region, and output format when prompted.

## Usage

To start and stop the EC2 instance automatically:

1. **Edit the `scheduler.py` file** to set the correct instance ID and any other configurations as needed.

2. **Run the scheduler**:
   ```
   python src/scheduler.py
   ```

The script will start the specified EC2 instance at 9 AM and stop it at 10 PM local time every day.

## Additional Information

- Ensure that the instance you are scheduling is within the AWS Free Tier limits to avoid unexpected charges.
- You may want to set up a cron job or a similar scheduling mechanism to run the `scheduler.py` script at regular intervals (e.g., every hour) to check if it needs to start or stop the instance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.