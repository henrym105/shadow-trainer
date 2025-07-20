// Static list of sample S3 videos and their thumbnails (for demo)
// In production, this could be fetched from a backend endpoint or S3 listing API
export const SAMPLE_VIDEOS = [
  {
    s3: "s3://shadow-trainer-prod/sample_input/henry-mini.mov",
    // Use .jpg extension for S3 static website or CloudFront public thumbnail
    thumb: "https://shadow-trainer-prod.s3.us-east-2.amazonaws.com/sample_input/henry.png",
    label: "Henry Mini"
  },
  {
    s3: "s3://shadow-trainer-prod/sample_input/adi.mov",
    thumb: "https://shadow-trainer-prod.s3.us-east-2.amazonaws.com/sample_input/adi.png",
    label: "adi"
  },
  {
    s3: "s3://shadow-trainer-prod/sample_input/cal-pitcher.mov",
    thumb: "https://shadow-trainer-prod.s3.us-east-2.amazonaws.com/sample_input/cal.png",
    label: "Cal Pitcher"
  }
  // Add more as needed
];
