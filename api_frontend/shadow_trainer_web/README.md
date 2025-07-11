# Shadow Trainer Web

A modern React web app for Shadow Trainer: AI-Powered Motion Analysis.

## Features
- Upload or paste an S3 path to your video
- "Create Your Shadow" button triggers backend API
- Shows API response and video preview or S3 URL
- Clean, professional, and responsive design

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm start
   ```
   The app will run at http://localhost:3000

3. To build for production:
   ```bash
   npm run build
   ```

## API Endpoint
- The app expects the backend API to be available at `/api/video/process` (reverse-proxied by Nginx or similar).

## Assets
- Place your logo at `public/assets/Shadow Trainer Logo.png`.
