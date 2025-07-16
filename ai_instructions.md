Update the frontend React app and backend API to switch from using /api routing under the same domain to a dedicated API subdomain https://api.shadow-trainer.com.
	1.	Update Frontend API Calls

	•	Go to the frontend directory at /home/ec2-user/shadow-trainer/api_frontend/shadow_trainer_web.
	•	Find all fetch, ajax, or HTTP calls that use relative paths like /api/....
	•	Replace those API endpoints with the new base URL https://api.shadow-trainer.com.
	•	Refactor the code to use an environment variable for the API base URL:
	•	Create or update a .env or .env.production file in the frontend root folder.
	•	Add the line:
REACT_APP_API_URL=https://api.shadow-trainer.com
	•	Change hardcoded URLs in the code to use this environment variable, for example:
fetch(${process.env.REACT_APP_API_URL}/some-endpoint)
	•	Ensure no fetch calls still use relative /api URLs.

	2.	Configure Backend CORS

	•	Go to the backend directory at /home/ec2-user/shadow-trainer/api_backend.
	•	Find the main app setup file where middleware is added.
	•	Add or update the CORS middleware to allow requests from your frontend domains:
	•	For FastAPI, add:
    ```
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://shadow-trainer.com", "https://www.shadow-trainer.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```
	•	If using another framework, add equivalent CORS configuration for the frontend domains.

	3.	Build and Deploy Frontend

	•	Run the frontend production build command (npm run build or yarn build).
	•	Copy the generated build folder to the directory served by NGINX, typically /var/www/frontend/build.
	•	Update any deployment scripts to reflect this path if needed.

	4.	Testing

	•	Verify that the React app loads correctly from https://shadow-trainer.com.
	•	Confirm API calls are sent to https://api.shadow-trainer.com.
	•	Check browser console for no CORS errors.

	5.	Cleanup

	•	Search the codebase for any remaining references to /api routing on the same domain and update or remove them.
	•	Confirm that both backend and frontend do not rely on the old routing structure.

Make sure to commit changes with clear messages for each step.