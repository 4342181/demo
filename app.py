from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline import load_base_kb, process_uploaded_file, run_rag_query
import uvicorn
import os
import bcrypt # For password hashing
from dotenv import load_dotenv # Import load_dotenv for .env file support
from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# --- Load environment variables from .env file ---
# This must be called at the very beginning to load variables into os.environ
load_dotenv()

app = FastAPI()

# --- CORS Middleware ---
# This allows your React frontend (running on a different port/origin) to
# make requests to your FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, specify your frontend's origin(s).
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Azure Table Storage Configuration ---
# Retrieve connection string and table name from environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME", "Users") # Default to "Users" if not set in .env

# Validate that the connection string is set
if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError(
        "AZURE_STORAGE_CONNECTION_STRING environment variable is not set. "
        "Please ensure it's in your .env file or system environment variables."
    )

# Initialize Azure Table Service Client and get table client for users
try:
    table_service_client = TableServiceClient.from_connection_string(conn_str=AZURE_STORAGE_CONNECTION_STRING)
    users_table_client = table_service_client.get_table_client(table_name=TABLE_NAME)

    # Attempt to create the table. If it already exists, create_table() will
    # raise an HttpResponseError with status code 409 (Conflict), which we can safely ignore.
    users_table_client.create_table()
    print(f"Azure Table '{TABLE_NAME}' created or already exists.")
except HttpResponseError as e:
    if e.response.status_code == 409:
        print(f"Azure Table '{TABLE_NAME}' already exists.")
    else:
        # Re-raise other HTTP errors to prevent the app from starting without table access
        print(f"Failed to connect to Azure Table Storage or create table: {e}")
        raise
except Exception as e:
    # Catch any other unexpected errors during initialization
    print(f"An unexpected error occurred during Azure Table Storage initialization: {e}")
    raise

# --- User Entity Model for Azure Table Storage ---
# This class defines the structure of a user entity stored in the Azure Table.
# PartitionKey and RowKey are mandatory for Azure Table Storage entities.
class UserEntity:
    def __init__(self, username: str, hashed_password: str):
        # PartitionKey helps in organizing data and improves query performance.
        # Using a static 'user' PartitionKey means all users are in one partition,
        # which is fine for small/dummy setups.
        self.PartitionKey = "user"
        # RowKey uniquely identifies an entity within its PartitionKey.
        # Using lowercase username ensures consistency and uniqueness.
        self.RowKey = username.lower()
        self.username = username # Storing username redundantly but for clarity/ease of access
        self.hashed_password = hashed_password # This will store the bcrypt hash

# --- Load base Knowledge Base for RAG pipeline at application startup ---
# This remains the same as your original app.py
base_kb = load_base_kb()

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "API is running. Visit /docs to test the endpoint."}

# --- Authentication Endpoints ---

@app.post("/register-dummy-user")
async def register_dummy_user(username: str = Form(...), password: str = Form(...)):
    """
    Registers a new dummy user by hashing the password and storing it
    in Azure Table Storage.
    """
    # Hash the plaintext password using bcrypt.
    # bcrypt.gensalt() generates a random salt.
    # .decode('utf-8') converts the bytes hash back to a string for storage.
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Create a UserEntity object. Its __dict__ representation will be stored.
    user_entity = UserEntity(username=username, hashed_password=hashed_password)

    try:
        # Attempt to create the entity in Azure Table Storage.
        # If a user with the same RowKey (username) already exists,
        # Azure will return a 409 Conflict error.
        users_table_client.create_entity(entity=user_entity.__dict__)
        return JSONResponse({"message": "Dummy user registered successfully."})
    except HttpResponseError as e:
        if e.response.status_code == 409:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists. Please choose a different one."
            )
        # Re-raise other Azure-specific HTTP errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user due to Azure error: {e.message}"
        )
    except Exception as e:
        # Catch any other unexpected errors during registration
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during user registration: {str(e)}"
        )

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Authenticates a user by retrieving the hashed password from Azure Table Storage
    and verifying it against the provided plaintext password.
    """
    try:
        # Retrieve the user entity from Azure Table Storage using PartitionKey and RowKey.
        # The RowKey is the lowercase version of the username.
        user_entity = users_table_client.get_entity(
            partition_key="user",
            row_key=username.lower()
        )

        # Verify the provided plaintext password against the stored bcrypt hash.
        # Both passwords must be encoded to bytes for bcrypt.checkpw().
        if bcrypt.checkpw(password.encode('utf-8'), user_entity['hashed_password'].encode('utf-8')):
            # For a dummy login, we return a simple success message and a placeholder token.
            # In a real production application, you would generate a secure JWT (JSON Web Token)
            # here, containing user-specific claims, and possibly manage sessions.
            return JSONResponse({
                "message": "Login successful!",
                "access_token": "dummy-jwt-token-for-client", # Placeholder token
                "token_type": "bearer"
            })
        else:
            # If passwords don't match
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password."
            )
    except ResourceNotFoundError:
        # This error is raised if the user with the given PartitionKey and RowKey is not found.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )
    except Exception as e:
        # Catch any other unexpected errors during the login process.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during login: {str(e)}"
        )

# --- Document Analysis Endpoint (Your existing RAG pipeline) ---
# This endpoint remains unauthenticated for this dummy setup.
# If you wanted to protect it, you would add a dependency like:
# async def analyze(file: UploadFile = File(...), query: str = Form(...),
#                  current_user: User = Depends(get_current_active_user)):
# where get_current_active_user would validate the access_token.
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), query: str = Form(...)):
    """
    Processes an uploaded document and answers a query using the RAG pipeline.
    """
    try:
        # Read the content of the uploaded file
        file_bytes = await file.read()

        # Process the newly uploaded document to create a temporary knowledge base
        temp_kb = process_uploaded_file(file_bytes)

        # Combine the base knowledge base with the temporary one from the uploaded file
        merged_kb = base_kb + temp_kb

        # Run the RAG query against the merged knowledge base
        response = run_rag_query(merged_kb, query)

        return JSONResponse({"response": response})
    except Exception as e:
        # Catch any errors during the file processing or RAG query
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Something went wrong during analysis: {str(e)}"}
        )

# --- Main entry point for Uvicorn server ---
if __name__ == "__main__":
    # Runs the FastAPI application using Uvicorn.
    # host="0.0.0.0" makes the server accessible from other devices on the network.
    # port=8000 is the standard port for local development.
    # --reload enables hot-reloading for development, restarting the server on code changes.
    uvicorn.run(app, host="0.0.0.0", port=8000)
