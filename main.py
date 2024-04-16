from fastapi import FastAPI
from fastapi.openapi.models import Info
import uvicorn
from fastapi.responses import JSONResponse, RedirectResponse
import psutil
from functions.logger import init_logging
from functions.ai import ModularAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
init_logging()

ai = ModularAI(credentials=os.getenv("GIGACHAT_API_KEY"))


@app.get(
    "/",
    response_class=JSONResponse,
    responses={"301": {"description": "Moved permanently"}},
)
async def read_root():
    """
    A function that reads the root endpoint.

    Parameters:
    - None

    Returns:
    - A RedirectResponse object with a status code of 301, redirecting to the "/docs" endpoint.
    """
    return RedirectResponse(url="/docs", status_code=301)


@app.get(
    "/info",
    responses={
        200: {
            "description": "Info retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "cpu_load": f"{psutil.cpu_percent()} %",
                        "memory_usage": f"{psutil.virtual_memory().percent} %",
                        "disk_usage": f"{psutil.disk_usage('/').percent} %",
                        "bytes_sent": psutil.net_io_counters().bytes_sent,
                        "bytes_received": psutil.net_io_counters().bytes_recv,
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error retrieving data."}}
            },
        },
    },
    response_class=JSONResponse,
)
async def server_info():
    """
    Retrieves information about the server.

    This function sends a GET request to the "/info" endpoint of the API and retrieves the server information. The server information includes the CPU load, memory usage, disk usage, bytes sent, and bytes received.

    Returns:
        A JSONResponse object containing the server information.

    Raises:
        HTTPException: If there is an internal server error.
    """
    try:
        return JSONResponse(
            content={
                "cpu_load": f"{psutil.cpu_percent()} %",
                "memory_usage": f"{psutil.virtual_memory().percent} %",
                "disk_usage": f"{psutil.disk_usage('/').percent} %",
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_received": psutil.net_io_counters().bytes_recv,
            },
            status_code=200,
        )
    except:
        return JSONResponse(
            content={"detail": "Error retrieving data."}, status_code=500
        )


@app.get(
    "/generate/{text}",
    responses={
        200: {
            "description": "A successful response",
            "content": {
                "application/json": {
                    "example": {"text": "Generate Python code for web request"}
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error"}}
            },
        },
    },
)
async def read_item(text: str):
    """
    When receiving a `GET` request on this root with the `text` parameter, the server tries to process the input data (simulating processing by a neural network) and generate the corresponding response. If the processing is successful, a JSON document with the processing result and status 200 is returned to the user.

    If the `ValueError` exception occurs, it is assumed that the user has entered incorrect data. Then the server will return an `HTTPException` exception with a status code of 400 and a detailed description of the error "Invalid input provided".

    If any other exception occurs, an `HTTPException` will be returned to the user with a status code of 500 and a detailed description of the error that occurred.

    ## Input data:

    `text` (str): A user-entered string intended for further processing.

    ## Output data:

    `JSONResponse` with the `text` key, which holds the output (e.g., neural network response or generated code).
    """
    try:
        # Simulate neural network processing and generating a response
        response = ai.ask(text)
        return JSONResponse(content={"text": response}, status_code=200)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # OpenAPI info
    openapi_info = Info(
        title="ProgNet API",
        version="1.0.0",
        description="API for the ProgNet language model",
    )

    # Running HTTP server
    uvicorn.run(app, host="127.0.0.1", port=8000, server_header=False)
