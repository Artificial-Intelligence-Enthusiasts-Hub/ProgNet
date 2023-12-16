from fastapi import FastAPI
from fastapi.openapi.models import Info
import uvicorn
from fastapi.responses import JSONResponse, RedirectResponse
import psutil
from functions.logger import init_logging

app = FastAPI()
init_logging()

@app.get("/", response_class=JSONResponse, responses={"301": {"description": "Moved permanently"}})
def read_root():
    """
    A function that reads the root endpoint.

    Parameters:
    - None

    Returns:
    - A RedirectResponse object with a status code of 301, redirecting to the "/docs" endpoint.
    """
    return RedirectResponse(url="/docs", status_code=301)

@app.get("/info",
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
                     "application/json": {
                         "example": {"detail": "Error retrieving data."}
                     }
                 },
             }
         }, response_class=JSONResponse)
def server_info():
    """
    Retrieves information about the server.
    
    This function sends a GET request to the "/info" endpoint of the API and retrieves the server information. The server information includes the CPU load, memory usage, disk usage, bytes sent, and bytes received.
    
    Returns:
        A JSONResponse object containing the server information.
        
    Raises:
        HTTPException: If there is an internal server error.
    """
    try:
        return JSONResponse(content={"cpu_load": f"{psutil.cpu_percent()} %", "memory_usage": f"{psutil.virtual_memory().percent} %", "disk_usage": f"{psutil.disk_usage('/').percent} %", "bytes_sent": psutil.net_io_counters().bytes_sent, "bytes_received": psutil.net_io_counters().bytes_recv}, status_code=200)
    except:
        return JSONResponse(content={"detail": "Error retrieving data."}, status_code=500)

if __name__ == "__main__":
    # OpenAPI info
    openapi_info = Info(
        title="ProgNet API",
        version="1.0.0",
        description="API for the ProgNet language model",
    )

    # Running HTTP server
    uvicorn.run(app, host="127.0.0.1", port=8000, server_header=False)
