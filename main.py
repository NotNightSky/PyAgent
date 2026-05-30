import os
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_python_files import run_python_files
from functions.write_file import write_file
from dotenv import load_dotenv
from google import genai
from google.genai import types
from collections.abc import Callable
import argparse

def call_function(function_call: types.FunctionCall, verbose: bool = False) -> types.Content:

    function_name = function_call.name or ""

    function_map: dict[str, Callable[..., str]] = {
        "get_file_content": get_file_content,
        "get_files_info": get_files_info,
        "run_python_files": run_python_files,
        "write_file": write_file
    }

    if function_name not in function_map:
        return types.Content(role="tool", parts=[types.Part.from_function_response(name=function_name,response={"error": f"Unknown function: {function_name}"},)],)

    args = dict(function_call.args) if function_call.args else {}

    args["working_directory"] = "./calculator"

    function_result = function_map[function_name](**args)

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
        ],
    )

system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""

load_dotenv()
try:
    api_key = os.environ.get("GEMINI_API_KEY")
except RuntimeError:
    print("API kye not found or is inaccessable")

schema_get_files_info = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in a specified directory relative to the working directory, providing file size and directory status",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description="Directory path to list files from, relative to the working directory (default is the working directory itself)",
            ),
        },
    ),
)
 
schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description="Reads and returns the contents of a file relative to the working directory",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file to read, relative to the working directory",
            ),
        },
        required=["file_path"],
    ),
)
 
schema_run_python_file = types.FunctionDeclaration(
    name="run_python_files",
    description="Executes a Python file relative to the working directory and returns its output",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the Python file to run, relative to the working directory",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Optional list of arguments to pass to the script",
            ),
        },
        required=["file_path"],
    ),
)
 
schema_write_file = types.FunctionDeclaration(
    name="write_file",
    description="Writes content to a file at the specified path relative to the working directory, creating it or overwriting it if it exists",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file to write, relative to the working directory",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="The content to write to the file",
            ),
        },
        required=["file_path", "content"],
    ),
)
 
available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_get_file_content,
        schema_run_python_file,
        schema_write_file,
    ]
)


client = genai.Client(api_key=api_key)

parser = argparse.ArgumentParser(description="CLI chatbot")
parser.add_argument("user_prompt", type=str, help="User prompt")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

history: list[types.Content] = [
    types.Content(role="user", parts=[types.Part(text=args.user_prompt)])
]

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=history, config=types.GenerateContentConfig(
        tools=[available_functions], system_instruction=system_prompt
    )
)

if args.verbose:
    print(f"User prompt: {args.user_prompt}")
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

function_results = []
if response.function_calls:
    for function_call in response.function_calls:

        function_call_result = call_function(function_call, verbose=args.verbose)

        if not function_call_result.parts:
            raise ValueError("Function call returned no parts")
        if function_call_result.parts[0].function_response is None:
            raise ValueError("Function call returned no function_response")
        if function_call_result.parts[0].function_response.response is None:
            raise ValueError("Function call returned no response data")

        function_results.append(function_call_result.parts[0])

        if args.verbose:
            print(f"-> {function_call_result.parts[0].function_response.response}")

print(f"Response: {response.text}")
