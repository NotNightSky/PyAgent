import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import argparse

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
    name="run_python_file",
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

if len(response.function_calls) != 0:
    for call in response.function_calls:
        print(f"Calling function: {call.name}({call.args})")

print(f"Response: {response.text}")
