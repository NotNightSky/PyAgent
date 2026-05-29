import os

def get_files_info(working_directory: str, directory: str = ".") -> str:
    try:
        abs_working_dir = os.path.abspath(working_directory)
        target_dir = os.path.normpath(os.path.join(abs_working_dir, directory))

        if os.path.commonpath([abs_working_dir, target_dir]) != abs_working_dir:
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'

        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        entries = os.listdir(target_dir)
        lines = []
        for name in entries:
            item_path = os.path.join(target_dir, name)
            size = os.path.getsize(item_path)
            is_dir = os.path.isdir(item_path)
            lines.append(f"- {name}: file_size={size} bytes, is_dir={is_dir}")

        return "\n".join(lines) + "\n"

    except Exception as e:
        return f"Error: {e}"
