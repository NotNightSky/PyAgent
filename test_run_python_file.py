from functions.run_python_files import run_python_file

def test() -> None:
    print(run_python_files("calculator", "main.py"))
    print(run_python_files("calculator", "main.py", ["3 + 5"]))
    print(run_python_files("calculator", "tests.py"))
    print(run_python_files("calculator", "../main.py"))
    print(run_python_files("calculator", "nonexistent.py"))
    print(run_python_files("calculator", "lorem.txt"))


if __name__ == "__main__":
    test()