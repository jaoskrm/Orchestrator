import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str

class DockerSandbox:
    def __init__(self, container_name: str = "agent-sandbox"):
        self.container = container_name

    def reset_task_dir(self) -> None:
        subprocess.run(
            ["docker", "exec", "-u", "root", self.container, "sh", "-lc", "rm -rf /work/task && mkdir -p /work/task"],
            check=True
        )

    def copy_task_in(self, task_dir: Path) -> None:
        # Copy the whole task dir to /work/task_tmp, then copy its CONTENTS into /work/task
        subprocess.run(["docker", "cp", str(task_dir), f"{self.container}:/work/task_tmp"], check=True)
        subprocess.run(
            ["docker", "exec", "-u", "root", self.container, "sh", "-lc",
             "cp -r /work/task_tmp/. /work/task/ && rm -rf /work/task_tmp"],
            check=True
        )

    def pytest(self, *files: str) -> RunResult:
        """
        Run pytest in the container's /work/task directory.
        
        Args:
            *files: Optional specific files to test (e.g., "main.py", "test_*.py").
                   If not provided, pytest uses its default discovery.
        
        Returns:
            RunResult with exit code, stdout, and stderr
        """
        # Build the pytest command
        pytest_cmd = "cd /work/task && python -m pytest -q -p no:cacheprovider"
        if files:
            # Add specific files to the command
            pytest_cmd += " " + " ".join(files)
        
        p = subprocess.run(
            ["docker", "exec", "-u", "sandbox", self.container, "sh", "-lc", pytest_cmd],
            capture_output=True, text=True
        )
        return RunResult(p.returncode, p.stdout, p.stderr)