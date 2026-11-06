import os
import shutil
import stat
from pathlib import Path

from git import Repo

class GitHubLoader:

    def __init__(self,base_path="data/repositories"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True,exist_ok=True)

    def _remove_readonly(self,func,path,exc_info):
        """
        Handle read-only files on Windows.
        """
        os.chmod(path,stat.S_IWRITE)
        func(path)

    def _get_available_path(self,repo_name):
        """
        Find an available directory for the repository.
        Example:
        requests
        requests_1
        requests_2
        ...
        """

        base_path = self.base_path / repo_name
        if not base_path.exists():
            return base_path

        counter = 1

        while True:
            new_path = (self.base_path/ f"{repo_name}_{counter}")
            if not new_path.exists():
                return new_path

            counter += 1

    def clone_repository(self,repo_url: str):
        repo_name = (repo_url.rstrip("/").split("/")[-1])

        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        repo_path = self.base_path / repo_name

        # ----------------------------------------------------
        # Try to remove the old repository
        # ----------------------------------------------------

        if repo_path.exists():
            print(f"Existing repository found: "f"{repo_path}")
            try:
                shutil.rmtree(repo_path,onerror=self._remove_readonly)
                print("Old repository removed.")
            except PermissionError:
                print(
                    "Could not remove existing repository because Windows has locked a file."
                )

                repo_path = self._get_available_path(repo_name)

                print(f"Using alternate path: "f"{repo_path}")

        # ----------------------------------------------------
        # Clone repository
        # ----------------------------------------------------
        print(f"Cloning repository "f"{repo_url} to {repo_path}")
        Repo.clone_from(repo_url,repo_path)
        print(f"Repository cloned to: "f"{repo_path}")

        return str(repo_path)