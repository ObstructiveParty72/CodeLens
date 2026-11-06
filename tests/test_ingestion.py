from ingestion.github_loader import GitHubLoader
from ingestion.repository import RepositoryScanner


REPOSITORY_URL = "https://github.com/psf/requests.git"

def main():
    loader = GitHubLoader()
    repository_path = loader.clone_repository(REPOSITORY_URL)
    scanner = RepositoryScanner()
    files = scanner.scan(repository_path)
    print(f"\nTotal supported files: {len(files)}\n")
    for file in files[:20]:
        print(file)

if __name__ == "__main__":
    main()