from ingestion.parser import CodeParser
from ingestion.extractor import CodeExtractor

code = """
import redis
from database import get_user

class AuthService:
    def authenticate(self, token):
        user = get_user(token)
        return user
"""

parser = CodeParser()
tree = parser.parse(code, "python")

extractor = CodeExtractor()
results = extractor.extract(tree,code)
for result in results:
    print(result)