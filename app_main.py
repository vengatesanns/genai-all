import os
from dotenv import load_dotenv


if __name__ == "__main__":
    try:
        load_dotenv()
        print(os.getenv("LANGSMITH_PROJECT"))
    except Exception as e:
        print(f"Error loading .env file: {e}")


