"""
Main processing layer - Orchestrate skill extraction and analysis.
Supports both OLLAMA and OpenAI GPT models.
"""

import os
import sys
from dotenv import load_dotenv
from skill_extractor import SkillExtractor, SkillAnalyzer

# Load environment variables from .env file
load_dotenv()


def get_llm_config():
    """Get LLM configuration from user input."""
    print("\n" + "="*70)
    print("SKILL EXTRACTION - LLM CONFIGURATION")
    print("="*70)

    print("\nAvailable Models:")
    print("  1. Regex-based extraction (fast, no API needed)")
    print("  2. OLLAMA (local LLM, free)")
    print("  3. OpenAI GPT (API-based, requires API key)")

    choice = input("\nSelect extraction method (1-3, default: 1): ").strip() or "1"

    if choice == "2":
        print("\nOLLAMA Configuration:")
        print("  Make sure OLLAMA is running locally (http://localhost:11434)")

        model = input("Enter OLLAMA model name (default: mistral): ").strip() or "mistral"
        url = input("Enter OLLAMA URL (default: http://localhost:11434): ").strip() or "http://localhost:11434"

        return {
            "model_type": "ollama",
            "llm_model": model,
            "ollama_url": url
        }

    elif choice == "3":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️  OPENAI_API_KEY environment variable not set!")
            api_key = input("Enter your OpenAI API key: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key

        model = input("Enter GPT model name (default: gpt-4-turbo): ").strip() or "gpt-4-turbo"

        return {
            "model_type": "gpt",
            "llm_model": model,
            "ollama_url": None
        }

    else:
        print("Using Regex-based extraction (fast)")
        return {
            "model_type": "regex",
            "llm_model": None,
            "ollama_url": None
        }


def process_jobs_file(csv_file: str = "data/processed/jobs.csv"):
    """Process jobs file and extract skills."""

    if not os.path.exists(csv_file):
        print(f"❌ Jobs file not found: {csv_file}")
        print("Please run fetch_jobs.py first to generate processed jobs data.")
        sys.exit(1)

    # Get LLM configuration
    config = get_llm_config()

    print(f"\n🚀 Starting skill extraction ({config['model_type'].upper()})...")

    # Initialize extractor
    extractor = SkillExtractor(
        model_type=config["model_type"],
        llm_model=config.get("llm_model"),
        ollama_url=config.get("ollama_url")
    )

    # Initialize analyzer
    analyzer = SkillAnalyzer()

    # Load and process jobs
    analyzer.load_jobs(csv_file)

    # Categorize roles
    analyzer.categorize_roles()

    # Extract skills
    analyzer.extract_skills_from_jobs(extractor)

    # Print summary
    analyzer.print_summary()

    # Generate report
    report = analyzer.generate_report()

    print("\n✅ Skill extraction complete!")
    print(f"   Report saved to: data/reports/skill_analysis.json")

    return analyzer, report


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              SKILLTREND - SKILL EXTRACTION & ANALYSIS                ║
║                                                                      ║
║  Extracts and analyzes skills from job descriptions using:          ║
║  • Regex-based keyword matching                                      ║
║  • OLLAMA (local LLM models)                                         ║
║  • OpenAI GPT (API-based models)                                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    analyzer, report = process_jobs_file()
