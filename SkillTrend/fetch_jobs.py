import os
import shutil
import re
import pandas as pd
from Ingestion.adzuna_client import fetch_jobs_from_adzuna
from Ingestion.utils import save_raw_data

try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except ImportError:
    # Fallback if NLTK not installed
    STEMMER = None


def get_user_input():
    """
    Get user input for job search parameters.

    Returns:
        dict: Contains cities, role, category, domain, and experience
    """
    print("\n" + "="*60)
    print("JOB FETCHER - SEARCH PARAMETERS")
    print("="*60)

    # Get cities
    cities_input = input("\nEnter cities (comma-separated, e.g., Bangalore,Mumbai,Delhi): ").strip()
    cities = [city.strip() for city in cities_input.split(",") if city.strip()]
    if not cities:
        cities = ["India"]
        print("Using default: India")

    # Get role/position
    role = input("\nEnter job role/position (e.g., Data Engineer, Python Developer, Data Scientist): ").strip()
    if not role:
        role = "data engineer"
        print("Using default: data engineer")

    # Get category/tag
    category = input("\nEnter job category/tag (e.g., IT, Finance, Healthcare, or leave blank): ").strip()

    # Get domain (optional)
    print("\nAvailable domains: Healthcare, Finance, IT, Sales, Marketing, HR, Analytics")
    domain = input("Enter domain (optional, e.g., Healthcare, Finance, or leave blank): ").strip()

    # Get experience
    experience_input = input("\nEnter minimum experience required (in years, e.g., 2, 5, or leave blank): ").strip()
    experience = None
    if experience_input:
        try:
            experience = int(experience_input)
        except ValueError:
            print("Invalid experience input. Will fetch all experience levels.")

    print("\n" + "="*60)
    print("SEARCH FILTERS:")
    print(f"  Cities: {', '.join(cities)}")
    print(f"  Role: {role}")
    print(f"  Category: {category if category else 'Any'}")
    print(f"  Domain: {domain if domain else 'Any'}")
    print(f"  Experience: {experience if experience else 'Any'} years")
    print("="*60 + "\n")

    return {
        "cities": cities,
        "role": role,
        "category": category,
        "domain": domain,
        "experience": experience
    }


def stem_words(words):
    """
    Stem a list of words using Porter Stemmer.

    Args:
        words (list): List of words to stem

    Returns:
        list: List of stemmed words
    """
    if not STEMMER:
        return words

    return [STEMMER.stem(word.lower()) for word in words]


def extract_keywords(text):
    """
    Extract keywords from text by splitting and cleaning.

    Args:
        text (str): Input text

    Returns:
        list: List of keywords
    """
    if not text:
        return []

    # Remove special characters and split
    keywords = re.findall(r'\b[a-z]+\b', text.lower())
    return keywords


def get_domain_keywords(domain):
    """
    Get relevant keywords for a specific domain.

    Args:
        domain (str): Domain name

    Returns:
        dict: Domain with associated keywords
    """
    domain_keywords = {
        "Healthcare": ["health", "medical", "doctor", "nurse", "hospital", "clinical", "pharma", "clinical", "physician"],
        "Finance": ["finance", "accounting", "banking", "investment", "audit", "treasury", "risk", "compliance", "trading"],
        "IT": ["software", "developer", "engineer", "programming", "devops", "cloud", "database", "infrastructure", "architect"],
        "Sales": ["sales", "account", "business", "development", "customer", "client", "revenue", "pipeline"],
        "Marketing": ["marketing", "digital", "brand", "content", "campaign", "seo", "advertising", "demand"],
        "HR": ["human", "recruitment", "talent", "people", "culture", "compensation", "employee", "relations"],
        "Analytics": ["analytics", "data", "business", "intelligence", "science", "machine", "learning", "statistical"],
    }

    return domain_keywords.get(domain, [domain.lower().split()])


def matches_role_with_domain(title, description, role, domain=None, category=None):
    """
    Check if job matches the role with specific domain/category keywords.
    Uses stemming to match related words (e.g., engineer, engineering, engineers).

    Args:
        title (str): Job title
        description (str): Job description
        role (str): Role to match (e.g., "Data Engineer", "Director of Architecture")
        domain (str): Domain filter (optional)
        category (str): Category filter (optional)

    Returns:
        bool: True if job matches role + domain/category combination
    """
    if not role:
        return True

    search_text = f"{title} {description}".lower()

    # Extract role keywords
    role_keywords = role.lower().split()

    # Get domain/category keywords
    domain_words = []
    if domain:
        domain_words.extend(get_domain_keywords(domain))
    if category:
        domain_words.extend(get_domain_keywords(category))

    # Stem all keywords
    stemmed_role = stem_words(role_keywords)
    stemmed_domain = stem_words(domain_words) if domain_words else []

    # Extract all words from search text
    text_keywords = extract_keywords(search_text)
    stemmed_text = stem_words(text_keywords)

    # Check if all role keywords are present
    role_match_count = 0
    for keyword in stemmed_role:
        if len(keyword) > 2:  # Skip very short words
            if keyword in stemmed_text:
                role_match_count += 1

    # Need to match at least 50% of role keywords or have strong keyword match
    min_role_matches = max(1, len(stemmed_role) // 2)

    if role_match_count < min_role_matches:
        return False

    # If domain/category specified, at least one domain keyword should match
    if stemmed_domain:
        domain_match = any(kw in stemmed_text for kw in stemmed_domain if len(kw) > 2)
        if not domain_match:
            return False

    return True


def cleanup_old_data(data_dir="data/raw", verbose=True):
    """
    Remove all old files from the data/raw folder.

    Args:
        data_dir (str): Directory to clean up
        verbose (bool): Print cleanup status
    """
    if os.path.exists(data_dir):
        try:
            files_removed = 0
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_removed += 1
            if verbose:
                print(f"Cleaned up {files_removed} old files from {data_dir}/")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    else:
        if verbose:
            print(f"Directory {data_dir}/ does not exist yet.")


def match_experience(description, min_experience):
    """
    Extract experience requirement from job description using regex.

    Args:
        description (str): Job description
        min_experience (int): Minimum experience required

    Returns:
        bool: True if job matches experience requirement or description not available
    """
    if not description or not min_experience:
        return True

    # Regex patterns to find experience mentions (e.g., "2 years", "5+ years", "3-5 years")
    patterns = [
        r'(\d+)\s*\+?\s*years',
        r'(\d+)\s*-\s*(\d+)\s*years',
        r'experience:\s*(\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    # Range of years (e.g., "3-5 years")
                    years = int(match[0])
                else:
                    # Single year (e.g., "5 years" or "5+ years")
                    years = int(match)

                if years >= min_experience:
                    return True
    return False


def match_category(title, description, category):
    """
    Check if job matches the specified category using regex.

    Args:
        title (str): Job title
        description (str): Job description
        category (str): Category/tag to match

    Returns:
        bool: True if job matches category
    """
    if not category:
        return True

    search_text = f"{title} {description}".lower()

    # Category mappings with regex patterns
    category_patterns = {
        "IT": r"(it|information technology|software|developer|engineer|programming|python|java|c\+\+|javascript|web|database|cloud)",
        "Finance": r"(finance|financial|accounting|accountant|audit|banking|investment|cfa|chartered)",
        "Healthcare": r"(healthcare|health|doctor|nurse|medical|physician|clinical|hospital|pharmaceutical)",
        "Sales": r"(sales|sales executive|account executive|business development|bde|sales manager)",
        "Marketing": r"(marketing|digital marketing|marketing executive|brand|content marketing|seo|sem)",
        "HR": r"(human resources|hr|recruiter|recruitment|talent acquisition|people operations)",
        "Analytics": r"(analytics|analyst|data analytics|business intelligence|bi|bi analyst|data analyst)",
    }

    # Get pattern for category or use the category itself as pattern
    pattern = category_patterns.get(category, category)

    return bool(re.search(pattern, search_text, re.IGNORECASE))


def match_domain(title, description, domain):
    """
    Check if job matches the specified domain using regex.

    Args:
        title (str): Job title
        description (str): Job description
        domain (str): Domain to match (Healthcare, Finance, IT, etc.)

    Returns:
        bool: True if job matches domain
    """
    if not domain:
        return True

    search_text = f"{title} {description}".lower()

    # Domain mappings with comprehensive regex patterns
    domain_patterns = {
        "Healthcare": r"(healthcare|health|doctor|nurse|medical|physician|clinical|hospital|pharmaceutical|nhs|doctor|therapist|physiotherapist|dentist|pharmacist)",
        "Finance": r"(finance|financial|accounting|accountant|audit|banking|investment|cfa|chartered|chartered accountant|ca|bfsi|fintech|insurance|mutual fund|stock|forex|derivatives)",
        "IT": r"(it|information technology|software|developer|engineer|programming|python|java|c\+\+|javascript|web|database|cloud|devops|aws|azure|gcp|docker|kubernetes)",
        "Sales": r"(sales|sales executive|account executive|business development|bde|sales manager|account manager|customer success|enterprise sales)",
        "Marketing": r"(marketing|digital marketing|marketing executive|brand|content marketing|seo|sem|social media|advertising|campaign|email marketing|product marketing)",
        "HR": r"(human resources|hr|recruiter|recruitment|talent acquisition|people operations|employee relations|compensation|benefits)",
        "Analytics": r"(analytics|analyst|data analytics|business intelligence|bi|bi analyst|data analyst|data science|machine learning|predictive|statistical)",
    }

    # Get pattern for domain or use the domain itself as pattern
    pattern = domain_patterns.get(domain, domain)

    return bool(re.search(pattern, search_text, re.IGNORECASE))


def match_role(title, description, role, domain=None, category=None):
    """
    Check if job matches the specified role using stemming and domain context.

    Args:
        title (str): Job title
        description (str): Job description
        role (str): Role/position to match
        domain (str): Domain context (optional)
        category (str): Category context (optional)

    Returns:
        bool: True if job matches role
    """
    if not role:
        return True

    return matches_role_with_domain(title, description, role, domain, category)



def normalize_jobs(raw_json, role=None, category=None, domain=None, experience=None):
    """
    Normalize job data and filter by role, category, domain, and experience.
    Uses smart keyword matching with stemming.

    Args:
        raw_json (dict): Raw API response
        role (str): Job role to filter by
        category (str): Job category to filter by
        domain (str): Job domain to filter by
        experience (int): Minimum experience required

    Returns:
        pd.DataFrame: Normalized job data
    """
    jobs = raw_json.get("results", [])

    normalized = []

    for job in jobs:
        title = job.get("title", "")
        description = job.get("description", "")

        # Apply filters
        if not match_role(title, description, role, domain=domain, category=category):
            continue
        if not match_category(title, description, category):
            continue
        if not match_domain(title, description, domain):
            continue
        if not match_experience(description, experience):
            continue

        normalized.append({
            "job_id": job.get("id"),
            "title": title,
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "description": description,
            "created": job.get("created"),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
            "redirect_url": job.get("redirect_url"),
        })

    return pd.DataFrame(normalized)


def run_ingestion(query="data engineer", cities=None, page=1, cleanup=True, role=None, category=None, domain=None, experience=None):
    """
    Fetch jobs for specified cities with advanced filtering.

    Args:
        query (str): Job title to search for
        cities (list): List of cities to fetch jobs from
        page (int): Page number for API results
        cleanup (bool): Remove old files from data/raw/ before fetching (default: True)
        role (str): Specific role/position to filter by
        category (str): Job category to filter by
        domain (str): Job domain to filter by (optional)
        experience (int): Minimum experience required (in years)
    """
    if cities is None:
        cities = ["India"]

    # Cleanup old data by default
    if cleanup:
        cleanup_old_data()

    print(f"Fetching {role or query} jobs in cities: {', '.join(cities)}")
    if category:
        print(f"Category filter: {category}")
    if domain:
        print(f"Domain filter: {domain}")
    if experience:
        print(f"Experience filter: {experience}+ years")

    all_dfs = []
    all_raw_jobs = []  # Accumulate all raw jobs from all cities
    total_before_filter = 0
    total_after_filter = 0

    for city in cities:
        print(f"\n  Fetching from {city}...")

        try:
            raw_data = fetch_jobs_from_adzuna(
                query=query,
                location=city,
                page=page
            )

            # Accumulate raw jobs for combined file
            raw_jobs = raw_data.get("results", [])
            for job in raw_jobs:
                job["city_fetched"] = city  # Add city info to raw data
            all_raw_jobs.extend(raw_jobs)

            # Normalize with filters
            df = normalize_jobs(raw_data, role=role, category=category, domain=domain, experience=experience)

            jobs_before = len(raw_jobs)
            jobs_after = len(df)

            total_before_filter += jobs_before
            total_after_filter += jobs_after

            if len(df) > 0:
                df['city'] = city  # Add city column
                all_dfs.append(df)
                print(f"  Jobs before filtering: {jobs_before} | After filtering: {jobs_after}")
            else:
                print(f"  No matching jobs found for {city} with applied filters")

        except Exception as e:
            print(f"  Error fetching from {city}: {str(e)}")
            continue

    if all_dfs:
        # Save combined raw data to single file
        combined_raw_file = f"data/raw/{query.replace(' ', '_')}_combined_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(combined_raw_file), exist_ok=True)

        import json
        with open(combined_raw_file, "w") as f:
            json.dump({"results": all_raw_jobs}, f, indent=2)
        print(f"\nCombined raw data saved at: {combined_raw_file}")

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save processed
        processed_file = "data/processed/jobs.csv"
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        combined_df.to_csv(processed_file, index=False)

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"Total jobs before filtering: {total_before_filter}")
        print(f"Total jobs after filtering: {total_after_filter}")
        print(f"Combined raw data: {combined_raw_file}")
        print(f"Processed data: {processed_file}")
        print(f"{'='*60}\n")
    else:
        print("\nNo jobs found for any of the specified cities with applied filters.")


if __name__ == "__main__":
    # Get user input for search parameters
    params = get_user_input()

    # Run ingestion with user-provided parameters
    run_ingestion(
        query=params["role"],
        cities=params["cities"],
        role=params["role"],
        category=params["category"],
        domain=params["domain"],
        experience=params["experience"]
    )