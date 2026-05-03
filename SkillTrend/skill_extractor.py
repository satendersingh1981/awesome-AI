"""
Skill Extraction Layer - Extract and analyze skills from job descriptions.
Supports both Regex-based and LLM-based extraction using OLLAMA and OpenAI GPT.
"""

import os
import json
import re
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional

# Try importing LLM libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ==================== LLM COMPATIBILITY LAYER ====================

class LLMCompatibility:
    """Handle compatibility across different LLM models."""
    
    # Model families and their characteristics
    OPENAI_MODELS = [
        'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-5', 'gpt-5-nano',
        'gpt-3.5', 'gpt-3.5-turbo', 'gpt-4-turbo-preview', 'gpt-4o-mini'
    ]
    
    NEWER_OPENAI_MODELS = ['gpt-5', 'gpt-5-nano', 'gpt-4o', 'gpt-4o-mini']
    
    @staticmethod
    def detect_model_type(model_name: str) -> str:
        """Detect LLM model type."""
        model_lower = model_name.lower()
        if 'gpt-' in model_lower:
            return 'openai'
        elif 'claude' in model_lower:
            return 'claude'
        elif 'llama' in model_lower or 'mistral' in model_lower:
            return 'ollama'
        else:
            return 'generic'
    
    @staticmethod
    def is_newer_openai_model(model_name: str) -> bool:
        """Check if it's a newer OpenAI model that uses max_completion_tokens."""
        model_lower = model_name.lower()
        # gpt-5, gpt-4o, and newer models use max_completion_tokens
        return any(new_model in model_lower for new_model in LLMCompatibility.NEWER_OPENAI_MODELS)
    
    @staticmethod
    def supports_temperature(model_name: str) -> bool:
        """Check if model supports custom temperature parameter."""
        model_lower = model_name.lower()
        # gpt-5-nano doesn't support temperature
        if 'gpt-5-nano' in model_lower:
            return False
        return True
    
    @staticmethod
    def get_token_param_name(model_name: str) -> str:
        """Get the correct token parameter name for the model."""
        if LLMCompatibility.is_newer_openai_model(model_name):
            return 'max_completion_tokens'
        else:
            return 'max_tokens'
    
    @staticmethod
    def build_openai_params(model_name: str, temperature: float = 0.3, 
                           max_tokens: int = 500) -> Dict:
        """Build compatible OpenAI API parameters for any model."""
        params = {
            'model': model_name,
            'temperature': temperature if LLMCompatibility.supports_temperature(model_name) else None,
            LLMCompatibility.get_token_param_name(model_name): max_tokens
        }
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}


# ==================== SKILL KEYWORDS DATABASE ====================

SKILL_KEYWORDS = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "php", "ruby", "scala", "kotlin", "swift", "r programming", "matlab",
        "sql", "plsql", "nodejs", "node.js"
    ],
    "Web Frameworks": [
        "django", "flask", "fastapi", "spring", "springboot", "spring boot",
        "react", "angular", "vue", "nextjs", "next.js", "express", "asp.net",
        "rails", "laravel", "gatsby"
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "cassandra", "dynamodb", "elasticsearch",
        "redis", "memcached", "oracle", "sql server", "mariadb", "nosql",
        "firebase", "couchdb", "neo4j","snowflake"," pinecone", "milvus", "qdrant", "weaviate","pgvector"
    ],
    "Cloud Platforms": [
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
        "alibaba cloud", "ibm cloud", "kubernetes", "docker", "terraform"
    ],
    "DevOps & Tools": [
        "docker", "kubernetes", "jenkins", "gitlab", "github", "git", "terraform",
        "ansible", "helm", "prometheus", "grafana", "ci/cd", "devops",
        "nginx", "apache", "linux", "bash", "shell"
    ],
    "Data & Analytics": [
        "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "keras", "spark", "hadoop", "airflow", "kafka", "data warehousing",
        "etl", "dbt", "lookahead", "tableau", "power bi", "analytics"
    ],
    "AI & Machine Learning": [
        "machine learning", "deep learning", "neural networks", "nlp",
        "computer vision", "pytorch", "tensorflow", "keras", "scikit-learn",
        "hugging face", "langchain", "llm", "generative ai", "ai", "ml",
        "model training", "prediction", "classification", "regression"
    ],
    "Soft Skills": [
        "communication", "leadership", "project management", "teamwork",
        "problem solving", "critical thinking", "agile", "scrum", "kanban",
        "documentation", "mentoring", "analytical", "strategic thinking"
    ]
}

# Role Categories
ROLE_CATEGORIES = {
    "Data Engineering": [
        "data engineer", "etl", "data pipeline", "data infrastructure",
        "data architecture", "analytics engineer"
    ],
    "Data Science": [
        "data scientist", "machine learning engineer", "ml engineer",
        "ai engineer", "deep learning", "research scientist"
    ],
    "Backend Engineering": [
        "backend engineer", "backend developer", "server-side", "api developer",
        "full stack", "software engineer"
    ],
    "Frontend Engineering": [
        "frontend engineer", "frontend developer", "ui developer", "ux engineer",
        "web developer", "react developer", "angular developer"
    ],
    "DevOps & Cloud": [
        "devops engineer", "cloud engineer", "infrastructure engineer",
        "site reliability engineer", "sre", "platform engineer"
    ],
    "Data Analytics": [
        "data analyst", "business analyst", "analytics engineer",
        "business intelligence", "bi analyst"
    ],
    "Architecture & Leadership": [
        "architect", "director", "engineering manager", "tech lead",
        "principal engineer", "head of engineering"
    ]
}

# Skill Type Categories - Map skills to broad categories
SKILL_TYPE_CATEGORIES = {
    "Technical": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby",
        "django", "flask", "fastapi", "spring", "react", "angular", "vue", "express",
        "mysql", "postgresql", "mongodb", "cassandra", "redis", "elasticsearch",
        "sql", "html", "css", "api", "rest", "graphql"
    ],
    "Data Science & ML": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "nlp",
        "computer vision", "pandas", "numpy", "scikit-learn", "data science", "ai",
        "ml", "neural networks", "classification", "regression", "prediction",
        "hugging face", "langchain", "llm", "generative ai"
    ],
    "DevOps & Cloud": [
        "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "gitlab", "github",
        "terraform", "ansible", "ci/cd", "devops", "cloud", "infrastructure",
        "nginx", "apache", "linux", "bash", "shell", "prometheus", "grafana"
    ],
    "Data Engineering": [
        "data engineer", "etl", "data pipeline", "spark", "hadoop", "airflow",
        "kafka", "data warehousing", "dbt", "tableau", "power bi", "analytics",
        "snowflake", "redshift", "bigquery"
    ],
    "Soft Skills": [
        "communication", "leadership", "project management", "teamwork",
        "problem solving", "critical thinking", "agile", "scrum", "kanban",
        "documentation", "mentoring", "analytical", "strategic thinking"
    ],
    "Architectural": [
        "microservices", "architecture", "design pattern", "system design",
        "scalability", "performance", "optimization", "high availability"
    ]
}


class SkillExtractor:
    """Extract skills from job descriptions using Regex or LLM."""

    def __init__(self, model_type: str = "regex", llm_model: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize skill extractor with automatic model compatibility detection.

        Args:
            model_type: "regex", "ollama", or "gpt"
            llm_model: Model name (e.g., "mistral" for OLLAMA, "gpt-4" for OpenAI)
            ollama_url: OLLAMA server URL
        """
        self.model_type = model_type.lower()
        self.llm_model = llm_model
        self.ollama_url = ollama_url

        if self.model_type == "gpt":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library required for GPT models. Install: pip install openai")
            
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Auto-detect model type and warn about compatibility if needed
            if self.llm_model:
                model_type_detected = LLMCompatibility.detect_model_type(self.llm_model)
                is_newer = LLMCompatibility.is_newer_openai_model(self.llm_model)
                supports_temp = LLMCompatibility.supports_temperature(self.llm_model)
                
                if is_newer:
                    print(f"✓ Using newer model '{self.llm_model}' - automatically using max_completion_tokens")
                if not supports_temp:
                    print(f"✓ Model '{self.llm_model}' doesn't support custom temperature - using default")
        
        elif self.model_type == "ollama":
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests library required for OLLAMA. Install: pip install requests")
            print(f"ℹ️ Using OLLAMA model '{self.llm_model or 'mistral'}' at {ollama_url}")
        
        elif self.model_type == "regex":
            print(f"ℹ️ Using Regex-based extraction (no LLM required)")

    def extract_skills_regex(self, text: str) -> List[str]:
        """Extract skills using regex and keyword matching."""
        if not text:
            return []

        text_lower = text.lower()
        found_skills = set()

        for category, skills in SKILL_KEYWORDS.items():
            for skill in skills:
                # Case-insensitive whole word match
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(skill)

        return sorted(list(found_skills))

    def extract_skills_ollama(self, text: str) -> List[str]:
        """Extract skills using OLLAMA LLM with robust error handling."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install: pip install requests")

        model_name = self.llm_model or "mistral"
        
        prompt = f"""Extract ONLY technical skills and technologies from this job description.
Return as a JSON list of skills.

Job Description:
{text}

Return format: ["skill1", "skill2", "skill3"]
Only return the JSON list, nothing else."""

        try:
            # Build request payload with smart defaults
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add temperature only for models that support it
            # Most open source models accept temperature, but we add a fallback
            model_lower = model_name.lower()
            if 'deepseek' not in model_lower:  # deepseek may have issues with temperature
                payload["temperature"] = 0.3
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")

                # Try JSON parsing first
                json_match = re.search(r'\[.*?\]', text, re.DOTALL)
                if json_match:
                    try:
                        skills = json.loads(json_match.group())
                        return sorted([s.strip().lower() for s in skills if s.strip()])
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: parse as comma-separated list
                if ',' in text or text.strip():
                    skills = [s.strip().lower() for s in text.split(',') if s.strip()]
                    return sorted(list(set(skills)))

            return []
        except requests.exceptions.ConnectionError:
            print(f"⚠️ OLLAMA Error: Cannot connect to {self.ollama_url}")
            print(f"   Make sure OLLAMA is running: ollama serve")
            return []
        except requests.exceptions.Timeout:
            print(f"⚠️ OLLAMA Error: Request timeout. Model may be processing slowly.")
            return []
        except Exception as e:
            print(f"OLLAMA error: {type(e).__name__}: {e}")
            return []

    def extract_skills_gpt(self, text: str) -> List[str]:
        """Extract skills using OpenAI GPT with compatibility layer."""
        try:
            model_name = self.llm_model or "gpt-4"
            
            # Build compatible parameters for any model
            api_params = LLMCompatibility.build_openai_params(
                model_name=model_name,
                temperature=0.3,
                max_tokens=500
            )
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": "Extract ONLY technical skills, technologies, and tools from job descriptions. Return as a comma-separated list of skills. Be specific and include frameworks, languages, tools, platforms, and methodologies."
                },
                {
                    "role": "user",
                    "content": f"Extract all technical skills from this job description:\n\n{text[:2000]}\n\nReturn ONLY a comma-separated list of skills, nothing else."
                }
            ]
            
            # Make API call with compatible parameters
            response = self.client.chat.completions.create(
                messages=messages,
                **api_params
            )

            response_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON list first
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                try:
                    skills = json.loads(json_match.group())
                    return sorted([s.strip().lower() for s in skills if s.strip()])
                except json.JSONDecodeError:
                    pass
            
            # If not JSON, parse as comma-separated list
            if ',' in response_text or response_text:
                skills = [s.strip().lower() for s in response_text.split(',') if s.strip()]
                return sorted(list(set(skills)))  # Remove duplicates

            return []
        except Exception as e:
            print(f"GPT error: {e}")
            return []

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using configured method."""
        if self.model_type == "ollama":
            return self.extract_skills_ollama(text)
        elif self.model_type == "gpt":
            return self.extract_skills_gpt(text)
        else:
            return self.extract_skills_regex(text)


class SkillAnalyzer:
    """Analyze extracted skills and generate insights."""

    def __init__(self):
        self.skill_counts = Counter()
        self.role_skills = {}
        self.jobs_data = []

    def load_jobs(self, csv_file: str):
        """Load processed jobs from CSV."""
        self.jobs_data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.jobs_data)} jobs from {csv_file}")

    def extract_skills_from_jobs(self, extractor: SkillExtractor):
        """Extract skills from all job descriptions."""
        print(f"Extracting skills using {extractor.model_type.upper()}...")
        self.jobs_data['skills'] = self.jobs_data['description'].apply(
            lambda x: extractor.extract_skills(x) if isinstance(x, str) else []
        )

        # Update skill counts
        for skills in self.jobs_data['skills']:
            self.skill_counts.update(skills)

        print(f"Found {len(self.skill_counts)} unique skills")

    def categorize_roles(self):
        """Categorize jobs into role groups."""
        def get_role_category(title):
            title_lower = title.lower() if isinstance(title, str) else ""
            for category, keywords in ROLE_CATEGORIES.items():
                for keyword in keywords:
                    if keyword in title_lower:
                        return category
            return "Other"

        self.jobs_data['role_category'] = self.jobs_data['title'].apply(get_role_category)

    def get_skills_by_type(self) -> Dict[str, List[tuple]]:
        """Categorize skills by type and return with counts."""
        skills_by_type = {}
        
        for skill_type, keywords in SKILL_TYPE_CATEGORIES.items():
            type_skills = Counter()
            
            for skills in self.jobs_data['skills']:
                for skill in skills:
                    # Check if skill matches any keyword in this type
                    if any(keyword.lower() in skill.lower() for keyword in keywords):
                        type_skills[skill] += 1
            
            # Sort by count
            skills_by_type[skill_type] = type_skills.most_common()
        
        return skills_by_type

    def calculate_weight_score(self, count: int, max_count: int) -> float:
        """Calculate weight score (1-10) based on skill frequency."""
        if max_count == 0:
            return 0.0
        percentage = (count / max_count) * 100
        # Scale 0-100% to 1-10 score
        score = (percentage / 10)
        return min(10.0, max(1.0, score))

    def get_trend_emoji(self, trend_value: float) -> str:
        """Get emoji and trend label based on trend percentage."""
        if trend_value > 20:
            return "🚀 Fast Rising"
        elif trend_value > 10:
            return "🔥 Rising"
        elif trend_value > 0:
            return "📈 Stable"
        elif trend_value > -10:
            return "📉 Declining"
        else:
            return "⚠️ Falling"

    def get_top_skills(self, n: int = 20) -> List[tuple]:
        """Get top N skills by frequency."""
        return self.skill_counts.most_common(n)

    def get_role_skills(self, n: int = 10) -> Dict[str, List[tuple]]:
        """Get top skills for each role category."""
        role_skills = {}

        for category in self.jobs_data['role_category'].unique():
            role_jobs = self.jobs_data[self.jobs_data['role_category'] == category]
            role_skill_counter = Counter()

            for skills in role_jobs['skills']:
                role_skill_counter.update(skills)

            role_skills[category] = role_skill_counter.most_common(n)

        return role_skills

    def get_skill_trends(self) -> Dict[str, float]:
        """
        Calculate skill trends (percentage change).
        Compares recent jobs vs earlier jobs.
        """
        if len(self.jobs_data) < 2:
            return {}

        mid_point = len(self.jobs_data) // 2
        earlier_jobs = self.jobs_data[:mid_point]
        recent_jobs = self.jobs_data[mid_point:]

        earlier_skills = Counter()
        recent_skills = Counter()

        for skills in earlier_jobs['skills']:
            earlier_skills.update(skills)

        for skills in recent_jobs['skills']:
            recent_skills.update(skills)

        trends = {}
        all_skills = set(earlier_skills.keys()) | set(recent_skills.keys())

        for skill in all_skills:
            earlier_count = earlier_skills.get(skill, 0) or 1
            recent_count = recent_skills.get(skill, 0)

            # Calculate percentage change
            pct_change = ((recent_count - earlier_count) / earlier_count) * 100
            trends[skill] = pct_change

        # Sort by trend
        return dict(sorted(trends.items(), key=lambda x: x[1], reverse=True))

    def generate_report(self, output_file: str = "data/reports/skill_analysis.json"):
        """Generate comprehensive skill analysis report organized by skill type."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Get skills organized by type
        skills_by_type = self.get_skills_by_type()
        
        # Convert to report format with weights
        skills_by_type_report = {}
        total_skill_instances = sum(count for skill_count in skills_by_type.values() 
                                    for _, count in skill_count)
        
        for skill_type, skills in skills_by_type.items():
            type_total = sum(count for _, count in skills)
            skills_by_type_report[skill_type] = {
                "total_count": type_total,
                "percentage": round((type_total / total_skill_instances * 100) if total_skill_instances > 0 else 0, 2),
                "skills": [
                    {
                        "name": skill,
                        "count": count,
                        "weight": round((count / type_total * 100) if type_total > 0 else 0, 2)
                    }
                    for skill, count in skills[:20]  # Top 20 skills per type
                ]
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": len(self.jobs_data),
            "total_unique_skills": len(self.skill_counts),
            "summary": {
                "total_jobs_analyzed": len(self.jobs_data),
                "unique_skills_found": len(self.skill_counts),
                "average_skills_per_job": round(sum(len(s) for s in self.jobs_data['skills']) / len(self.jobs_data) if len(self.jobs_data) > 0 else 0, 2)
            },
            "skills_by_type": skills_by_type_report,
            "skills_by_role": {
                role: {
                    "total": sum(count for _, count in skills),
                    "skills": [
                        {"name": skill, "count": count}
                        for skill, count in skills[:15]
                    ]
                }
                for role, skills in self.get_role_skills(15).items()
            },
            "top_skills_overall": [
                {"name": skill, "count": count, "percentage": round((count / len(self.jobs_data) * 100), 2)}
                for skill, count in self.get_top_skills(20)
            ],
            "skill_trends": {
                skill: round(trend, 2)
                for skill, trend in self.get_skill_trends().items()
                if abs(trend) > 5  # Show only significant trends
            },
            "role_distribution": self.jobs_data['role_category'].value_counts().to_dict(),
            "cities": self.jobs_data['city'].value_counts().to_dict() if 'city' in self.jobs_data.columns else {}
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {output_file}")
        return report

    def print_summary(self):
        """Print summary of skill analysis with tabular format."""
        print("\n" + "="*130)
        print("SKILL ANALYSIS REPORT - COMPREHENSIVE BREAKDOWN")
        print("="*130)

        print(f"\nTotal Jobs Analyzed: {len(self.jobs_data)}")
        print(f"Unique Skills Found: {len(self.skill_counts)}")
        avg_skills = sum(len(s) for s in self.jobs_data['skills']) / len(self.jobs_data) if len(self.jobs_data) > 0 else 0
        print(f"Average Skills per Job: {avg_skills:.1f}")

        # Get top skills and trends
        top_skills = self.get_top_skills(15)
        trends = self.get_skill_trends()
        max_count = top_skills[0][1] if top_skills else 1

        # Tabular format - Top Skills Overview
        print("\n" + "📊 TOP SKILLS MARKET ANALYSIS (Tabular Format):")
        print("-" * 130)
        
        # Table header
        header = f"{'Skill':<20} {'Job Mentions':<15} {'% of Jobs':<15} {'Growth (MoM)':<15} {'Weight Score':<15} {'Trend':<40}"
        print(header)
        print("-" * 130)

        # Table rows
        for skill, count in top_skills[:15]:
            job_mention_count = f"{count:,}"  # Format with comma separator
            pct_of_jobs = (count / len(self.jobs_data)) * 100
            pct_of_jobs_str = f"{pct_of_jobs:.0f}%"
            
            # Get trend value
            trend_value = trends.get(skill, 0)
            trend_str = f"{trend_value:+.0f}%"
            
            # Calculate weight score
            weight_score = self.calculate_weight_score(count, max_count)
            weight_str = f"{weight_score:.1f} / 10"
            
            # Get trend emoji
            trend_emoji = self.get_trend_emoji(trend_value)
            
            # Print row
            row = f"{skill:<20} {job_mention_count:<15} {pct_of_jobs_str:<15} {trend_str:<15} {weight_str:<15} {trend_emoji:<40}"
            print(row)

        print("-" * 130)

        # Skills by Type with detailed breakdown
        print("\n" + "🏆 SKILLS BY CATEGORY (Detailed Breakdown):")
        print("-" * 130)
        skills_by_type = self.get_skills_by_type()
        total_skill_instances = sum(count for skill_count in skills_by_type.values() 
                                    for _, count in skill_count)
        
        for skill_type in sorted(skills_by_type.keys()):
            skills = skills_by_type[skill_type]
            if not skills:
                continue
            
            type_total = sum(count for _, count in skills)
            type_pct = (type_total / total_skill_instances * 100) if total_skill_instances > 0 else 0
            
            print(f"\n  📌 {skill_type}: {type_pct:.1f}% of all skills (Total: {type_total} mentions)")
            print(f"  {'Skill':<30} {'Count':<10} {'Weight %':<15}")
            print("  " + "-" * 70)
            
            for skill, count in skills[:10]:
                weight = (count / type_total * 100) if type_total > 0 else 0
                print(f"  {skill:<30} {count:<10} {weight:>6.1f}%")

        # Skills by Role
        print("\n" + "🎯 SKILLS BY ROLE CATEGORY:")
        print("-" * 130)
        for category, skills in self.get_role_skills(10).items():
            print(f"\n  {category}:")
            print(f"  {'Skill':<30} {'Job Count':<15}")
            print("  " + "-" * 60)
            for skill, count in skills[:10]:
                print(f"  {skill:<30} {count:<15}")

        print("\n" + "="*130 + "\n")
