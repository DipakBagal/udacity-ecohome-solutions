"""
EcoHome Project Validation Script
Tests all code for syntax errors, imports, and basic functionality
Run this before full setup to validate the codebase
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def main():
    print_header("EcoHome Energy Advisor - Code Validation")
    
    # Track results
    total_checks = 0
    passed_checks = 0
    
    # 1. Check Python files syntax
    print_header("1. Checking Python File Syntax")
    python_files = [
        'agent.py',
        'tools.py',
        'models/energy.py',
        'models/__init__.py'
    ]
    
    for file in python_files:
        total_checks += 1
        if check_file_exists(file):
            is_valid, error = check_python_syntax(file)
            if is_valid:
                print_success(f"{file} - Valid syntax")
                passed_checks += 1
            else:
                print_error(f"{file} - Syntax error: {error}")
        else:
            print_error(f"{file} - File not found")
    
    # 2. Check required files exist
    print_header("2. Checking Required Files")
    required_files = [
        'requirements.txt',
        'README.md',
        'PROJECT_SUMMARY.md',
        '.gitignore',
        '.env.example',
        '01_db_setup.ipynb',
        '02_rag_setup.ipynb',
        '03_run_and_evaluate.ipynb'
    ]
    
    for file in required_files:
        total_checks += 1
        if check_file_exists(file):
            print_success(f"{file} exists")
            passed_checks += 1
        else:
            print_error(f"{file} missing")
    
    # 3. Check knowledge base documents
    print_header("3. Checking Knowledge Base Documents")
    kb_files = [
        'data/documents/tip_device_best_practices.txt',
        'data/documents/tip_energy_savings.txt',
        'data/documents/hvac_optimization.txt',
        'data/documents/smart_home_automation.txt',
        'data/documents/renewable_energy_integration.txt',
        'data/documents/seasonal_energy_management.txt',
        'data/documents/energy_storage_optimization.txt'
    ]
    
    kb_total_lines = 0
    for file in kb_files:
        total_checks += 1
        if check_file_exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                kb_total_lines += lines
            print_success(f"{Path(file).name} exists ({lines} lines)")
            passed_checks += 1
        else:
            print_error(f"{file} missing")
    
    print(f"\n  Total knowledge base: {kb_total_lines} lines")
    
    # 4. Check environment setup
    print_header("4. Checking Environment Configuration")
    total_checks += 1
    if check_file_exists('.env'):
        print_success(".env file exists")
        passed_checks += 1
        
        # Check if API key is set
        try:
            with open('.env', 'r') as f:
                content = f.read()
                if 'OPENAI_API_KEY=' in content and 'your_openai_api_key_here' not in content:
                    print_success("OpenAI API key appears to be configured")
                else:
                    print_warning("OpenAI API key may not be set in .env")
        except:
            pass
    else:
        print_warning(".env file not found (copy from .env.example)")
    
    # 5. Test imports (without installing)
    print_header("5. Testing Import Statements")
    
    print("Testing if required packages are importable...")
    packages_to_test = [
        ('langchain', 'langchain'),
        ('langchain_openai', 'langchain-openai'),
        ('langchain_community', 'langchain-community'),
        ('langgraph', 'langgraph'),
        ('openai', 'openai'),
        ('chromadb', 'chromadb'),
        ('sqlalchemy', 'sqlalchemy'),
        ('dotenv', 'python-dotenv')
    ]
    
    for module, package in packages_to_test:
        total_checks += 1
        try:
            __import__(module)
            print_success(f"{package} is installed")
            passed_checks += 1
        except ImportError:
            print_warning(f"{package} not installed (run: pip install {package})")
    
    # 6. Check code structure
    print_header("6. Checking Code Structure")
    
    # Check agent.py structure
    total_checks += 1
    if check_file_exists('agent.py'):
        with open('agent.py', 'r', encoding='utf-8') as f:
            content = f.read()
            required_elements = [
                ('ECOHOME_SYSTEM_PROMPT', 'System prompt defined'),
                ('class EcoHomeAgent', 'Agent class defined'),
                ('def create_agent', 'Factory function defined'),
                ('LangGraph', 'LangGraph imported'),
                ('StateGraph', 'StateGraph used')
            ]
            
            all_present = True
            for element, description in required_elements:
                if element in content:
                    print_success(description)
                else:
                    print_error(f"{description} - NOT FOUND")
                    all_present = False
            
            if all_present:
                passed_checks += 1
    
    # Check tools.py structure
    total_checks += 1
    if check_file_exists('tools.py'):
        with open('tools.py', 'r', encoding='utf-8') as f:
            content = f.read()
            required_tools = [
                'get_weather_forecast',
                'get_electricity_prices',
                'query_energy_usage',
                'query_solar_generation',
                'search_energy_tips'
            ]
            
            all_present = True
            for tool in required_tools:
                if f'def {tool}' in content:
                    print_success(f"Tool: {tool}")
                else:
                    print_error(f"Tool: {tool} - NOT FOUND")
                    all_present = False
            
            if all_present:
                passed_checks += 1
    
    # Check models/energy.py structure
    total_checks += 1
    if check_file_exists('models/energy.py'):
        with open('models/energy.py', 'r', encoding='utf-8') as f:
            content = f.read()
            required_elements = [
                'class EnergyUsage',
                'class SolarGeneration',
                'def get_engine',
                'def get_session',
                'def init_db'
            ]
            
            all_present = True
            for element in required_elements:
                if element in content:
                    print_success(element)
                else:
                    print_error(f"{element} - NOT FOUND")
                    all_present = False
            
            if all_present:
                passed_checks += 1
    
    # Final summary
    print_header("Validation Summary")
    
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"Total checks: {total_checks}")
    print(f"Passed: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"Failed: {Colors.RED}{total_checks - passed_checks}{Colors.END}")
    print(f"Success rate: {success_rate:.1f}%\n")
    
    if success_rate == 100:
        print_success("All validation checks passed! ✨")
        print("\nNext steps:")
        print("  1. Ensure virtual environment is activated")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Set up .env file with OPENAI_API_KEY")
        print("  4. Run: jupyter notebook 01_db_setup.ipynb")
        print("  5. Run: jupyter notebook 02_rag_setup.ipynb")
        print("  6. Run: jupyter notebook 03_run_and_evaluate.ipynb")
    elif success_rate >= 80:
        print_warning("Most checks passed, but some issues need attention")
        print("\nReview the errors above and fix before proceeding")
    else:
        print_error("Multiple validation failures detected")
        print("\nPlease fix the errors above before running the project")
    
    return 0 if success_rate == 100 else 1

if __name__ == "__main__":
    sys.exit(main())
