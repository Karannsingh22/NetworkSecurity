"""
setup.py packages the `networksecurity` library so it can be installed with
`pip install -e .` and imported as `import networksecurity...` from
anywhere, including inside app.py / main.py.
"""

from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """Read requirements.txt and return a clean list of requirement strings."""
    requirement_lst: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst


setup(
    name="NetworkSecurity",
    version="0.1.0",
    description="Local, fully offline network security (phishing website) threat detection ML project",
    author="Your Name",
    packages=find_packages(),
    install_requires=get_requirements(),
)
