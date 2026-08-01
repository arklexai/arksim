import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import json
from agent.financial_agent import FinancialAdvisorAgent

BASE_DIR = Path(__file__).resolve().parents[1]
SCENARIO_PATH = BASE_DIR / "scenarios.json"


def main():
    scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    scenario = scenarios[0]

    agent = FinancialAdvisorAgent(model="gpt-4o-mini")
    result = agent.run(scenario)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()