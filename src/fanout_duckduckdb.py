import os
import json
import datetime
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from tavily import TavilyClient

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

AGENT_DIR = Path(__file__).parent.parent
RESULTS_DIR = AGENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def get_result_filename(prefix):
    return f"{prefix}_fanout_decompose_gpt3.5_synthesis_gpt3.5_duckduckgo.json"


# Initialize components
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=openai_api_key)
llm_decompose = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=openai_api_key)
llm_synthesis = ChatOpenAI(temperature=0, model="gpt-4o",api_key=openai_api_key)


# Query decomposition prompt
decompose_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Break down the following complex question into 3-5 sub-questions that need to be answered to fully address the main query. Present them as bullet points.

    Main Question: {query}

    Sub-questions:"""
)

# Answer synthesis prompt
synthesize_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Synthesize a comprehensive answer to the following question using the provided context.
    Clearly indicate which information comes from which source. If you don't know the answer, say so.
    Format in markdown.

    Question: {query}

    Context: {context}

    Answer:"""
)

# Configure search tool
search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool(
    name="Deep Search",
    func=search.run,
    description="Useful for searching current information from diverse sources"
)




class DeepSearchAgent:
    def __init__(self):
        self.llm = llm
        self.decompose_chain = LLMChain(llm=llm_decompose, prompt=decompose_prompt)
        self.synthesize_chain = LLMChain(llm=llm_synthesis, prompt=synthesize_prompt)
        self.search_tool = search_tool

    def save_results(self, data, prefix):
        filename = RESULTS_DIR / get_result_filename(prefix)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename

    def decompose_query(self, query):
        decomposition = self.decompose_chain.run(query=query)
        subqueries = [q.strip("- ") for q in decomposition.split("\n") if q.strip()]
        self.save_results(
            {"original_query": query, "subqueries": subqueries},
            "subqueries"
        )
        return subqueries
    
    def search_subqueries(self, subqueries):
        search_results = []
        for i, subq in enumerate(subqueries, 1):
            print(f"Searching subquery {i}/{len(subqueries)}: {subq}")
            results = self.search_tool.run(subq)
            search_results.append({
                "subquery": subq,
                "results": results
            })
        
        # Save search results
        filename = self.save_results(
            {"search_results": search_results},
            "search_results"
        )
        
        # Format context for synthesis
        context = []
        for result in search_results:
            context.append(f"## Subquestion: {result['subquery']}\nSearch Results:\n{result['results']}")
        return "\n\n".join(context)


    def synthesize_answer(self, query, context):
        final_answer = self.synthesize_chain.run(query=query, context=context)
        # Save final answer
        self.save_results(
            {
                "query": query,
                "final_answer": final_answer,
                "context_used": context
            },
            "final_answer"
        )
        return final_answer
    
    def run(self, query):
        print("Decomposing query...")
        subqueries = self.decompose_query(query)
        print(f"Subqueries: {subqueries}")

        print("\nGathering information...")
        context = self.search_subqueries(subqueries)
        print(f"Context: {context}")

        print("\nSynthesizing final answer...")
        return self.synthesize_answer(query, context)

# Usage

def main():
    query = """I am a researcher interested in bringing mixed-gas sorption capabilities to my lab. Please discuss the differences between pure- and mixed-gas sorption for glassy polymers, how the dual-mode sorption model can be used to predict mixed-gas sorption behavior in glassy polymers (include equations where applicable), and what challenges there are in accurately predicting pure- and mixed-gas sorption using the dual-mode sorption model."""
    agent = DeepSearchAgent()
    result = agent.run(query)
    print("\nFinal Answer:")
    print(result)

if __name__ == "__main__":
    main()