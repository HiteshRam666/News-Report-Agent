from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool


# define the websearch tool
web_search_tool = SerperDevTool()

# create a pydantic model for structured output
class NewsItem(BaseModel):
    headline: str = Field(description="Headline for the news")
    url: str = Field(description="URL of the news article")
    news_summary: str = Field(description="One-paragraph summary of the news")
    news_agency_name: str = Field(description="Publishing news agency")

class NewsReportOutput(BaseModel):
    country: str = Field(description="Country for which news was fetched")
    top_headlines: List[NewsItem]

@CrewBase
class NewsReportCrew():
    """NewsReportCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # config paths
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    # define my agent for the crew 
    @agent
    def news_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["news_reporter"],
            verbose=True,
            tools=[web_search_tool]
        )
        
    # define the task for the crew
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            output_file="news.json",
            output_json=NewsReportOutput  
        )
    
    
    # define the crew with my agents and tasks
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential
        )