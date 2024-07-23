# L5 Automate event planning
# from https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai/lesson/12/automate-event-planning-(code)

# The Json model creation according to the pedantic schema seems to have issues in both openai and local models such as gemma2:27b


# Warning control
import os
import warnings
from langchain_community.llms import Ollama
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

#Memory requires embedding and currently Ollama (https://ollama.com/blog/embedding-models) is not supported AFAIK
# Ref https://docs.crewai.com/core-concepts/Memory/?h=memory#implementing-memory-in-your-crew  and https://ollama.com/blog/embedding-models
#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()


llm = Ollama(
    model = "llama3:8b",
    base_url = "http://host.docker.internal:11434")

from crewai import Agent, Task, Crew

from crewai_tools import ScrapeWebsiteTool, \
                         WebsiteSearchTool

scrape_tool = ScrapeWebsiteTool()
search_tool = WebsiteSearchTool()

# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
    "based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    ),
    llm=llm
)

 # Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event "
        "including catering and equipmen"
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    ),
    llm=llm
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and "
         "communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    ),
    llm=llm
)

from pydantic import BaseModel
# Define a Pydantic model for venue details 
# (demonstrating Output as Pydantic)
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str


venue_task = Task(
    description="Find a venue in {event_city} "
                "that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",  
      # Outputs the venue details as a JSON file
    agent=venue_coordinator
)

logistics_task = Task(
    description="Coordinate catering and "
                 "equipment for an event "
                 "with {expected_participants} participants "
                 "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} "
                "aiming to engage at least"
                "{expected_participants} potential attendees.",
    expected_output="Report on marketing activities "
                    "and attendee engagement formatted as markdown.",
    output_file="marketing_report.md",  # Outputs the report as a text file
    agent=marketing_communications_agent
)

# Define the crew with agents and tasks
event_management_crew = Crew(
    agents=[venue_coordinator, 
            logistics_manager, 
            marketing_communications_agent],
    
    tasks=[venue_task, 
           logistics_task, 
           marketing_task],
    
    verbose=True
)

event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "Stavanger Norway",
    'tentative_date': "2024-10-15",
    'expected_participants': 100,
    'budget': 200000,
    'venue_type': "Conference Hall"
}

result = event_management_crew.kickoff(inputs=event_details)