# Try to import and use pysqlite3-binary for ChromaDB compatibility on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    # print("Successfully switched to pysqlite3-binary for sqlite3.") # Optional: for local debugging
except ImportError:
    # print("pysqlite3-binary not found, using system sqlite3.") # Optional: for local debugging
    pass # If pysqlite3-binary is not there, it will fall back, error might persist
except KeyError:
    # print("sqlite3 module already replaced or pysqlite3 not correctly imported.") # Optional: for local debugging
    pass


# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

# Attempt to load environment variables from .env file
load_dotenv()

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Conditional imports for LLMs
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

# --- App Configuration ---
st.set_page_config(
    page_title="PoliScanSight Bot 🕵️", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    # Correct password fetched from environment variable
    correct_password = os.environ.get("APP_PASSWORD")

    # If no password is set in environment, deny access or allow for local dev
    if not correct_password:
        # For local development, you might want to bypass this
        # return True # Uncomment for local dev if no password is set
        st.error("🔒 Password not configured for this application. Access denied.")
        st.stop()
        return False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.session_state.password_correct = False # Initialize
        password_placeholder = st.empty()
        password = password_placeholder.text_input("🔑 Enter Password", type="password", key="app_password_input")
        
        submit_button_placeholder = st.empty()
        if submit_button_placeholder.button("Submit", key="password_submit"):
            if password == correct_password:
                st.session_state.password_correct = True
                password_placeholder.empty() # Clear password input
                submit_button_placeholder.empty() # Clear submit button
                st.rerun() # Rerun to clear password elements and show app
            else:
                st.error("🚫 Incorrect password. Please try again.")
                st.session_state.password_correct = False
        else: # Only show input and button if not submitted yet
             st.info("🔐 This application is password protected.")
             st.stop() # Stop execution until password is submitted
        return False # Ensure app doesn't load until password is correct
        
    elif not st.session_state.password_correct:
        # Password not correct, show input again (should be handled by rerun logic)
        password_placeholder = st.empty()
        password = password_placeholder.text_input("🔑 Enter Password", type="password", key="app_password_input_retry")
        
        submit_button_placeholder = st.empty()
        if submit_button_placeholder.button("Submit", key="password_submit_retry"):
            if password == correct_password:
                st.session_state.password_correct = True
                password_placeholder.empty()
                submit_button_placeholder.empty()
                st.rerun()
            else:
                st.error("🚫 Incorrect password. Please try again.")
                st.session_state.password_correct = False
        else:
            st.info("🔐 This application is password protected.")
            st.stop()
        return False
    
    return True


# --- Tool and LLM Initialization ---
def initialize_serper_tool(api_key=None):
    """Initializes the SerperDevTool with the given API key."""
    if api_key:
        os.environ["SERPER_API_KEY"] = api_key
        return SerperDevTool()
    elif os.environ.get("SERPER_API_KEY"):
        # print("Serper API Key found in environment variables.") # For debugging
        return SerperDevTool()
    else:
        # st.warning("⚠️ Serper API Key not provided. Web search capabilities will be disabled. Validation and analysis quality may be affected.")
        return None

def create_llm(llm_choice, openai_api_key_value=None):
    """Creates and returns the selected LLM instance."""
    if llm_choice == "GPT-4o-mini":
        if not ChatOpenAI:
            st.error("❌ OpenAI library (langchain_openai) is not installed. Please install it to use GPT models.")
            return None
        if not openai_api_key_value and not os.environ.get("OPENAI_API_KEY"):
            st.error("❌ OpenAI API Key is required for GPT-4o-mini. Please provide it in the sidebar or set OPENAI_API_KEY environment variable.")
            return None
        api_key_to_use = openai_api_key_value or os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key_to_use)
    
    elif llm_choice == "Llama3.1 (Ollama)":
        if not Ollama:
            st.error("❌ Ollama library (langchain_community.llms) is not installed. Please install it to use Ollama models.")
            return None
        try:
            # Check if Ollama server is accessible by trying to list models or a similar lightweight call
            # This is a simplified check; a more robust check might involve an actual API call
            # For now, we assume if Ollama is chosen, the user has it running.
            return Ollama(model="llama3.1", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        except Exception as e:
            st.error(f"❌ Could not connect to Ollama. Ensure Ollama is running and accessible. Error: {e}")
            return None
    return None

# --- Validator Agent and Task ---
def create_validator_agent_and_task(entity_name, llm, serper_tool):
    """Creates a validator agent and its task to check if the entity is political."""
    validator_agent = Agent(
        role="Political Entity Validator",
        goal=f"""Your primary goal is to determine if the entity '{entity_name}' is a known political figure (e.g., minister, MP, MLA, party leader), a registered political party, or a significant political organization, with a strong focus on the Indian political landscape unless the name clearly indicates otherwise.
Search the web for evidence of political roles, affiliations, or activities.
If clear evidence of political status is found, respond with the exact string 'POLITICAL'.
If the entity is a common name with no clear, prominent political affiliation, or if it's clearly a non-political entity (e.g., a company, a fictional character, a general concept), respond with the exact string 'NON_POLITICAL'.
If the search is ambiguous but leans towards non-political or unknown, also respond 'NON_POLITICAL'.""",
        backstory="You are an expert in identifying political entities through web research. You are precise and only confirm political status if there is clear evidence. You provide concise, direct answers as specified.",
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False,
        verbose=True, # Set to False for less console output in production
        max_iter=3 # Limit iterations for validation
    )

    validation_task = Task(
        description=f"""Search the web for information about '{entity_name}'.
Determine if '{entity_name}' is a recognized political leader, political party, or a significant political organization, primarily within the Indian context unless the name strongly suggests otherwise.
Your final answer MUST be a single string: either 'POLITICAL' or 'NON_POLITICAL'. Do not provide any other explanation or text.""",
        expected_output="A single string: 'POLITICAL' or 'NON_POLITICAL'.",
        agent=validator_agent,
        # human_input=False # No human input for this automated task
    )
    return validator_agent, validation_task

# --- Political Analysis Agents and Tasks ---
def create_political_agents(political_entity_name, llm, serper_tool):
    """Creates the set of AI agents for political analysis."""
    political_affairs_investigator = Agent(
        role=f"Political Affairs & Community Impact Investigator for {political_entity_name}",
        goal=f"Investigate and detail {political_entity_name}'s recent (last 3-6 months unless specified otherwise) political activities, new schemes or policies launched, involvement in any significant controversies, and their perceived impact on local communities and public welfare. Identify specific examples and, where possible, URLs of news articles or official sources supporting these findings.",
        backstory=f"You are a seasoned investigative journalist specializing in Indian political dynamics and governance. You have a keen eye for detail, a knack for uncovering factual information from diverse online sources, and a commitment to objective reporting on {political_entity_name}.",
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=5 
    )

    state_political_landscape_analyst = Agent(
        role=f"State-Level Political Landscape & Competitor Strategy Analyst (Context: {political_entity_name})",
        goal=f"Analyze the current political landscape in the state/region relevant to {political_entity_name}. Identify key competing political parties and figures, their recent strategic moves, major campaigns, public reception, and any notable successes or failures. Focus on how these dynamics might affect {political_entity_name}. Identify specific examples and, where possible, URLs of news articles or official sources supporting these findings.",
        backstory=f"You are a political strategist with deep expertise in Indian state-level politics. You excel at dissecting competitor strategies, understanding voter sentiment shifts, and identifying emerging political trends that could impact {political_entity_name}.",
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=5
    )
    
    public_sentiment_analyst = Agent(
        role=f"Public Sentiment & Narrative Analyst for {political_entity_name}",
        goal=f"Analyze the prevailing public sentiment (positive, negative, neutral) towards {political_entity_name} over a specified recent period and compare it with a defined earlier period if data is available. Identify key themes, narratives, and events shaping this sentiment. Pinpoint the primary drivers for any significant shifts in public opinion. Identify specific examples and, where possible, URLs of news articles, social media trend discussions (if findable via web search), or official sources supporting these findings.",
        backstory=f"You are a data-driven media analyst specializing in tracking and interpreting public opinion trends related to political entities in India. You are adept at sifting through online information to gauge sentiment and understand the 'why' behind it for {political_entity_name}.",
        llm=llm,
        tools=[serper_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=5
    )

    chief_political_strategist = Agent(
        role=f"Chief Political Strategist & Report Architect for {political_entity_name}",
        goal=f"""Synthesize all gathered intelligence from the Political Affairs Investigator, State Landscape Analyst, and Public Sentiment Analyst into a comprehensive, coherent, and actionable strategic advisory report for {political_entity_name}.
The report should be structured, easy to understand (simple English), and cover:
1.  Executive Summary of {political_entity_name}'s current standing.
2.  Detailed analysis of {political_entity_name}'s recent activities and community impact.
3.  Assessment of the state-level political landscape and competitor strategies.
4.  In-depth look at public sentiment trends and their drivers.
5.  SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats) for {political_entity_name}.
6.  Actionable strategic recommendations for {political_entity_name} to enhance their political standing, address challenges, and capitalize on opportunities.
7.  A consolidated list of all unique source URLs provided by other agents, formatted plainly one URL per line. Ensure no duplicate URLs.
Your final output is ONLY this comprehensive report.""",
        backstory=f"You are a highly experienced Chief Political Strategist, renowned for your ability to translate complex political intelligence into clear, actionable strategies. You are crafting a vital advisory document for {political_entity_name}.",
        llm=llm,
        # No tools for the strategist, it synthesizes from other agents' work.
        allow_delegation=True, # Can delegate back to other agents if clarification is needed, though tasks are sequential.
        verbose=True,
        max_iter=5 
    )
    return political_affairs_investigator, state_political_landscape_analyst, public_sentiment_analyst, chief_political_strategist


def create_political_tasks(political_entity_name, agents, start_date, end_date, comparison_start_date, comparison_end_date):
    """Creates the set of tasks for the political analysis agents."""
    (political_affairs_investigator, state_political_landscape_analyst, 
     public_sentiment_analyst, chief_political_strategist) = agents

    task1_investigate = Task(
        description=f"""Investigate and report on the recent political activities, new schemes or policies launched, involvement in significant controversies, and the perceived community impact of '{political_entity_name}'.
Focus strictly on the period from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
Provide specific examples and find relevant URLs from news articles or official sources to support your findings.
Your output should be a detailed report section covering these aspects for {political_entity_name}.""",
        expected_output=f"A detailed report section on {political_entity_name}'s activities, schemes, controversies, and community impact for the specified period, including supporting URLs.",
        agent=political_affairs_investigator,
        # human_input=False
    )

    task2_analyze_landscape = Task(
        description=f"""Analyze the current political landscape in the state/region relevant to '{political_entity_name}'.
Identify key competing political parties and figures, their recent strategic moves, major campaigns, public reception, and any notable successes or failures.
Focus strictly on the period from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
Explain how these dynamics might affect '{political_entity_name}'.
Provide specific examples and find relevant URLs from news articles or official sources.
Your output should be a detailed report section covering these aspects.""",
        expected_output="A detailed report section on the state-level political landscape, competitor strategies, and their potential impact on the entity, including supporting URLs.",
        agent=state_political_landscape_analyst,
        # human_input=False
    )

    task3_analyze_sentiment = Task(
        description=f"""Analyze the prevailing public sentiment (e.g., positive, negative, neutral) towards '{political_entity_name}'.
Focus on sentiment during the primary period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
Also, research and compare this with the sentiment during a past reference period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}.
Identify key themes, narratives, and specific events shaping public sentiment in both periods. Pinpoint primary drivers for any significant shifts in opinion between these periods.
Provide specific examples and find relevant URLs (news, social media trends if possible via web search) to support your analysis.
Your output should be a detailed report section on public sentiment trends and drivers.""",
        expected_output="A detailed report section on public sentiment analysis, comparison between periods, key themes, drivers of sentiment shifts, including supporting URLs.",
        agent=public_sentiment_analyst,
        # human_input=False
    )
    
    task4_compile_report = Task(
        description=f"""Compile all the information gathered by the other agents regarding '{political_entity_name}' for the analysis period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
Structure this into a comprehensive 7-section strategic advisory report as outlined in your goal (Executive Summary, Recent Activities, State Landscape, Public Sentiment, SWOT, Strategic Recommendations, Consolidated URLs).
Write the report in simple, accessible English. Ensure all sections are well-developed.
The final section must be a plain list of all unique source URLs gathered by all agents, with each URL on a new line. Do not include any other text in this URL list section.
Your final output is ONLY this complete report.""",
        expected_output="A comprehensive 7-section strategic advisory report, including a SWOT analysis, actionable recommendations, and a consolidated list of unique source URLs.",
        agent=chief_political_strategist,
        context=[task1_investigate, task2_analyze_landscape, task3_analyze_sentiment], # Depends on the output of other tasks
        # human_input=False 
    )
    return [task1_investigate, task2_analyze_landscape, task3_analyze_sentiment, task4_compile_report]


# --- Main Application Logic ---
def run_app():
    """Runs the Streamlit application."""
    st.title("🕵️ PoliScanSight Bot")
    st.markdown("""
    Welcome to PoliScanSight Bot! This AI-powered tool helps you analyze a political entity in India.
    Provide the entity's name, analysis period, and select your preferred AI model.
    The bot will generate a report covering recent activities, competitor landscape, public sentiment, and strategic recommendations.
    """)
    st.markdown("---")

    # Initialize session state variables if they don't exist
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_in_progress" not in st.session_state:
        st.session_state.analysis_in_progress = False
    if "entity_name_for_report" not in st.session_state:
        st.session_state.entity_name_for_report = ""
    if "start_date_for_report" not in st.session_state:
        st.session_state.start_date_for_report = date.today() - timedelta(days=30)


    with st.sidebar:
        st.header("⚙️ Configuration")
        
        entity_name_input = st.text_input("👤 Political Entity Name (e.g., Person or Party)", 
                                          help="Enter the full name of the political leader or party you want to analyze.",
                                          value=st.session_state.get("last_entity_name", ""))
        
        # Analysis period
        today = date.today()
        default_start_date = today - timedelta(days=30) # Default to last 30 days
        
        start_date_input = st.date_input(
            "🗓️ Analysis Start Date",
            value=st.session_state.get("last_start_date", default_start_date),
            min_value=today - timedelta(days=365*3), # Max 3 years back
            max_value=today - timedelta(days=1), # Must be at least yesterday
            help="The beginning of the period for analysis. Ends today."
        )
        
        llm_options = []
        if ChatOpenAI:
            llm_options.append("GPT-4o-mini")
        if Ollama:
            llm_options.append("Llama3.1 (Ollama)")

        if not llm_options:
            st.error("No LLMs available. Please ensure relevant libraries are installed (OpenAI, Ollama).")
            st.stop()
            
        llm_choice = st.selectbox("🤖 Select LLM", options=llm_options, 
                                  help="Choose the AI model to power the analysis.")

        with st.expander("🔑 API Keys & Advanced Settings"):
            openai_api_key_value = st.text_input("OpenAI API Key", type="password", 
                                                 help="Required if using GPT models. Leave blank if set as environment variable.",
                                                 value=os.environ.get("OPENAI_API_KEY", ""))
            serper_api_key_value = st.text_input("Serper API Key (for Web Search)", type="password", 
                                                 help="Required for web search capabilities. Leave blank if set as environment variable.",
                                                 value=os.environ.get("SERPER_API_KEY", ""))
            # ollama_base_url_value = st.text_input("Ollama Base URL", 
            #                                     value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            #                                     help="URL for your Ollama server if not default.")
            # if ollama_base_url_value:
            #     os.environ["OLLAMA_BASE_URL"] = ollama_base_url_value


        if st.button("🚀 Generate Report", use_container_width=True, type="primary", disabled=st.session_state.analysis_in_progress):
            if not entity_name_input:
                st.error("❌ Please enter a Political Entity Name.")
            elif not start_date_input:
                st.error("❌ Please select an Analysis Start Date.")
            elif start_date_input >= date.today():
                st.error("❌ Analysis Start Date must be in the past.")
            else:
                st.session_state.analysis_in_progress = True
                st.session_state.analysis_result = None # Clear previous results
                st.session_state.entity_name_for_report = entity_name_input
                st.session_state.start_date_for_report = start_date_input
                st.session_state.last_entity_name = entity_name_input # Remember for next time
                st.session_state.last_start_date = start_date_input  # Remember for next time
                st.rerun() # Rerun to show progress indicator and disable button

    # --- Main Content Area ---
    if st.session_state.analysis_in_progress:
        with st.spinner(f"🔄 Analyzing '{st.session_state.entity_name_for_report}'... This may take a few minutes."):
            # Initialize LLM and Tools
            llm = create_llm(llm_choice, openai_api_key_value if llm_choice == "GPT-4o-mini" else None)
            serper_tool = initialize_serper_tool(serper_api_key_value)

            if not llm:
                st.error("Failed to initialize the LLM. Please check your configuration and API keys.")
                st.session_state.analysis_in_progress = False
                st.rerun()
                return # Stop execution

            if not serper_tool:
                st.error("🚫 Serper API Key for web search is not configured or is invalid. Cannot perform entity validation or detailed political analysis. Please provide the API key.")
                st.session_state.analysis_in_progress = False
                st.rerun()
                return # Stop execution

            # 1. Validate Entity
            validation_status_placeholder = st.empty()
            validation_status_placeholder.info(f"🔍 Validating '{st.session_state.entity_name_for_report}' as a political entity...")
            
            validator_agent, validation_task = create_validator_agent_and_task(st.session_state.entity_name_for_report, llm, serper_tool)
            validation_crew = Crew(
                agents=[validator_agent],
                tasks=[validation_task],
                verbose=0, # 0 for less output on Streamlit page, 1 or 2 for console debugging
                process=Process.sequential,
                memory=False,
            )
            try:
                validation_result = validation_crew.kickoff()
                validation_status_placeholder.empty() # Clear validation message

                if validation_result and "POLITICAL" in validation_result.upper():
                    st.success(f"✅ '{st.session_state.entity_name_for_report}' identified as a political entity. Proceeding with analysis.")
                    
                    # 2. Proceed with Political Analysis
                    analysis_status_placeholder = st.empty()
                    analysis_status_placeholder.info(f"📊 Performing detailed analysis for '{st.session_state.entity_name_for_report}'...")

                    end_date_val = date.today()
                    # Define comparison period (e.g., 5-10 days before the start_date_input)
                    # For simplicity, let's make it a fixed 7 days before the start date, for a 7-day window
                    comparison_end_date_val = st.session_state.start_date_for_report - timedelta(days=1)
                    comparison_start_date_val = comparison_end_date_val - timedelta(days=6) # 7 day window

                    agents = create_political_agents(st.session_state.entity_name_for_report, llm, serper_tool)
                    tasks = create_political_tasks(st.session_state.entity_name_for_report, agents, st.session_state.start_date_for_report, end_date_val, comparison_start_date_val, comparison_end_date_val)
                    
                    main_crew = Crew(
                        agents=list(agents),
                        tasks=tasks,
                        process=Process.sequential, # Tasks will be executed one after another
                        verbose=0, # Or 1 for more detailed console logs
                        # memory=True, # Enable memory if agents need to recall past interactions in a longer conversation (not typical for sequential task execution)
                        # embedder can be configured here if using memory with specific embedding needs
                    )
                    
                    final_report = main_crew.kickoff()
                    st.session_state.analysis_result = final_report
                    analysis_status_placeholder.empty() # Clear analysis message

                elif validation_result and "NON_POLITICAL" in validation_result.upper():
                    st.error(f"⚠️ '{st.session_state.entity_name_for_report}' does not appear to be a recognized political entity. Please provide a valid political entity name.")
                    st.session_state.analysis_result = None
                else:
                    st.warning(f"🤔 Could not definitively validate '{st.session_state.entity_name_for_report}' as a political entity. The validator agent returned: '{validation_result}'. Please provide a clear political entity name.")
                    st.session_state.analysis_result = None

            except Exception as e:
                st.error(f"An error occurred during the process: {e}")
                st.session_state.analysis_result = None
            finally:
                st.session_state.analysis_in_progress = False
                st.rerun() # Rerun to update UI based on results or errors

    # Display results if available and not in progress
    if not st.session_state.analysis_in_progress and st.session_state.analysis_result:
        st.subheader(f"📊 Analysis Report for: {st.session_state.entity_name_for_report}")
        st.markdown(f"**Analysis Period:** {st.session_state.start_date_for_report.strftime('%B %d, %Y')} to {date.today().strftime('%B %d, %Y')}")
        
        st.info(
            "📌 **Note on Source URLs:** This report includes URLs to support its findings. While the AI strives to provide accurate and relevant links, "
            "some URLs may be illustrative, lead to broader context pages, or occasionally be placeholders if a highly specific source for every "
            "synthesized detail could not be pinpointed during the automated generation process. Please verify critical information independently."
        )

        with st.expander("View Full Report", expanded=True):
            st.markdown(st.session_state.analysis_result)
        
        report_data_for_download = f"# Comprehensive Political Analysis & Strategic Advisory: {st.session_state.entity_name_for_report} (Analysis Period: {st.session_state.start_date_for_report.strftime('%Y-%m-%d')} to {date.today().strftime('%Y-%m-%d')})\n\n{st.session_state.analysis_result}"
        download_file_name = f"{st.session_state.entity_name_for_report.replace(' ', '_').lower()}_report_{st.session_state.start_date_for_report.strftime('%Y-%m-%d')}_to_{date.today().strftime('%Y-%m-%d')}.md"
        st.download_button(
            label="📥 Download Report as Markdown",
            data=report_data_for_download,
            file_name=download_file_name,
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.caption("PoliScanSight Bot | Powered by CrewAI & Streamlit")

# --- Main Execution with Password Check ---
if __name__ == "__main__":
    if check_password(): # Check password first
        run_app()
