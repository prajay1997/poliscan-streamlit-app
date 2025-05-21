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

# --- Global Variables & Helper Functions ---
search_tool_instance = None 

# --- Password Protection ---
def check_password():
    """Returns `True` if the user has entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    correct_password = os.environ.get("APP_PASSWORD")

    if not correct_password:
        st.error("🔒 APP_PASSWORD environment variable not set. Please configure it in your .env file or system environment.")
        st.info("Create a `.env` file in the app directory with `APP_PASSWORD=\"your_password_here\"`")
        return False

    if st.session_state.password_correct:
        return True

    password_placeholder = st.empty()
    password_attempt = password_placeholder.text_input("🔑 Enter App Password", type="password", key="password_input_main_v5") 

    if password_attempt:
        if password_attempt == correct_password:
            st.session_state.password_correct = True
            password_placeholder.empty() 
            st.rerun() 
        else:
            st.error("😕 Incorrect password. Please try again.")
            st.session_state.password_correct = False
    else:
        st.info("🔐 This application is password protected.")

    return st.session_state.password_correct

# --- Core Logic Functions ---
def initialize_serper_tool(api_key):
    global search_tool_instance
    if api_key:
        os.environ["SERPER_API_KEY"] = api_key
        try:
            search_tool_instance = SerperDevTool()
            st.session_state.serper_initialized = True
            return True
        except Exception as e:
            st.error(f"SerperDevTool Error: {e}") 
            search_tool_instance = None
            st.session_state.serper_initialized = False
            return False
    else:
        search_tool_instance = None
        st.session_state.serper_initialized = False
        return False

def create_llm(llm_choice, openai_api_key_value):
    if llm_choice == "GPT-4o-mini (OpenAI)":
        if not ChatOpenAI:
            st.error("langchain_openai is not available.")
            return None
        if not openai_api_key_value:
            st.error("OpenAI API Key is required for GPT-4o-mini. Please provide it in the sidebar under 'API Keys & Web Search Setup'.")
            return None
        os.environ["OPENAI_API_KEY"] = openai_api_key_value
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key_value)
    elif llm_choice == "Llama3.1 (Local Ollama)":
        if Ollama is None:
            st.error("Ollama from langchain_community.llms is not available.")
            return None
        try:
            return Ollama(model="llama3.1")
        except Exception as e:
            st.error(f"Error initializing Ollama: {e}. Ensure Ollama server is running.")
            return None
    return None

def create_political_agents(political_entity_name, llm):
    global search_tool_instance 
    agent_tools_list = []
    if search_tool_instance and st.session_state.get('serper_initialized', False):
        agent_tools_list.append(search_tool_instance)

    activity_researcher = Agent(
        role="Political Affairs & Community Impact Investigator",
        goal=f"Conduct an in-depth investigation into {political_entity_name}'s recent (user-defined analysis period) political activities, "
             f"new schemes, any significant controversies (with their direct and indirect impact on image/perception), "
             f"and the specific, detailed impact of these on local communities or caste groups. "
             f"For key factual claims, new schemes, and significant controversies, if a direct and relevant public URL (e.g., from a reputable news source or official government page) "
             f"is found that substantiates the information, include it. Prioritize quality and direct relevance of URLs. "
             f"Compile any collected URLs as a list at the end of your findings for this task.",
        backstory="You are a highly skilled investigative journalist specializing in state-level politics and socio-economic impacts. "
                  "You uncover not just facts, but also their nuanced implications. You are meticulous about verifying information and strive to provide relevant source URLs for critical information when available. "
                  "Your reports on community impact are expected to be substantive.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=20
    )

    landscape_monitor = Agent(
        role="State Political Landscape & Competitor Strategy Analyst",
        goal=f"Monitor and deeply analyze the significant activities, underlying STRATEGIES, and political campaigns of ALL major political parties "
             f"and their key leaders in the state (relevant to {political_entity_name}) during the user-defined analysis period. "
             f"Focus on their stated aims for state betterment, their actual methods for increasing vote share (e.g., narratives, target segments), "
             f"and analyze their successes or failures. "
             f"If direct public URLs from reputable sources are found for reported activities or strategic claims, include them. "
             f"Compile any collected URLs as a list at the end of your findings for this task.",
        backstory="An expert political strategist analyzing state-level competitive dynamics. You don't just list events; you dissect the strategies, "
                  "target audiences, and effectiveness of all major political players. You aim to source key strategic claims with relevant URLs when possible.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Public Sentiment & Narrative Analyst",
        goal=f"Conduct a DETAILED quantitative and qualitative sentiment analysis for {political_entity_name} for the user-defined analysis period, "
             f"and compare it with sentiment from 5 to 10 days BEFORE the start of that analysis period. "
             f"You MUST output: "
             f"1) Current sentiment breakdown (Positive: X%, Negative: Y%, Neutral: Z%). "
             f"2) Past sentiment breakdown (Positive: A%, Negative: B%, Neutral: C%). "
             f"3) Explicit trend analysis (e.g., 'Negative sentiment surged by K percentage points from B% to Y%'). "
             f"4) Identify 1-2 KEY THEMES or prominent HASHTAGS (e.g., 'MissingCM' if verifiable from data) driving the current sentiment. "
             f"5) Clearly explain reasons for any shifts in simple terms. "
             f"Use the search tool effectively for specific date ranges. If URLs for sentiment data/reports or articles directly reflecting these themes are found, include them. "
             f"Compile any collected URLs as a list at the end of your findings.",
        backstory="A specialist in dissecting public opinion. You go beyond surface-level sentiment, providing precise quantitative breakdowns, "
                  "identifying trends, and uncovering the core narratives and events that shape public perception. You try to link sentiment drivers to citable sources if available.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=15
    )

    report_writer = Agent(
        role="Chief Political Strategist & Report Architect",
        goal=f"Compile ALL findings from the other agents into a single, comprehensive, and strategically insightful report "
             f"about {political_entity_name}, following the detailed 7-section political analysis prompt. "
             f"The ENTIRE content of the report MUST be written in **simple, clear English** but must retain the analytical depth and detail requested. "
             f"Develop insightful and actionable 'Strategic Recommendations' (Section VI) with detailed justifications, similar in depth to prior examples of good recommendations. "
             f"Compile ALL unique and relevant URLs provided by other agents into a final 'Compiled Reference List' (Section VII). "
             f"Each URL in this list should be on a new line, formatted clearly as a plain URL (e.g., https://www.example.com/article) without any surrounding Markdown link syntax, as Streamlit will render these. " # Added instruction for plain URL formatting
             f"Ensure these URLs appear to be valid and directly support key information. Do not include URLs inline within Sections I-V.",
        backstory="A seasoned political strategist and master communicator. You transform complex intelligence into clear, actionable advice. "
                  "You excel at crafting comprehensive reports in simple language that directly address every part of a client's request, providing strategic depth and ensuring all relevant, verifiable sources are meticulously compiled at the end.",
        verbose=True, allow_delegation=False, llm=llm, max_iter=15
    )
    return [activity_researcher, landscape_monitor, sentiment_analyzer, report_writer]

def create_political_tasks(political_entity_name, agents, start_date_str, current_date_str, past_sentiment_start_str, past_sentiment_end_str):
    activity_researcher, landscape_monitor, sentiment_analyzer, report_writer = agents
    analysis_period_str = f"between {start_date_str} and {current_date_str}"
    past_sentiment_period_str = f"between {past_sentiment_start_str} and {past_sentiment_end_str}"

    date_filter_instruction_main_period = (
        f"Your research MUST focus on information published or relevant strictly {analysis_period_str}. "
        f"When using the web search tool, you MUST craft your queries to find information ONLY from this period. For example, "
        f"include terms like 'news after:{start_date_str} before:{current_date_str}' or similar specific date restrictions in your search query string provided to the tool."
    )
    date_filter_instruction_past_sentiment = (
         f"When using the web search tool for past sentiment context, you MUST craft your queries to find information ONLY from the period {past_sentiment_period_str}. For example, "
        f"include terms like 'news after:{past_sentiment_start_str} before:{past_sentiment_end_str}' in your search query string."
    )
    url_collection_instruction_agent = (
        "If you find direct and relevant public URLs (e.g., from reputable news sources or official pages) that substantiate key factual claims, "
        "please collect them. At the end of your response for THIS TASK, provide a consolidated list of these high-quality URLs under a clear heading like 'Collected URLs for this task:'. "
        "Do not invent URLs or use placeholders if a suitable source is not found for a specific detail."
    )

    task1_research_activities_community = Task(
        description=f"**Primary Goal for {political_entity_name}**: Conduct an in-depth investigation for the period {analysis_period_str}. {url_collection_instruction_agent}\\n"
                    f"   **Search Instructions**: {date_filter_instruction_main_period}\\n"
                    f"   **Specific Areas to Cover**:\\n"
                    f"   1.  **Section I - Recent Political Activities & Scheme Launches**: Detail the nature of each activity (rallies, policy announcements etc.), key messages conveyed, and any new schemes (name, purpose, beneficiaries, potential impact). Be specific and provide details.\\n"
                    f"   2.  **Section II - Controversies Involving {political_entity_name}**: Describe each significant controversy, identify key individuals/aspects, analyze media coverage intensity and general tone. Most importantly, provide a detailed analysis of the direct and indirect impact of these controversies on {political_entity_name}'s image and public perception.\\n"
                    f"   3.  **Section III - Community and Caste-Specific Impact Analysis**: Provide a substantive analysis (not just a brief mention) if and how specific communities or caste groups are reportedly being affected (positively or negatively) by {political_entity_name}'s recent activities. Also, research if opposition party campaigns are reportedly benefiting or targeting these specific groups. If specific data or strong anecdotal evidence is found, highlight it. If little information is found, state that clearly.",
        agent=activity_researcher,
        expected_output=f"A detailed, multi-part report covering Sections I, II, and III of the main political analysis prompt for {political_entity_name} specifically for the period {analysis_period_str}. "
                        f"The content must be factual, deeply analytical (especially for controversy impact and community effects), and written clearly. "
                        f"The output for this task should conclude with a list of relevant source URLs under the heading 'Collected URLs for this task:', if any were found.",
        async_execution=False
    )
    task2_monitor_landscape = Task(
        description=f"**Primary Goal**: Monitor and analyze the broader political landscape in the state of {political_entity_name} for the period {analysis_period_str}. {url_collection_instruction_agent}\\n"
                    f"   **Search Instructions**: {date_filter_instruction_main_period}\\n"
                    f"   **Specific Areas to Cover (Section V of final report)**:\\n"
                    f"   1. Report on significant activities of **ALL major political parties** and their key leaders in the state.\\n"
                    f"   2. For EACH major party, provide a detailed analysis of their: \\n"
                    f"      a. Recent public engagements, announcements, or policy stances.\\n"
                    f"      b. **Underlying STRATEGIES and specific CAMPAIGNS** they are visibly adopting/running aimed at i) the betterment of the state (e.g., development initiatives, governance reforms) AND ii) increasing their vote share or public support (e.g., outreach programs, narrative building, target voter segments, key issues highlighted).\\n"
                    f"      c. Key messages being pushed to the public.\\n"
                    f"      d. Any notable successes or failures in their recent strategic efforts.",
        agent=landscape_monitor,
        expected_output=f"A comprehensive and strategic analysis of the activities and positioning of ALL major political players in the state for {analysis_period_str}, aligning with Section V of the main political analysis prompt. "
                        f"The analysis for each party should go beyond listing events and delve into their strategies and campaign effectiveness. "
                        f"The output for this task should conclude with a list of relevant source URLs under the heading 'Collected URLs for this task:', if any were found.",
        context=[task1_research_activities_community],
        async_execution=False
    )
    task3_analyze_sentiment = Task(
        description=f"**Primary Goal**: Conduct a DETAILED public sentiment analysis for {political_entity_name}.\\n"
                    f"   **Current Sentiment Period**: Analyze sentiment for **{analysis_period_str}**. {url_collection_instruction_agent.replace('key factual claims, activities, and claims', 'articles or data points specifically used for sentiment context during this period')}\\n"
                    f"   **Past Sentiment Period for Comparison**: Analyze sentiment for **{past_sentiment_period_str}** (5-10 days before {start_date_str}). {date_filter_instruction_past_sentiment} {url_collection_instruction_agent.replace('key factual claims, activities, and claims', 'articles or data points specifically used for sentiment context during this past period')}\\n"
                    f"   **Required Output Details (Section IV of final report)**:\\n"
                    f"   1.  **Current Sentiment ({analysis_period_str})**: Provide breakdown: Positive: X%, Negative: Y%, Neutral: Z%.\\n"
                    f"   2.  **Past Sentiment ({past_sentiment_period_str})**: Provide breakdown: Positive: A%, Negative: B%, Neutral: C%.\\n"
                    f"   3.  **Sentiment Trend Analysis**: Explicitly state the trend. For example: 'Positive sentiment changed by K percentage points from A% (in {past_sentiment_period_str}) to X% (in {analysis_period_str}). Negative sentiment changed by M percentage points from B% to Y%.' Calculate and state these changes clearly.\\n"
                    f"   4.  **Key Themes/Hashtags**: Identify 1-2 specific, verifiable key themes or prominent hashtags (e.g., 'MissingCM' if data supports this for the current period) that are significantly driving the current sentiment or trends.\\n"
                    f"   5.  **Reasons for Sentiment Shifts**: Explain the reasons for any observed sentiment shifts in simple terms, linking them directly to specific recent activities, controversies, community impacts, or competitor actions from the respective periods analyzed.",
        agent=sentiment_analyzer,
        context=[task1_research_activities_community, task2_monitor_landscape],
        expected_output=f"A highly detailed sentiment analysis report for {political_entity_name}, precisely following the structure for Section IV of the main political analysis prompt, including:\\n"
                        f"- Current period ({analysis_period_str}) sentiment breakdown (Positive %, Negative %, Neutral %).\\n"
                        f"- Past period ({past_sentiment_period_str}) sentiment breakdown (Positive %, Negative %, Neutral %).\\n"
                        f"- Explicit trend analysis detailing percentage point changes between the two periods for each sentiment category.\\n"
                        f"- Identification of 1-2 key themes or prominent, verifiable hashtags driving current sentiment.\\n"
                        f"- Clear, evidence-based reasons for any sentiment shifts.\\n"
                        f"- The output for this task should conclude with a list of relevant source URLs under 'Collected URLs for this task:', if any were found for sentiment context.",
        async_execution=False
    )
    task4_compile_final_report = Task(
        description=f"**Primary Goal**: Compile all detailed findings from Task 1, Task 2, and Task 3 "
                    f"into a single, final, comprehensive report. The final report **MUST be written in simple, clear English** but must retain all the requested analytical depth and detail. "
                    f"Adhere strictly to the 7-section structure from the main political analysis prompt. The report title should be professional, like 'Comprehensive Political Analysis for {political_entity_name}', and specify the analysis period: {analysis_period_str}.\\n"
                    f"   **Section VI - Strategic Recommendations**: Develop insightful, highly specific, and actionable recommendations for {political_entity_name}. These should directly address the findings from all preceding sections. Aim for recommendations with depth and clear rationale.\\n"
                    f"   **Section VII - Compiled Reference List**: Meticulously compile ALL unique and seemingly valid URLs provided by Task 1, Task 2, and Task 3 into a single, clean, numbered list. "
                    f"Each URL in this list should be presented as a plain URL (e.g., https://www.example.com/article) without any surrounding Markdown link syntax. " # Added instruction for plain URL formatting
                    f"**NO URLs should appear inline within Sections I-V of the main report body.** If no URLs were provided by other agents, state that clearly in this section.",
        agent=report_writer,
        context=[task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment],
        expected_output=f"A final, comprehensive report on {political_entity_name} (analysis period: {analysis_period_str}) written in **simple, clear English**, "
                        f"precisely following the 7-section structure. "
                        f"Section VII must contain the consolidated URL list (formatted as plain URLs, each on a new line) or a note if no relevant URLs were found. "
                        f"The main title should be 'Comprehensive Political Analysis & Strategic Advisory: {political_entity_name} (Analysis Period: {start_date_str} to {current_date_str})'.",
        async_execution=False
    )
    return [task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment, task4_compile_final_report]

# --- Main Streamlit App ---
def run_app():
    st.title("🕵️ PoliScanSight Bot")
    st.markdown("Welcome to **PoliScanSight Bot** 🚀: Your one-stop AI for analyzing recent political activities, controversies, sentiment trends, and strategic suggestions for any Indian political entity from your chosen date.")
    st.markdown("---")

    with st.sidebar:
        st.subheader("📊 Analysis Inputs")
        entity_name_input = st.text_input(
            "👤 Political Entity Name", placeholder="e.g., M.K. Stalin, DMK Party",
            help="Enter the political leader, party, or influencer."
        )
        default_start_date = date.today() - timedelta(days=7)
        start_date_input = st.date_input(
            "🗓️ Analysis Start Date", value=default_start_date, max_value=date.today(),
            help="Select start date. End date is today."
        )
        
        st.subheader("🧠 LLM Selection")
        llm_options = []
        if ChatOpenAI: llm_options.append("GPT-4o-mini (OpenAI)")
        if Ollama: llm_options.append("Llama3.1 (Local Ollama)")
        if not llm_options:
            st.error("No LLMs available! Check Langchain installations.")
            selected_llm = None
        else:
            selected_llm = st.selectbox(
                "Choose Language Model", llm_options,
                help="Select the LLM. Ensure local Ollama server is running if selected."
            )
        
        with st.expander("🔑 API Keys & Web Search Setup (Optional)", expanded=False):
            st.caption("Provide API keys if you want to use OpenAI models or enable web search via Serper.")
            default_openai_key = os.environ.get("OPENAI_API_KEY", "")
            default_serper_key = os.environ.get("SERPER_API_KEY", "")
            
            openai_api_key_input_exp = st.text_input(
                "OpenAI API Key", type="password", value=default_openai_key,
                help="Needed for GPT models. Can also be set as OPENAI_API_KEY environment variable.",
                key="openai_api_key_exp_v2" 
            )
            serper_api_key_input_exp = st.text_input(
                "Serper API Key", type="password", value=default_serper_key,
                help="Needed for web search. Can also be set as SERPER_API_KEY environment variable.",
                key="serper_api_key_exp_v2" 
            )

            if serper_api_key_input_exp: 
                if 'serper_initialized' not in st.session_state or \
                   not st.session_state.get('serper_initialized', False) or \
                   st.session_state.get('serper_key_used') != serper_api_key_input_exp:
                    with st.spinner("Initializing Serper..."):
                        initialize_serper_tool(serper_api_key_input_exp) 
                        st.session_state.serper_key_used = serper_api_key_input_exp 
                        if st.session_state.get('serper_initialized'):
                            st.success("SerperDevTool active!")
            elif not serper_api_key_input_exp and default_serper_key and not st.session_state.get('serper_initialized_with_env', False):
                with st.spinner("Trying Serper with .env key..."):
                    if initialize_serper_tool(default_serper_key):
                        st.session_state.serper_initialized_with_env = True
                        st.session_state.serper_key_used = default_serper_key
                        st.success("SerperDevTool active (using .env key)!")
                    else:
                        st.session_state.serper_initialized_with_env = True 
            elif not serper_api_key_input_exp and not default_serper_key:
                 st.caption("Serper API key not provided. Web search disabled.")
        
        st.markdown("---") 
        generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True, disabled=st.session_state.get('analysis_running', False))

    if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
    if 'analysis_running' not in st.session_state: st.session_state.analysis_running = False

    openai_api_key_to_use = openai_api_key_input_exp if 'openai_api_key_input_exp' in locals() and openai_api_key_input_exp else default_openai_key

    if generate_button: 
        valid_input = True
        if not entity_name_input.strip():
            st.sidebar.error("❌ Entity name required.") 
            valid_input = False
        if not start_date_input:
            st.sidebar.error("❌ Start date required.") 
            valid_input = False
        
        if selected_llm == "GPT-4o-mini (OpenAI)" and not openai_api_key_to_use:
            st.sidebar.error("❌ OpenAI Key needed for GPT.") 
            valid_input = False
        
        serper_key_available = bool(serper_api_key_input_exp or default_serper_key)
        if not st.session_state.get('serper_initialized', False) : 
            if serper_key_available :
                 st.sidebar.warning("⚠️ Serper key issue. Check expander or .env.")
        
        if valid_input and selected_llm:
            st.session_state.analysis_result = None
            st.session_state.analysis_running = True
            st.rerun()

    if st.session_state.analysis_running:
        with st.status("🤖 PoliScanSight Crew at Work...", expanded=True) as status_ui:
            st.write("Initializing Language Model...")
            llm_instance = create_llm(selected_llm, openai_api_key_to_use) 

            if llm_instance:
                st.write(f"🧠 Using {selected_llm}.")
                
                current_serper_key_for_run = serper_api_key_input_exp if 'serper_api_key_input_exp' in locals() and serper_api_key_input_exp else default_serper_key
                if current_serper_key_for_run and not st.session_state.get('serper_initialized', False):
                    st.write("🛠️ Attempting Serper Tool initialization for run...")
                    initialize_serper_tool(current_serper_key_for_run)
                
                if not st.session_state.get('serper_initialized', False) and current_serper_key_for_run:
                     st.write("⚠️ Serper tool could not be initialized. Proceeding with limited/no web search.")
                elif not current_serper_key_for_run:
                     st.write("ℹ️ No Serper API key provided. Web search disabled.")
                else: 
                     st.write("🛠️ Serper tool active.")

                st.write("🛠️ Preparing analysis parameters and agents...")
                user_start_date_obj = start_date_input 
                current_date_obj = date.today()
                current_date_str_param = current_date_obj.strftime("%Y-%m-%d")
                start_date_str_param = user_start_date_obj.strftime("%Y-%m-%d")
                past_sentiment_start_obj = user_start_date_obj - timedelta(days=10)
                past_sentiment_end_obj = user_start_date_obj - timedelta(days=5)
                past_sentiment_start_str_param = past_sentiment_start_obj.strftime("%Y-%m-%d")
                past_sentiment_end_str_param = past_sentiment_end_obj.strftime("%Y-%m-%d")

                agents_list = create_political_agents(entity_name_input, llm_instance) 
                tasks_list = create_political_tasks(
                    entity_name_input, agents_list, start_date_str_param,
                    current_date_str_param, past_sentiment_start_str_param, past_sentiment_end_str_param
                )
                crew_instance = Crew(
                    agents=agents_list, tasks=tasks_list, process=Process.sequential, verbose=True
                )
                st.write(f"🚀 Launching analysis for **{entity_name_input}** from **{start_date_str_param}** to **{current_date_str_param}**...")
                st.caption("Detailed logs appear in terminal if `verbose=True`.")

                final_report_str = "Analysis failed or produced no output."
                try:
                    result = crew_instance.kickoff()
                    if result:
                        if isinstance(result, str): final_report_str = result
                        elif hasattr(result, 'raw') and result.raw: final_report_str = result.raw
                        elif hasattr(result, 'tasks_output') and result.tasks_output:
                             last_task_output = result.tasks_output[-1]
                             if hasattr(last_task_output, 'raw_output'): final_report_str = last_task_output.raw_output
                             elif hasattr(last_task_output, 'result'): final_report_str = str(last_task_output.result)
                             else: final_report_str = str(last_task_output)
                        else: final_report_str = str(result)

                        st.session_state.analysis_result = final_report_str
                        status_ui.update(label="✅ Political Analysis Completed!", state="complete", expanded=False)

                        report_file_name = f"{entity_name_input.replace(' ', '_').lower()}_report_{start_date_str_param}_to_{current_date_str_param}.md"
                        try:
                            with open(report_file_name, "w", encoding="utf-8") as f:
                                f.write(f"# Comprehensive Political Analysis & Strategic Advisory: {entity_name_input} (Analysis Period: {start_date_str_param} to {current_date_str_param})\n\n")
                                f.write(final_report_str)
                            st.info(f"📄 Report also saved locally as: {report_file_name}")
                        except Exception as e:
                            st.error(f"Error saving report to file: {e}")
                    else:
                        st.error("Analysis completed but no output was generated by the crew.")
                        st.session_state.analysis_result = "No output was generated by the crew."
                        status_ui.update(label="⚠️ Analysis completed with no output.", state="error", expanded=True)

                except Exception as e:
                    st.error(f"An error occurred during the analysis: {str(e)}")
                    import traceback
                    st.text_area("Error Traceback:", traceback.format_exc(), height=300)
                    st.session_state.analysis_result = f"Political analysis failed: {str(e)}"
                    status_ui.update(label="❌ Analysis Failed!", state="error", expanded=True)
            else:
                st.error("LLM could not be initialized. Check API keys and LLM selection.")
                status_ui.update(label="❌ LLM Initialization Failed!", state="error", expanded=True)

            st.session_state.analysis_running = False
            st.rerun()

    if st.session_state.analysis_result:
        st.markdown("---") 
        st.header("📜 Generated Report") 
        
        # General Disclaimer about URLs
        st.info(
            "📌 **Note on Source URLs:** This report includes URLs to support its findings. While the AI strives to provide accurate and relevant links, "
            "some URLs may be illustrative, lead to broader context pages, or occasionally be placeholders if a highly specific source for every "
            "synthesized detail could not be pinpointed during the automated generation process. Please verify critical information independently."
        )

        with st.expander("View Full Report", expanded=True):
            st.markdown(st.session_state.analysis_result)
        
        report_data_for_download = f"# Comprehensive Political Analysis & Strategic Advisory: {entity_name_input} (Analysis Period: {start_date_input.strftime('%Y-%m-%d')} to {date.today().strftime('%Y-%m-%d')})\n\n{st.session_state.analysis_result}"
        download_file_name = f"{entity_name_input.replace(' ', '_').lower()}_report_{start_date_input.strftime('%Y-%m-%d')}_to_{date.today().strftime('%Y-%m-%d')}.md"
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
    if check_password():
        run_app()
