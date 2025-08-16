# app.py
import streamlit as st
from pathlib import Path
import os
import pandas as pd
import io
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.arxiv import ArxivTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.pandas import PandasTools
from agno.tools.hackernews import HackerNewsTools

# --- Load environment variables ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# --- Setup download folder for Arxiv ---
arxiv_download_dir = Path(__file__).parent / "tmp"
arxiv_download_dir.mkdir(parents=True, exist_ok=True)

# --- Define Agents ---
News_Analyst = Agent(
    name="News Analyst",
    role="Find recent news on sustainability initiatives",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Search for city-level green projects in the past year",
    show_tool_calls=True,
    markdown=True,
)

Data_Analyst = Agent(
    name="Data Analyst",
    role="Analyze uploaded CSV datasets using comprehensive data analysis",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[],
    instructions="""Analyze uploaded CSV data using the provided dataset context.
    
    When analyzing data:
    1. Use the comprehensive dataset information provided in the context
    2. Provide detailed statistical analysis and insights
    3. Identify trends, patterns, and correlations in the data
    4. Detect anomalies, outliers, and data quality issues
    5. Suggest actionable recommendations based on findings
    6. Use markdown formatting for clear presentation with tables and lists
    
    Focus on providing thorough data analysis with practical insights and recommendations.""",
    show_tool_calls=True,
    markdown=True,
)

Policy_Reviewer = Agent(
    name="Policy Reviewer",
    role="Summarize government policies",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Search official sites for city policy updates",
    show_tool_calls=True,
    markdown=True,
)

Innovations_Scout = Agent(
    name="Innovations Scout",
    role="Find innovative green tech ideas",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools(), HackerNewsTools()],
    instructions="Search for “urban sustainability tech”",
    show_tool_calls=True,
    markdown=True,
)

# --- Team Agent (All working together) ---
discussion_team = Team(
    name="Discussion Team",
    mode="collaborate",
    model=Groq(id="qwen/qwen3-32b"),
    members=[News_Analyst, Data_Analyst, Policy_Reviewer, Innovations_Scout],
    instructions=["You are a discussion master. Stop when consensus is reached."],
    show_tool_calls=True,
    markdown=True,
)

# --- Helper Functions for Data Analysis ---
def create_data_analysis_context(df, topic):
    """Create a comprehensive data analysis context"""
    try:
        # Basic statistics
        stats = df.describe()
        
        # Data shape and info
        shape_info = f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns"
        
        # Column information
        columns_info = f"Columns: {', '.join(df.columns.tolist())}"
        
        # Data types
        dtypes_info = f"Data types: {dict(df.dtypes)}"
        
        # Missing values
        missing_info = f"Missing values: {df.isnull().sum().to_dict()}"
        
        # Sample data
        sample_data = df.head(10).to_string()
        
        # Correlation matrix for numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().to_string()
            correlation_info = f"\n## Correlation Matrix\n{corr_matrix}"
        else:
            correlation_info = "\n## Correlation Matrix\nNot enough numerical columns for correlation analysis."
        
        # Create comprehensive context
        context = f"""
## Dataset Overview
{shape_info}
{columns_info}
{dtypes_info}
{missing_info}

## Statistical Summary
{stats.to_string()}

## Sample Data (First 10 rows)
{sample_data}
{correlation_info}

## Analysis Request
User wants to analyze: {topic}

Please provide comprehensive insights based on this data including:
- Key trends and patterns
- Statistical relationships
- Data quality assessment
- Anomalies or outliers
- Actionable recommendations
"""
        return context
    except Exception as e:
        return f"Error analyzing data: {str(e)}"



# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Agent Research Tool", page_icon="🤖")
st.title("🤖 Multi-Agent Research Discussion Tool")

# Sidebar agent selection
st.sidebar.header("Agent Selection")
agent_choice = st.sidebar.radio(
    "Choose which agent to use:",
    ("News Analyst", "Data Analyst", "Policy Reviewer", "Innovations Scout", "All Agents (Team)")
)

# Map user choice to the actual agent
if agent_choice == "News Analyst":
    selected_agent = News_Analyst
elif agent_choice == "Data Analyst":
    selected_agent = Data_Analyst
elif agent_choice == "Policy Reviewer":
    selected_agent = Policy_Reviewer
elif agent_choice == "Innovations Scout":
    selected_agent = Innovations_Scout
else:
    selected_agent = discussion_team

# CSV Upload for Data Analysis
uploaded_file = None
if agent_choice == "Data Analyst":
    st.header("📊 Data Upload")
    
    # Sample data download option
    sample_data = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-01', '2025-01-02', '2025-01-03'],
        'City': ['New York', 'New York', 'New York', 'Los Angeles', 'Los Angeles', 'Los Angeles'],
        'PM2.5': [12.5, 15.2, 18.7, 8.9, 11.3, 14.6],
        'PM10': [25.3, 28.7, 32.1, 18.5, 21.2, 24.8],
        'NO2': [45.2, 48.9, 52.4, 38.7, 41.5, 44.9],
        'O3': [32.1, 35.6, 38.2, 42.3, 45.8, 48.1],
        'Temperature': [15.5, 12.3, 10.1, 22.1, 20.8, 19.2],
        'Humidity': [65, 70, 75, 45, 50, 55]
    })
    
    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_sample,
        file_name="sample_air_quality.csv",
        mime="text/csv",
        help="Download a sample CSV file to test the data analysis functionality"
    )
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file for analysis",
        type=['csv'],
        help="Upload your CSV file here for data analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temp directory
            temp_dir = Path("tmp")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file_path = temp_dir / uploaded_file.name
            
            # Save the uploaded file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read and validate the file
            df = pd.read_csv(temp_file_path, encoding='utf-8')
            
            if df.empty:
                st.error("❌ The CSV file contains no data rows.")
            elif df.shape[1] == 0:
                st.error("❌ The CSV file has no columns.")
            else:
                st.success(f"✅ Successfully uploaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
                st.info(f"📁 File saved as: {temp_file_path}")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head())
                    st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                    
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {e}")
            uploaded_file = None

# User input
topic = st.text_area(
    "Enter your discussion topic:",
    placeholder="Example: What is the best way to learn to code?"
)

# Run button
if st.button("Run Research"):
    if topic.strip():
        with st.spinner(f"Running {agent_choice}... please wait"):
            try:
                # Special handling for Data Analyst with RAG
                if agent_choice == "Data Analyst":
                    if uploaded_file is not None:
                        # Read the uploaded file
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                        
                        # Create comprehensive data analysis context
                        data_context = create_data_analysis_context(df, topic)
                        
                        # Run analysis with context
                        enhanced_topic = f"""
                        {data_context}
                        
                        Please provide a comprehensive analysis of this data focusing on the user's request: {topic}
                        """
                        
                        result = selected_agent.run(enhanced_topic)
                    else:
                        st.error("❌ Please upload a CSV file first for data analysis.")
                        st.stop()
                else:
                    # Regular agent execution
                    result = selected_agent.run(topic)

                # Display clean markdown output
                if result and hasattr(result, "content"):
                    st.markdown(result.content)
                else:
                    st.warning("No content returned from the agent.")

            except Exception as e:
                st.error(f"Error: {e}")
                # Provide more helpful error information
                if "tool call validation failed" in str(e):
                    st.info("💡 Tip: This might be a tool configuration issue. Try using a different agent.")
                elif "rate limit" in str(e).lower():
                    st.info("💡 Tip: Rate limit reached. Please wait a moment and try again.")
    else:
        st.warning("Please enter a topic first.")