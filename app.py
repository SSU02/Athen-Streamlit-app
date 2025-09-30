import streamlit as st
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Optional
import pandas as pd
import json
from sqlalchemy import create_engine
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ---------------------------------------
# Azure SQL Database Connection
# ---------------------------------------

server = st.secrets["az_sql"]["server"]
username = st.secrets["az_sql"]["username"]
password = st.secrets["az_sql"]["password"]
database = st.secrets["az_sql"]["database"]

def get_azure_connection():
    """Get Azure SQL Database connection using SQLAlchemy"""
    try:
        password_encoded = urllib.parse.quote_plus(password)
        connection_string = f'mssql+pyodbc://{username}:{password_encoded}@{server}:1433/{database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=100'
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to Azure SQL Database: {e}")
        return None

# ---------------------------------------
# Date/Time Utility Functions
# ---------------------------------------
def parse_sql_datetime(date_str):
    """Parse SQL Server datetime string to Python datetime"""
    if pd.isna(date_str) or date_str is None:
        return None
    try:
        date_str = str(date_str)
        if '.' in date_str:
            date_str = date_str.split('.')[0]
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except:
        try:
            return datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
        except:
            return None

def parse_time_to_hours(time_str):
    """Parse HH:MM:SS time format to hours (decimal)"""
    if pd.isna(time_str) or time_str is None:
        return 0.0
    
    try:
        time_str = str(time_str).strip()
        
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) >= 2:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2]) if len(parts) > 2 else 0.0
                return hours + (minutes / 60.0) + (seconds / 3600.0)
        else:
            return float(time_str)
    except:
        return 0.0

def get_week_start_end(date):
    """Get Monday and Friday of the week for a given date"""
    start_of_week = date - timedelta(days=date.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=4)  # Friday
    return start_of_week, end_of_week

# ---------------------------------------
# Data Loading Functions
# ---------------------------------------
@st.cache_data(ttl=300)
def load_master_skills():
    """Load master skills table for skill ID to name mapping"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = "SELECT Id as SkillId, SkillName FROM dbo.MasterSkills"
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading master skills: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_users_with_roles():
    """Load users with their roles from Azure SQL Database"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            u.Id as UserId,
            u.FirstName,
            u.LastName,
            u.Email,
            u.CostPerHour,
            u.RoleID,
            r.Name as RoleName
        FROM dbo.Users u
        LEFT JOIN dbo.Roles r ON u.RoleID = r.Id
        """
        
        df = pd.read_sql_query(query, engine)
        df['FullName'] = df['FirstName'].astype(str) + ' ' + df['LastName'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_user_skills():
    """Load user skills mapping with skill names"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            usm.UserId, 
            usm.SkillId,
            ms.SkillName
        FROM dbo.UserSkillMappings usm
        LEFT JOIN dbo.MasterSkills ms ON usm.SkillId = ms.Id
        """
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading user skills: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_projects():
    """Load projects data with enhanced information including ProjectStatusId"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            Id as ProjectId,
            Name as ProjectName,
            Budget,
            StartDate,
            EndDate,
            IsActive,
            ProjectStatusId,
            EstimatedHours
        FROM dbo.Projects
        """
        
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading projects: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_project_skills():
    """Load project required skills with skill names"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            psm.ProjectId, 
            psm.SkillId,
            ms.SkillName
        FROM dbo.ProjectSkillMappings psm
        LEFT JOIN dbo.MasterSkills ms ON psm.SkillId = ms.Id
        """
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading project skills: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_project_assignments():
    """Load project assignments with enhanced details"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            UserId, 
            ProjectId, 
            AvailablePercentage,
            IsActive,
            UserStartDate,
            UserEndDate
        FROM dbo.ProjectAssigneeDetails
        """
        
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading project assignments: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_timesheet_data():
    """Load timesheet data for burnt hours calculation"""
    try:
        engine = get_azure_connection()
        if not engine:
            return pd.DataFrame()
        
        query = """
        SELECT 
            TimeSheetDate,
            TaskId,
            ProjectId,
            UserId,
            BurnedHours
        FROM dbo.TimeSheet
        """
        
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading timesheet data: {e}")
        return pd.DataFrame()

def get_database_info():
    """Get database statistics"""
    try:
        engine = get_azure_connection()
        if not engine:
            return {'error': 'Connection failed'}
        
        stats = {}
        tables = ['Users', 'Roles', 'Projects', 'MasterSkills', 'UserSkillMappings', 'ProjectSkillMappings', 'ProjectAssigneeDetails', 'TimeSheet']
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM dbo.{table}"
            result = pd.read_sql_query(query, engine)
            stats[table] = result.iloc[0]['count']
        
        return stats
    except Exception as e:
        return {'error': str(e)}

# ---------------------------------------
# New Insight Functions
# ---------------------------------------
def get_project_status_insights():
    """Get Overall Project Status insights based on ProjectStatusId"""
    status_map = {1: "Not Started", 2: "Active", 3: "Completed"}
    
    # Filter out projects without ProjectStatusId
    valid_projects = projects_df[projects_df['ProjectStatusId'].notna()]
    
    if valid_projects.empty:
        return {"status_counts": {}, "total_projects": 0, "chart_data": []}
    
    status_counts = valid_projects['ProjectStatusId'].value_counts().to_dict()
    status_distribution = {}
    
    for status_id, count in status_counts.items():
        if status_id in status_map:
            status_distribution[status_map[status_id]] = count
    
    # Prepare chart data
    chart_data = []
    for status, count in status_distribution.items():
        chart_data.append({"Status": status, "Count": count})
    
    return {
        "status_counts": status_distribution,
        "total_projects": len(valid_projects),
        "chart_data": chart_data
    }

def get_project_overrun_insights():
    """Get Project Overrun insights by comparing EstimatedHours with actual BurnedHours"""
    overrun_projects = []
    
    if timesheet_df.empty or projects_df.empty:
        return {"overrun_projects": [], "total_analyzed": 0, "chart_data": []}
    
    # Calculate actual hours per project
    timesheet_df['BurnedHours_Float'] = timesheet_df['BurnedHours'].apply(parse_time_to_hours)
    actual_hours_per_project = timesheet_df.groupby('ProjectId')['BurnedHours_Float'].sum()
    
    for project_id, actual_hours in actual_hours_per_project.items():
        project_info = projects_df[projects_df['ProjectId'] == project_id]
        if not project_info.empty and pd.notna(project_info.iloc[0]['EstimatedHours']):
            estimated_hours = float(project_info.iloc[0]['EstimatedHours'])
            project_name = str(project_info.iloc[0]['ProjectName'])
            
            overrun_hours = float(actual_hours) - estimated_hours
            overrun_percentage = (overrun_hours / estimated_hours * 100) if estimated_hours > 0 else 0
            
            overrun_projects.append({
                "ProjectId": int(project_id),
                "ProjectName": project_name,
                "EstimatedHours": estimated_hours,
                "ActualHours": round(float(actual_hours), 2),
                "OverrunHours": round(overrun_hours, 2),
                "OverrunPercentage": round(overrun_percentage, 1),
                "IsOverrun": bool(overrun_hours > 0)
            })
    
    # Separate overrun and on-track projects for chart
    overrun_count = sum(1 for p in overrun_projects if p["IsOverrun"])
    on_track_count = len(overrun_projects) - overrun_count
    
    chart_data = [
        {"Status": "Overrun", "Count": overrun_count},
        {"Status": "On Track/Under Budget", "Count": on_track_count}
    ]
    
    return {
        "overrun_projects": overrun_projects,
        "total_analyzed": len(overrun_projects),
        "overrun_count": overrun_count,
        "on_track_count": on_track_count,
        "chart_data": chart_data
    }

def get_resource_utilization_insights():
    """Get Resource Utilization Rate by comparing logged hours with 40-hour work week"""
    now = datetime.now()
    current_month_start = now.replace(day=1)
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end.replace(day=1)
    
    if timesheet_df.empty:
        return {"current_month": [], "previous_month": [], "chart_data": []}
    
    # Convert TimeSheetDate to datetime
    timesheet_df['TimeSheetDate'] = pd.to_datetime(timesheet_df['TimeSheetDate'])
    timesheet_df['BurnedHours_Float'] = timesheet_df['BurnedHours'].apply(parse_time_to_hours)
    
    def analyze_month_utilization(start_date, end_date):
        month_data = timesheet_df[
            (timesheet_df['TimeSheetDate'] >= start_date) & 
            (timesheet_df['TimeSheetDate'] <= end_date)
        ]
        
        if month_data.empty:
            return []
        
        user_utilization = []
        user_hours = month_data.groupby('UserId')['BurnedHours_Float'].sum()
        
        # Calculate working days in the month (Monday to Friday)
        current_date = start_date
        working_days = 0
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                working_days += 1
            current_date += timedelta(days=1)
        
        expected_hours = working_days * 8  # 8 hours per working day
        
        for user_id, actual_hours in user_hours.items():
            user_info = users_df[users_df['UserId'] == user_id]
            if not user_info.empty:
                user_name = f"{user_info.iloc[0]['FirstName']} {user_info.iloc[0]['LastName']}"
                utilization_rate = (float(actual_hours) / expected_hours * 100) if expected_hours > 0 else 0
                
                user_utilization.append({
                    "UserId": int(user_id),
                    "UserName": str(user_name),
                    "ActualHours": round(float(actual_hours), 2),
                    "ExpectedHours": expected_hours,
                    "UtilizationRate": round(utilization_rate, 1),
                    "IsUnderUtilized": bool(utilization_rate < 100),
                    "UnderUtilizedHours": round(max(0, expected_hours - float(actual_hours)), 2)
                })
        
        return user_utilization
    
    current_month_data = analyze_month_utilization(current_month_start, now)
    previous_month_data = analyze_month_utilization(previous_month_start, previous_month_end)
    
    # Chart data for current month
    current_underutilized = sum(1 for u in current_month_data if u["IsUnderUtilized"])
    current_optimal = len(current_month_data) - current_underutilized
    
    chart_data = [
        {"Status": "Under-Utilized", "Count": current_underutilized, "Month": "Current"},
        {"Status": "Optimal/Over-Utilized", "Count": current_optimal, "Month": "Current"}
    ]
    
    return {
        "current_month": current_month_data,
        "previous_month": previous_month_data,
        "chart_data": chart_data,
        "current_underutilized_count": current_underutilized,
        "current_total_users": len(current_month_data)
    }

def get_resource_allocation_insights():
    """Get Resource Allocation Rate from ProjectAssigneeDetails"""
    if project_assignments_df.empty:
        return {"allocation_data": [], "chart_data": []}
    
    # Get active assignments only
    active_assignments = project_assignments_df[project_assignments_df['IsActive'] == 1]
    
    if active_assignments.empty:
        return {"allocation_data": [], "chart_data": []}
    
    # Calculate allocation per user
    user_allocation = active_assignments.groupby('UserId')['AvailablePercentage'].sum()
    
    allocation_analysis = []
    for user_id, total_allocation in user_allocation.items():
        user_info = users_df[users_df['UserId'] == user_id]
        if not user_info.empty:
            user_name = f"{user_info.iloc[0]['FirstName']} {user_info.iloc[0]['LastName']}"
            
            # Count projects for this user
            user_projects = active_assignments[active_assignments['UserId'] == user_id]
            project_count = len(user_projects)
            
            allocation_analysis.append({
                "UserId": int(user_id),
                "UserName": str(user_name),
                "TotalAllocation": float(total_allocation),
                "ProjectCount": int(project_count),
                "IsUnderAllocated": bool(total_allocation < 100),
                "IsOverAllocated": bool(total_allocation > 100),
                "AllocationStatus": (
                    "Over-Allocated" if total_allocation > 100 
                    else "Under-Allocated" if total_allocation < 100 
                    else "Fully Allocated"
                )
            })
    
    # Chart data
    under_allocated = sum(1 for u in allocation_analysis if u["IsUnderAllocated"])
    fully_allocated = sum(1 for u in allocation_analysis if u["TotalAllocation"] == 100)
    over_allocated = sum(1 for u in allocation_analysis if u["IsOverAllocated"])
    
    chart_data = [
        {"Status": "Under-Allocated", "Count": under_allocated},
        {"Status": "Fully Allocated", "Count": fully_allocated},
        {"Status": "Over-Allocated", "Count": over_allocated}
    ]
    
    return {
        "allocation_data": allocation_analysis,
        "chart_data": chart_data,
        "under_allocated_count": under_allocated,
        "total_users": len(allocation_analysis)
    }

# Load all data
try:
    master_skills_df = load_master_skills()
    users_df = load_users_with_roles()
    user_skills_df = load_user_skills()
    projects_df = load_projects()
    project_skills_df = load_project_skills()
    project_assignments_df = load_project_assignments()
    timesheet_df = load_timesheet_data()
    
    if users_df.empty:
        st.error("Failed to load user data from Azure SQL Database")
        st.stop()
except Exception as e:
    st.error(f"Error loading data from Azure SQL Database: {e}")
    st.stop()

# ---------------------------------------
# Helper Functions
# ---------------------------------------
def get_skill_name(skill_id):
    """Get skill name from skill ID"""
    if master_skills_df.empty:
        return f"Skill ID {skill_id}"
    
    skill_row = master_skills_df[master_skills_df['SkillId'] == skill_id]
    if not skill_row.empty:
        return skill_row.iloc[0]['SkillName']
    return f"Skill ID {skill_id}"

def get_user_skills(user_id):
    """Get skills for a specific user with names"""
    user_skills = user_skills_df[user_skills_df['UserId'] == user_id]
    skills = []
    
    for _, skill in user_skills.iterrows():
        if pd.notna(skill['SkillId']):
            skill_name = skill['SkillName'] if pd.notna(skill['SkillName']) else f"Skill ID {skill['SkillId']}"
            skills.append({
                'SkillId': skill['SkillId'],
                'SkillName': skill_name
            })
    
    return skills

def get_users_by_skills(required_skills: List[str], top_k: int = 10):
    """Find users who have required skills (by name or ID)"""
    required_skill_ids = []
    
    for skill in required_skills:
        if skill.isdigit():
            required_skill_ids.append(int(skill))
        else:
            skill_match = master_skills_df[master_skills_df['SkillName'].str.contains(skill, case=False, na=False)]
            if not skill_match.empty:
                required_skill_ids.extend(skill_match['SkillId'].tolist())
    
    if not required_skill_ids:
        return []
    
    user_scores = []
    
    for _, user in users_df.iterrows():
        user_id = user['UserId']
        user_skills = get_user_skills(user_id)
        user_skill_ids = [skill['SkillId'] for skill in user_skills]
        
        matching_skill_ids = [skill_id for skill_id in required_skill_ids if skill_id in user_skill_ids]
        skill_match_score = len(matching_skill_ids) / len(required_skill_ids) if required_skill_ids else 0
        
        current_projects = project_assignments_df[
            (project_assignments_df['UserId'] == user_id) & 
            (project_assignments_df['IsActive'] == 1)
        ]
        
        workload_score = max(0, (5 - len(current_projects)) / 5)
        overall_score = 0.8 * skill_match_score + 0.2 * workload_score
        
        matching_skills = [get_skill_name(skill_id) for skill_id in matching_skill_ids]
        
        user_scores.append({
            'UserId': user_id,
            'FullName': f"{user['FirstName']} {user['LastName']}",
            'FirstName': user['FirstName'],
            'LastName': user['LastName'],
            'Email': user['Email'],
            'RoleName': user['RoleName'],
            'CostPerHour': user['CostPerHour'],
            'Skills': [skill['SkillName'] for skill in user_skills],
            'MatchingSkills': matching_skills,
            'SkillMatchPercent': round(skill_match_score * 100, 1),
            'CurrentProjects': len(current_projects),
            'OverallScore': round(overall_score, 3)
        })
    
    user_scores.sort(key=lambda x: x['OverallScore'], reverse=True)
    return user_scores[:top_k]

# ---------------------------------------
# LangChain Tools
# ---------------------------------------
@tool
def find_users_for_project(task_description: str, required_skills: List[str], num_recommendations: int = 5) -> str:
    """Find users with specific skills for a project or task."""
    recommendations = get_users_by_skills(required_skills, num_recommendations)
    
    if not recommendations:
        return json.dumps({
            "status": "no_candidates",
            "message": "No suitable users found with matching skills",
            "task": task_description,
            "required_skills": required_skills
        })
    
    return json.dumps({
        "status": "success",
        "task": task_description,
        "required_skills": required_skills,
        "recommendations": recommendations
    })

@tool
def get_project_overrun_details(project_name: str = None) -> str:
    """Get detailed information about project overruns."""
    overrun_data = get_project_overrun_insights()
    
    if project_name:
        # Filter by project name
        filtered_projects = [
            p for p in overrun_data["overrun_projects"] 
            if project_name.lower() in p["ProjectName"].lower()
        ]
    else:
        filtered_projects = overrun_data["overrun_projects"]
    
    return json.dumps({
        "status": "success",
        "overrun_projects": filtered_projects,
        "total_analyzed": overrun_data["total_analyzed"],
        "overrun_count": overrun_data["overrun_count"]
    })

@tool
def get_resource_utilization_details(month: str = "current") -> str:
    """Get detailed resource utilization information for current or previous month."""
    utilization_data = get_resource_utilization_insights()
    
    if month.lower() == "previous":
        data = utilization_data["previous_month"]
    else:
        data = utilization_data["current_month"]
    
    return json.dumps({
        "status": "success",
        "month": month,
        "user_utilization": data,
        "underutilized_count": sum(1 for u in data if u["IsUnderUtilized"]),
        "total_users": len(data)
    })

@tool
def get_resource_allocation_details() -> str:
    """Get detailed resource allocation information."""
    allocation_data = get_resource_allocation_insights()
    
    return json.dumps({
        "status": "success",
        "allocation_data": allocation_data["allocation_data"],
        "under_allocated_count": allocation_data["under_allocated_count"],
        "total_users": allocation_data["total_users"]
    })

@tool
def get_project_status_details(status: str = "all") -> str:
    """Get detailed information about projects by their status.
    
    Args:
        status: Project status to filter by - 'active', 'completed', 'not_started', or 'all'
    
    Returns:
        JSON string with project status details
    """
    if projects_df.empty:
        return json.dumps({
            "status": "error",
            "message": "No project data available"
        })
    
    # Filter projects with valid ProjectStatusId
    valid_projects = projects_df[projects_df['ProjectStatusId'].notna()].copy()
    
    if valid_projects.empty:
        return json.dumps({
            "status": "no_data",
            "message": "No projects with valid status found"
        })
    
    # Status mapping
    status_map = {1: "Not Started", 2: "Active", 3: "Completed"}
    
    # Filter by requested status
    if status.lower() == "active":
        filtered_projects = valid_projects[valid_projects['ProjectStatusId'] == 2]
    elif status.lower() == "completed":
        filtered_projects = valid_projects[valid_projects['ProjectStatusId'] == 3]
    elif status.lower() == "not_started":
        filtered_projects = valid_projects[valid_projects['ProjectStatusId'] == 1]
    else:
        filtered_projects = valid_projects
    
    project_details = []
    
    for _, project in filtered_projects.iterrows():
        project_id = int(project['ProjectId'])
        
        # Get assigned users
        assigned_users = project_assignments_df[project_assignments_df['ProjectId'] == project_id]
        active_users = []
        
        for _, assignment in assigned_users.iterrows():
            if assignment['IsActive'] == 1:
                user_info = users_df[users_df['UserId'] == assignment['UserId']]
                if not user_info.empty:
                    user_name = f"{user_info.iloc[0]['FirstName']} {user_info.iloc[0]['LastName']}"
                    active_users.append({
                        "UserName": user_name,
                        "Allocation": float(assignment['AvailablePercentage'])
                    })
        
        # Get required skills
        required_skills = project_skills_df[project_skills_df['ProjectId'] == project_id]
        skill_names = [str(skill_name) for skill_name in required_skills['SkillName'].tolist() if pd.notna(skill_name)]
        
        # Calculate actual hours if available
        actual_hours = 0
        if not timesheet_df.empty:
            project_timesheet = timesheet_df[timesheet_df['ProjectId'] == project_id]
            if not project_timesheet.empty:
                project_timesheet['BurnedHours_Float'] = project_timesheet['BurnedHours'].apply(parse_time_to_hours)
                actual_hours = round(float(project_timesheet['BurnedHours_Float'].sum()), 2)
        
        project_details.append({
            "ProjectId": project_id,
            "ProjectName": str(project['ProjectName']),
            "ProjectStatus": status_map.get(int(project['ProjectStatusId']), "Unknown"),
            "Budget": float(project['Budget']) if pd.notna(project['Budget']) else 0,
            "EstimatedHours": float(project['EstimatedHours']) if pd.notna(project['EstimatedHours']) else 0,
            "ActualHours": actual_hours,
            "StartDate": str(project['StartDate']) if pd.notna(project['StartDate']) else None,
            "EndDate": str(project['EndDate']) if pd.notna(project['EndDate']) else None,
            "RequiredSkills": skill_names,
            "ActiveUsers": active_users,
            "ActiveUserCount": len(active_users)
        })
    
    # Sort by status and then by name
    project_details.sort(key=lambda x: (x['ProjectStatus'], x['ProjectName']))
    
    return json.dumps({
        "status": "success",
        "requested_status": status,
        "projects": project_details,
        "total_projects": len(project_details),
        "projects_by_status": {
            status: len([p for p in project_details if p['ProjectStatus'] == status])
            for status in ["Not Started", "Active", "Completed"]
        }
    })

# ---------------------------------------
# LLM Agent Setup
# ---------------------------------------
def create_agent():
    """Create LangChain agent with Azure OpenAI"""
    system_prompt = """You are an intelligent HR and Project Management assistant with access to comprehensive database insights and analytics.

Your Tools:
1. find_users_for_project: Find users with specific skills for projects or tasks
2. get_project_overrun_details: Get detailed information about projects that have exceeded estimated hours
3. get_resource_utilization_details: Get information about user work hour utilization (current/previous month)  
4. get_resource_allocation_details: Get information about how users are allocated across projects
5. get_project_status_details: Get detailed information about projects by status (active, completed, not_started, or all)

Database Insights Available:
- Project Status Analysis: Projects categorized as Not Started, Active, or Completed
- Project Overrun Analysis: Projects where actual hours exceed estimated hours
- Resource Utilization: How well users are utilizing their 40-hour work weeks
- Resource Allocation: How users are allocated across projects (under/over/fully allocated)

Provide actionable insights and recommendations based on the data."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Azure OpenAI Configuration
    llm = AzureChatOpenAI(
        azure_endpoint=st.secrets["azure_openai"]["endpoint"],  
        api_key=st.secrets["azure_openai"]["api_key"],
        api_version=st.secrets["azure_openai"]["api_version"],  
        deployment_name=st.secrets["azure_openai"]["deployment_name"],  
        temperature=0.1
    )
    
    tools = [
        find_users_for_project, 
        get_project_overrun_details, 
        get_resource_utilization_details, 
        get_resource_allocation_details, 
        get_project_status_details
    ]
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# ---------------------------------------
# Display Functions
# ---------------------------------------
def display_insights_charts():
    """Display the four main insight charts"""
    
    # 1. Overall Project Status
    st.subheader("📊 Overall Project Status")
    project_status_data = get_project_status_insights()
    
    if project_status_data["chart_data"]:
        fig1 = px.pie(
            values=[item["Count"] for item in project_status_data["chart_data"]], 
            names=[item["Status"] for item in project_status_data["chart_data"]],
            title="Project Status Distribution",
            color_discrete_map={
                "Not Started": "#ffc107",
                "Active": "#28a745", 
                "Completed": "#6c757d"
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        for status, count in project_status_data["status_counts"].items():
            if status == "Not Started":
                col1.metric("Not Started", count)
            elif status == "Active":
                col2.metric("Active", count)  
            elif status == "Completed":
                col3.metric("Completed", count)
    else:
        st.info("No project status data available")
    
    # 2. Project Overrun
    st.subheader("⚠️ Project Overrun Analysis")
    overrun_data = get_project_overrun_insights()
    
    if overrun_data["chart_data"]:
        fig2 = px.bar(
            x=[item["Status"] for item in overrun_data["chart_data"]],
            y=[item["Count"] for item in overrun_data["chart_data"]],
            title="Project Budget Performance",
            color=[item["Status"] for item in overrun_data["chart_data"]],
            color_discrete_map={
                "Overrun": "#dc3545",
                "On Track/Under Budget": "#28a745"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        col1, col2 = st.columns(2)
        col1.metric("Projects Over Budget", overrun_data["overrun_count"])
        col2.metric("Projects On Track", overrun_data["on_track_count"])
    else:
        st.info("No project overrun data available")
    
    # 3. Resource Utilization Rate  
    st.subheader("📈 Resource Utilization Rate")
    utilization_data = get_resource_utilization_insights()
    
    if utilization_data["chart_data"]:
        fig3 = px.bar(
            x=[item["Status"] for item in utilization_data["chart_data"]],
            y=[item["Count"] for item in utilization_data["chart_data"]],
            title="User Utilization (Current Month)",
            color=[item["Status"] for item in utilization_data["chart_data"]],
            color_discrete_map={
                "Under-Utilized": "#ffc107",
                "Optimal/Over-Utilized": "#28a745"
            }
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        col1, col2 = st.columns(2)
        col1.metric("Under-Utilized Users", utilization_data["current_underutilized_count"])
        col2.metric("Total Active Users", utilization_data["current_total_users"])
    else:
        st.info("No resource utilization data available")
    
    # 4. Resource Allocation Rate
    st.subheader("🎯 Resource Allocation Rate")
    allocation_data = get_resource_allocation_insights()
    
    if allocation_data["chart_data"]:
        fig4 = px.bar(
            x=[item["Status"] for item in allocation_data["chart_data"]],
            y=[item["Count"] for item in allocation_data["chart_data"]],
            title="Resource Allocation Distribution",
            color=[item["Status"] for item in allocation_data["chart_data"]],
            color_discrete_map={
                "Under-Allocated": "#ffc107",
                "Fully Allocated": "#28a745",
                "Over-Allocated": "#dc3545"
            }
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        for item in allocation_data["chart_data"]:
            if item["Status"] == "Under-Allocated":
                col1.metric("Under-Allocated", item["Count"])
            elif item["Status"] == "Fully Allocated":
                col2.metric("Fully Allocated", item["Count"])
            elif item["Status"] == "Over-Allocated":
                col3.metric("Over-Allocated", item["Count"])
    else:
        st.info("No resource allocation data available")

def extract_structured_data(response):
    """Extract structured data from agent response"""
    try:
        for step in response.get("intermediate_steps", []):
            if len(step) > 1 and isinstance(step[1], str):
                try:
                    data = json.loads(step[1])
                    if data.get("status") == "success":
                        return data
                except:
                    continue
        return None
    except:
        return None

def display_response_data(data):
    """Display different types of response data"""
    if "recommendations" in data:
        display_user_recommendations(data)
    elif "overrun_projects" in data:
        display_overrun_details(data)
    elif "user_utilization" in data:
        display_utilization_details(data)
    elif "allocation_data" in data:
        display_allocation_details(data)


def display_user_recommendations(data):
    """Display user recommendations"""
    st.success(f"Found {len(data['recommendations'])} suitable candidates")
    
    st.write("**Task Details:**")
    st.write(f"**Task:** {data['task']}")
    st.write(f"**Required Skills:** {', '.join(data['required_skills'])}")
    
    if data["recommendations"]:
        df = pd.DataFrame(data["recommendations"])
        
        display_df = df[[
            'FullName', 'RoleName', 'CostPerHour',
            'SkillMatchPercent', 'CurrentProjects', 'OverallScore'
        ]].copy()
        
        display_df.columns = [
            'Name', 'Role', 'Cost/Hour ($)',
            'Skill Match %', 'Active Projects', 'Score'
        ]
        
        st.write("### User Recommendations")
        st.dataframe(display_df, use_container_width=True)
        
        top_candidate = data["recommendations"][0]
        st.write("### Top Recommendation")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{top_candidate['FullName']}** - {top_candidate['RoleName']}")
            st.write(f"Email: {top_candidate['Email']}")
            st.write(f"**All Skills:** {', '.join(top_candidate['Skills'])}")
        with col2:
            st.metric("Skill Match", f"{top_candidate['SkillMatchPercent']}%")
            st.metric("Cost per Hour", f"${top_candidate['CostPerHour']}")
            st.metric("Active Projects", top_candidate['CurrentProjects'])
        
        if top_candidate['MatchingSkills']:
            st.write(f"**Matching Skills:** {', '.join(top_candidate['MatchingSkills'])}")

def display_overrun_details(data):
    """Display project overrun details"""
    st.success(f"Analyzed {data['total_analyzed']} projects, {data['overrun_count']} are over budget")
    
    if data["overrun_projects"]:
        df = pd.DataFrame(data["overrun_projects"])
        
        # Show overrun projects first
        overrun_df = df[df['IsOverrun'] == True].copy()
        if not overrun_df.empty:
            st.write("### Projects Over Budget")
            display_overrun_df = overrun_df[['ProjectName', 'EstimatedHours', 'ActualHours', 'OverrunHours', 'OverrunPercentage']].copy()
            display_overrun_df.columns = ['Project', 'Estimated Hours', 'Actual Hours', 'Overrun Hours', 'Overrun %']
            st.dataframe(display_overrun_df, use_container_width=True)
        
        # Show on-track projects
        on_track_df = df[df['IsOverrun'] == False].copy()
        if not on_track_df.empty:
            with st.expander("Projects On Track/Under Budget"):
                display_on_track_df = on_track_df[['ProjectName', 'EstimatedHours', 'ActualHours', 'OverrunHours']].copy()
                display_on_track_df.columns = ['Project', 'Estimated Hours', 'Actual Hours', 'Hours Saved']
                st.dataframe(display_on_track_df, use_container_width=True)

def display_utilization_details(data):
    """Display resource utilization details"""
    st.success(f"Resource utilization for {data['month']} month")
    
    if data["user_utilization"]:
        df = pd.DataFrame(data["user_utilization"])
        
        # Show under-utilized users first
        under_df = df[df['IsUnderUtilized'] == True].copy()
        if not under_df.empty:
            st.write("### Under-Utilized Users")
            display_under_df = under_df[['UserName', 'ActualHours', 'ExpectedHours', 'UtilizationRate', 'UnderUtilizedHours']].copy()
            display_under_df.columns = ['User', 'Actual Hours', 'Expected Hours', 'Utilization %', 'Unused Hours']
            st.dataframe(display_under_df, use_container_width=True)
        
        # Show optimal/over-utilized users
        optimal_df = df[df['IsUnderUtilized'] == False].copy()
        if not optimal_df.empty:
            with st.expander("Optimal/Over-Utilized Users"):
                display_optimal_df = optimal_df[['UserName', 'ActualHours', 'ExpectedHours', 'UtilizationRate']].copy()
                display_optimal_df.columns = ['User', 'Actual Hours', 'Expected Hours', 'Utilization %']
                st.dataframe(display_optimal_df, use_container_width=True)

def display_allocation_details(data):
    """Display resource allocation details"""
    st.success(f"Resource allocation analysis for {data['total_users']} users")
    
    if data["allocation_data"]:
        df = pd.DataFrame(data["allocation_data"])
        
        # Show under-allocated users
        under_df = df[df['IsUnderAllocated'] == True].copy()
        if not under_df.empty:
            st.write("### Under-Allocated Users")
            display_under_df = under_df[['UserName', 'TotalAllocation', 'ProjectCount', 'AllocationStatus']].copy()
            display_under_df.columns = ['User', 'Total Allocation %', 'Active Projects', 'Status']
            st.dataframe(display_under_df, use_container_width=True)
        
        # Show other allocation statuses
        other_df = df[df['IsUnderAllocated'] == False].copy()
        if not other_df.empty:
            with st.expander("Fully/Over-Allocated Users"):
                display_other_df = other_df[['UserName', 'TotalAllocation', 'ProjectCount', 'AllocationStatus']].copy()
                display_other_df.columns = ['User', 'Total Allocation %', 'Active Projects', 'Status']
                st.dataframe(display_other_df, use_container_width=True)

# ---------------------------------------
# Main Streamlit App
# ---------------------------------------
def main():
    st.title("HR & Project Management Analytics Dashboard")
    st.write("Comprehensive insights and user recommendations with advanced database analytics!")
    
    # Show database info in sidebar
    with st.sidebar:
        st.header("Database Connection")
        
        db_info = get_database_info()
        if 'error' in db_info:
            st.error(f"Database error: {db_info['error']}")
        else:
            st.success("Connected to Azure SQL Database")
            for table, count in db_info.items():
                st.write(f"**{table}:** {count} records")
                
        st.header("Team Overview")
        if not users_df.empty:
            role_counts = users_df['RoleName'].value_counts()
            st.write("**By Role:**")
            for role, count in role_counts.items():
                if pd.notna(role):
                    st.write(f"- {role}: {count}")
        
        st.write("---")
        st.write("**Try asking:**")
        st.write("""
        **Find Users:**
        - "Find Azure developers"
        - "Who has CI CD skills?"
        
        **Analytics:**
        - "Show me project overrun details"
        - "Which users are under-utilized?"
        - "How are resources allocated?"
        - "What are all the active projects?"
        - "Show me completed projects"
        - "List projects that are not started"
        """)
    
    # Display insights charts
    with st.expander("Database Analytics Dashboard", expanded=True):
        display_insights_charts()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("Initializing assistant..."):
            st.session_state.agent = create_agent()
        st.success("Assistant ready! Ask me anything about users, projects, or analytics.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "data" in message:
                display_response_data(message["data"])
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about users, projects, overruns, utilization, or allocation..."):
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                try:
                    response = st.session_state.agent.invoke({
                        "input": prompt,
                        "chat_history": []
                    })
                    
                    output = response["output"]
                    
                    # Try to extract and display structured data
                    structured_data = extract_structured_data(response)
                    if structured_data:
                        display_response_data(structured_data)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": output,
                            "data": structured_data
                        })
                    else:
                        st.write(output)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output
                        })
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.write("Please try rephrasing your request or ensure your API key is set correctly.")

if __name__ == "__main__":
    main()