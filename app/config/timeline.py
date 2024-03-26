import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Create tasks with a new 'Group' column
df = pd.DataFrame([
    
    {'Task': 'Initiate Project: Define Objectives and Scope', 'duration': 5, 'Group': 'Project Start'},
    {'Task': 'Explore Dataset: Attributes and Characteristics', 'duration': 6, 'Group': 'Data Exploration'},
    {'Task': 'Prepare Data: Cleaning and Transformation', 'duration': 3, 'Group': 'Data Packaging'},
    {'Task': 'Refine Models: Shortlist and Train Models', 'duration': 4, 'Group': 'Model Refinement'},
    {'Task': 'Design Frontend: UI/UX Planning', 'duration': 3, 'Group': 'Front End'},
    {'Task': 'Optimize Models: Fine-tuning and Evaluation', 'duration': 4, 'Group': 'Fine Tuning'},
    {'Task': 'Presentation Practice: Rehearse Demo', 'duration': 4, 'Group': 'Practice'},
    {'Task': 'Final Presentation: Showcase Product', 'duration': 1, 'Group': 'Demo Day'}
])

# Set your desired start date
start_date = pd.to_datetime('2024-03-04')

# Calculate start and end dates based on the cumulative sum of durations
df['start'] = start_date + pd.to_timedelta(df['duration'].cumsum() - df['duration'], unit='D')
df['end'] = start_date + pd.to_timedelta(df['duration'].cumsum(), unit='D')

# Change the color theme to 'plotly_dark'
fig = px.timeline(
    data_frame=df,
    x_start="start",
    x_end="end",
    y="Task",
    color='Group',
    title='Market: Project Team Programme 04/03/2024 - 06/04/2024',
    height=600,
    template='plotly_dark'
)

fig.update_yaxes(autorange="reversed")
fig.show()
