# CivicData AI

This project is an AI-powered web application that allows users to ask questions in natural language about open datasets from Ukraine's official open data portal, [data.gov.ua](https://data.gov.ua/).

The AI agent finds, analyzes, and presents information to the user, making complex data accessible to everyone.

## Key Features

- **AI Chat Interface**: Users interact with the data through a simple and intuitive chat window.
- **Automated Data Discovery**: The agent automatically searches `data.gov.ua` to find the most relevant dataset to answer a user's question.
- **On-the-Fly Analysis**: The agent can process data to calculate summaries, filter by categories, and identify trends.
- **Source Transparency**: Every answer includes a direct link to the source dataset, ensuring data integrity.

## Tech Stack

- **Frontend**: React (Deployed on Netlify)
- **Backend**: Python with FastAPI (Deployed on OnRender)
- **AI Agent**: LangChain with a Mistral AI model
- **Data Processing**: Pandas
- **Database**: MongoDB Atlas (for caching metadata and datasets)
