# PoliScanSight Bot 🕵️

AI-Powered Political Intelligence & Sentiment Analysis Platform built using CrewAI, Streamlit, LangChain, OpenAI GPT-4o-mini, Ollama, and Serper API.

## 📌 Overview

PoliScanSight Bot is a multi-agent AI-powered political intelligence platform designed to automate political research, controversy analysis, public sentiment tracking, 
opposition monitoring, and strategic recommendation generation for political leaders and parties.

The platform uses multiple AI agents working together to generate comprehensive political intelligence reports using real-time web research and LLM reasoning.

---

## 🚀 Features

- Multi-Agent Political Analysis System
- Political Activity & Scheme Monitoring
- Controversy & Reputation Analysis
- Public Sentiment Tracking
- Community & Caste Impact Analysis
- Opposition Strategy Monitoring
- AI-Generated Strategic Recommendations
- Automated Markdown Report Generation
- OpenAI + Ollama Support
- Real-Time Web Search Integration

---

## 🏗️ Architecture

```text
                    +----------------------+
                    |      Streamlit UI    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |     CrewAI Engine    |
                    +----------+-----------+
                               |
        ------------------------------------------------
        |              |               |               |
        v              v               v               v
+---------------+ +---------------+ +---------------+ +---------------+
| Activity      | | Landscape     | | Sentiment     | | Report Writer |
| Researcher    | | Monitor       | | Analyzer      | | Agent         |
+---------------+ +---------------+ +---------------+ +---------------+
        |                |                |
        -----------------------------------
                       |
                       v
              +------------------+
              | Serper Web Search|
              +------------------+
                       |
                       v
              +------------------+
              | OpenAI / Ollama  |
              +------------------+
```

---

## 🧠 AI Agents

### 1️⃣ Political Affairs Investigator
- Tracks political activities
- Analyzes controversies
- Monitors government schemes
- Studies community impact

### 2️⃣ Political Landscape Analyst
- Tracks opposition strategies
- Monitors political campaigns
- Analyzes narrative warfare

### 3️⃣ Sentiment Analyzer
- Performs public sentiment analysis
- Tracks narrative shifts
- Identifies key themes & hashtags

### 4️⃣ Strategic Report Writer
- Generates final intelligence reports
- Creates strategic recommendations
- Compiles political insights

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### AI Frameworks
- CrewAI
- LangChain

### LLMs
- OpenAI GPT-4o-mini
- Ollama (Llama3.1)

### APIs & Tools
- Serper API
- dotenv

### Programming Language
- Python

---

## 📂 Project Structure

```bash
PoliScanSight/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── reports/
│   ├── stalin_report.md
│   ├── dmk_report.md
│
└── assets/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/poliscansight.git
cd poliscansight
```

---

### 2️⃣ Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
APP_PASSWORD=your_password
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔐 Authentication

The application includes password-protected access using environment variables and Streamlit session state management.

---

## 📊 Generated Report Includes

- Political Activities & Schemes
- Controversy Analysis
- Community & Caste Impact
- Public Sentiment Analysis
- Opposition Strategy Monitoring
- Strategic Recommendations
- Reference Sources & URLs

---

## 📈 Use Cases

### 🗳️ Election Campaign Monitoring
Track political sentiment and opposition activity during elections.

### 📢 Political Reputation Management
Monitor controversies and public narratives.

### 🧠 Political Strategy Planning
Generate AI-powered political strategy recommendations.

### 📊 Political Intelligence Automation
Automate political reporting workflows using multi-agent AI systems.

---

## 📸 Screenshots

### Dashboard
(Add Screenshot Here)

### Generated Political Report
(Add Screenshot Here)

### AI Agent Workflow
(Add Screenshot Here)

---

## 🔮 Future Improvements

- Twitter/X Sentiment Integration
- YouTube Political Monitoring
- WhatsApp Narrative Analysis
- Constituency-Level Intelligence
- Real-Time Political Dashboard
- RAG-Based Political Memory System
- Election Forecasting Models

---

## 👨‍💻 Author

**Prajay Urkude**

Political Data Analyst | AI Workflow Builder | Survey Analytics Specialist

### Skills
- Political Analytics
- Survey Analytics
- AI Workflow Automation
- Sentiment Analysis
- Election Intelligence
- Political Reporting Systems

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

Special thanks to:
- CrewAI
- Streamlit
- LangChain
- OpenAI
- Ollama

for enabling advanced AI workflow orchestration.

---

## 🚀 Final Note

PoliScanSight Bot transforms traditional political analysis into an AI-powered intelligence workflow by combining:

- Multi-agent reasoning
- Real-time web intelligence
- Sentiment analytics
- Strategic political reporting

into a single automated political intelligence platform.
