# LifeOS v2 - Frontend Setup & Usage Guide

## 🚀 Quick Start

### Option 1: Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Flask backend
python app/app.py

# 3. Open the frontend
# Open this file in your browser:
file:///path/to/frontend.html

# Backend will run at: http://localhost:5000
```

### Option 2: Docker Deployment

```bash
docker-compose up
# Frontend available at: http://localhost:5000
```

### Option 3: Hugging Face Spaces

Push to your HF repository and create a new Space with Docker SDK.

---

## 📋 Frontend Features

### 1. **Dashboard** 🎯

- Real-time episode visualization
- 4 stat cards (Total Episodes, Avg Reward, Max Reward, Tools Used)
- 3 interactive charts:
  - Persona Distribution (doughnut chart)
  - Policy Drift Events (bar chart)
  - Reward Over Episodes (line chart)
- Recent episodes table
- Run episodes with one click

**Functions:**

- `loadDashboardData()` - Fetch and display all episodes
- `runTestEpisode()` - Generate 1 episode
- `runMultipleEpisodes(count)` - Generate N episodes
- `updateCharts(personas, drifts, rewards)` - Render Chart.js visualizations

---

### 2. **Executive Resolver** ⚖️

Intelligent conflict resolution engine with multi-agent orchestration.

**Input Fields:**

- Task A (string)
- Task B (string)
- Priority (Task A or Task B)
- Context/Message (textarea)

**Output:**

- Decision analysis
- Reasoning
- Delegation instructions
- Mediation advice
- Email template
- Risk level & confidence

**Function:**

```javascript
resolveConflict()
  → POST /resolve
  → Calls solve_conflict() backend
```

---

### 3. **Emotion Analyst** 💭

Detects emotional patterns and stress indicators in text.

**Input:**

- Any text message

**Output:**

- Detected emotion (angry, stressed, frustrated, positive, neutral)
- Analysis details

**Emotions Detected:**

- 🔴 **Angry** - Contains anger words (angry, furious, hate, etc.)
- ⚠️ **Stressed** - Contains stress keywords (urgent, deadline, panic, etc.)
- 😤 **Frustrated** - Contains frustration signals
- ✅ **Positive** - Contains positive words (thanks, great, happy, etc.)
- ⚪ **Neutral** - No strong emotion

**Function:**

```javascript
analyzeEmotion()
  → POST /emotion
  → Calls detect_emotion() backend
```

---

### 4. **Calm Rewriter** ✍️

Transforms heated/aggressive messages into professional communication.

**Input:**

- Original message (textarea)
- LLM Level (Advanced/Standard)

**Output:**

- Rewritten version with:
  - Tone: description
  - Rewritten: professional message

**Features:**

- Removes aggression
- Maintains intent
- Adds professionalism
- Concise format

**Function:**

```javascript
rewriteMessage()
  → POST /rewrite
  → Calls rewrite_calm_message() backend
```

---

### 5. **Neutral Mediator** 🤝

De-escalates conflicts between multiple parties.

**Input:**

- Person A Position (textarea)
- Person B Position (textarea)
- Shared Goal (string)

**Output:**

- Summary of conflict
- Common Ground identified
- Mediation Plan (3 actionable steps)
- Suggested Message for both parties

**Function:**

```javascript
mediate()
  → POST /mediate
  → Calls mediate_conflict() backend
```

---

## 🔗 API Endpoints

| Method | Endpoint              | Purpose            |
| ------ | --------------------- | ------------------ |
| POST   | `/resolve`            | Resolve conflict   |
| POST   | `/emotion`            | Analyze emotion    |
| POST   | `/rewrite`            | Rewrite message    |
| POST   | `/mediate`            | Mediate conflict   |
| POST   | `/api/run-episode`    | Generate episode   |
| GET    | `/api/episodes`       | Get all episodes   |
| GET    | `/api/episode/latest` | Get latest episode |
| GET    | `/dashboard`          | Dashboard HTML     |
| GET    | `/health`             | Health check       |
| GET    | `/history`            | Chat history       |

---

## 🎨 UI Components

### Navigation

- Sticky navbar with active section highlighting
- 5 main sections (Dashboard, Resolver, Analyzer, Rewriter, Mediator)

### Cards

- Gradient borders on hover
- Backdrop blur effect
- Responsive grid layouts

### Buttons

- Primary gradient buttons
- Secondary transparent buttons
- Loading spinner support

### Badges

- `badge-persona` - Purple badges
- `badge-drift` - Cyan badges
- `badge-reward` - Green/Red badges
- `badge-emotion` - Pink badges

### Charts

- Chart.js integration
- Dark theme colors
- Responsive sizing

---

## 🛠️ Customization

### Change API Base URL

```javascript
// In frontend.html script section
const API_BASE = "http://your-server:5000";
```

### Add Custom Sections

```javascript
// Add to HTML:
<div id="your-section" class="section">...</div>

// Add navigation button:
<button class="nav-btn" onclick="switchSection('your-section')">Label</button>

// Add section function:
function switchSection(sectionId) { ... }
```

### Modify Colors

```css
:root {
  --primary: #22d3ee; /* Cyan */
  --secondary: #8b5cf6; /* Purple */
  --accent: #ec4899; /* Pink */
  --success: #10b981; /* Green */
  --danger: #ef4444; /* Red */
}
```

---

## 📊 Dashboard Details

### Persona Types

1. **Executive Parent** - High responsibility, family-sensitive
2. **Startup Founder** - Speed-focused, deadline-driven
3. **Balanced Professional** - Stability-first, context-aware

### Policy Drift Events

- **v1** - No hard limits
- **v2** - No work meetings after 8 PM
- **v3** - Family events cannot be auto-cancelled

### Tool Set

- **calendar** - Schedule rescheduling
- **email** - Message composition
- **rides** - Travel coordination
- **shopping** - E-commerce actions

### Reward Breakdown

- `priority_alignment` - Chose right priority (+15 points)
- `rescheduling_quality` - Good rescheduling (+20 points)
- `tone_quality` - Professional tone (+5 points)
- `policy_compliance` - Respects policies (+6 points)
- `prediction_bonus` - Anticipates conflicts (+4 points)
- `tool_execution_bonus` - Tools executed well (+8 points)
- `penalty` - No strategy (-10 points)

---

## 🧪 Testing

### Quick Test

```bash
python test_dashboard.py
```

### Manual Test Steps

1. Open `frontend.html` in browser
2. Go to Dashboard section
3. Click "Run Single Episode"
4. Check if chart updates
5. Go to Resolver section
6. Enter conflict details
7. Click "Resolve Conflict"
8. Verify output appears

---

## 🐛 Troubleshooting

### CORS Errors

- Ensure `flask-cors` is installed: `pip install flask-cors`
- Frontend URL must match API base URL

### Chart Not Loading

- Clear browser cache
- Check browser console for errors
- Verify `/api/episodes` returns data

### Button Not Working

- Check browser console for errors
- Verify Flask server is running
- Check API base URL matches

### No Data in Dashboard

- Run at least one episode
- Click "Refresh Data" button
- Check network tab in DevTools

---

## 📝 File Structure

```
lifeos-v2/
├── frontend.html           # Main frontend file (this one!)
├── app/
│   └── app.py             # Flask backend
├── openenv_env/           # Multi-agent environment
│   ├── agents.py
│   ├── env.py
│   ├── personas.py
│   ├── reward.py
│   ├── planner.py
│   ├── workflow_engine.py
│   └── tools.py
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker development
└── README.md            # Project overview
```

---

## 🚀 Deployment

### To Hugging Face Spaces

1. Commit all files: `git add .`
2. Push to repository: `git push`
3. Create Space at [hf.co/new-space](https://huggingface.co/new-space)
4. Select **Docker** as SDK
5. Connect your repo
6. Space will auto-deploy!

---

## 📞 Support

For issues or questions:

- Check `/health` endpoint
- Review browser console errors
- Verify all ports are accessible
- Ensure requirements are installed

---

**Happy Conflict Resolution! 🎉**
