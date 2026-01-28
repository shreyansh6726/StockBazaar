# StockBazaar - Real-Time Stock Market Dashboard

A full-stack web application for visualizing and analyzing real-time stock market data across major global companies. StockBazaar provides an intuitive interface to explore historical stock price trends, powered by the Alpha Vantage API with a modern React frontend and Python Flask backend.


## Overview

**StockBazaar** is a comprehensive stock market analytics platform that helps users track and visualize stock price movements over time. The application supports 55+ global companies across various sectors including technology, finance, energy, healthcare, retail, and manufacturing.

The project combines a **Flask backend** that interfaces with the Alpha Vantage financial data API and a **React frontend** that renders interactive charts using Recharts. Whether you're an investor, trader, or simply interested in stock market trends, StockBazaar provides real-time insights with a clean, user-friendly interface.

---

## Features

### Frontend Features

- **Interactive Stock Charts**: Beautiful line charts visualizing closing prices over time using Recharts
- **Company Selection**: Dropdown menu with 55+ major global companies organized by ticker symbol and full name
- **Real-Time Data Display**: Fetch and display the latest stock market data with a single click
- **Responsive Design**: Fully responsive UI that works on desktop, tablet, and mobile devices
- **Loading States**: Visual feedback during data retrieval to enhance user experience
- **Error Handling**: Graceful error messages when data is unavailable or API limits are reached
- **Color-Coded Interface**: Carefully designed color scheme for better readability and user engagement

### Backend Features

- **RESTful API**: Clean API endpoint for fetching stock data
- **CORS Support**: Configured for secure cross-origin requests from the frontend
- **Alpha Vantage Integration**: Direct connection to Alpha Vantage API for real financial data
- **Error Handling**: Comprehensive error management for API failures and invalid requests
- **CLI Tool**: Command-line interface for interactive stock price exploration with data visualization

### Additional Utilities

- **Python CLI Tool** (`stock.py`): Interactive command-line application for exploring stock data without the web interface
- **Data Visualization**: Matplotlib integration for generating charts from terminal

---

## Project Architecture

StockBazaar follows a **client-server architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Frontend (React 19)                   │
│  - StockDashboard.jsx: Main component rendering charts & UI     │
│  - Axios: HTTP client for API communication                     │
│  - Recharts: Interactive charting library                       │
│  - Deployed on: Vercel (Production)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTP REST API (JSON)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Flask Backend (Python)                         │
│  - app.py: Main Flask application with API endpoint            │
│  - stock.py: CLI utility for stock data analysis               │
│  - CORS enabled for secure frontend communication              │
│  - Deployed on: Render (Production)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTPS API Request (JSON)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Alpha Vantage Financial API                    │
│  - TIME_SERIES_DAILY: Provides historical daily stock prices   │
│  - Returns: JSON data with OHLCV (Open, High, Low, Close, Vol) │
│  - Rate Limited: Free tier allows 5 requests/min               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Interaction**: User selects a company from the dropdown in the React dashboard
2. **Frontend Request**: React component makes HTTP GET request to Flask backend
3. **Backend Processing**: Flask receives the stock symbol and formats it for Alpha Vantage API
4. **External API Call**: Flask queries Alpha Vantage with the stock symbol and API key
5. **Response Processing**: Flask receives JSON response with historical price data
6. **Frontend Rendering**: React receives the data and renders an interactive line chart
7. **User Visualization**: Chart displays closing prices over time with interactive tooltips

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| **React** | 19.2.4 | UI framework and component management |
| **ReactDOM** | 19.2.4 | DOM rendering for React components |
| **Recharts** | 3.7.0 | Interactive charting and visualization |
| **Axios** | 1.13.4 | HTTP client for API requests |
| **React Scripts** | 5.0.1 | Build tools and development server |

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.x | Backend programming language |
| **Flask** | Latest | Web framework for RESTful API |
| **Flask-CORS** | Latest | Cross-Origin Resource Sharing support |
| **Requests** | Latest | HTTP library for API calls |
| **Python-dotenv** | Latest | Environment variable management |
| **Gunicorn** | Latest | WSGI HTTP Server for production deployment |
| **Pandas** | Latest | Data manipulation and analysis |
| **Matplotlib** | Latest | Data visualization for CLI tool |

### External Services

- **Alpha Vantage API**: Real-time financial data for 55+ companies
- **Vercel**: Frontend hosting and deployment (production)
- **Render**: Backend hosting and deployment (production)

---

## Installation & Setup

### Prerequisites

Ensure you have the following installed:
- **Node.js** (v14 or higher) and npm
- **Python** (v3.7 or higher) and pip
- **Git** for version control

### Backend Setup

#### Step 1: Clone the Repository

```bash
cd c:\Users\shrey\Documents\Github\stockbazaar
git clone <repository-url>
cd stockbazaar/backend
```

#### Step 2: Create a Virtual Environment

```bash
# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask: Web framework
- Flask-CORS: Cross-origin support
- requests: HTTP library
- python-dotenv: Environment variables
- gunicorn: Production server
- pandas: Data analysis
- matplotlib: Visualization

#### Step 4: Set Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# .env file
ALPHA_VANTAGE_KEY=your_api_key_here
PORT=5000
```

**To get an API key:**
1. Visit [Alpha Vantage](https://www.alphavantage.co/api/)
2. Sign up for a free account
3. Copy your API key
4. Paste it into the `.env` file

#### Step 5: Run the Backend Server

```bash
# Development mode with auto-reload
python app.py

# Production mode with Gunicorn
gunicorn app:app --bind 0.0.0.0:5000
```

The server will start at `http://localhost:5000`

---

### Frontend Setup

#### Step 1: Navigate to Frontend Directory

```bash
cd stockbazaar/frontend
```

#### Step 2: Install Dependencies

```bash
npm install
```

This installs React, Recharts, Axios, testing libraries, and other utilities.

#### Step 3: Configure API Base URL (Optional)

The application automatically detects production vs development environment:
- **Development**: Uses `http://localhost:5000`
- **Production**: Uses `https://stockbazaar.onrender.com`

To modify this, edit `frontend/src/StockDashboard.jsx`:

```jsx
const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://stockbazaar.onrender.com' 
    : 'http://localhost:5000';
```

#### Step 4: Run the Development Server

```bash
npm start
```

This starts the React development server at `http://localhost:3000` with hot reloading enabled.

#### Step 5: Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` directory, ready for deployment.

---

## Project Structure

```
stockbazaar/
├── LICENSE                          # MIT License
├── README.md                        # Documentation (this file)
│
├── backend/                         # Flask backend directory
│   ├── app.py                       # Main Flask application
│   │   └── Routes:
│   │       ├── GET /                # Health check
│   │       └── GET /api/stock/<symbol>  # Fetch stock data
│   │
│   ├── stock.py                     # CLI utility for stock analysis
│   │   ├── Fetches stock data from Alpha Vantage
│   │   ├── Interactive menu for company selection
│   │   └── Matplotlib visualization of prices
│   │
│   ├── requirements.txt             # Python dependencies
│   │   ├── Flask
│   │   ├── Flask-CORS
│   │   ├── requests
│   │   ├── python-dotenv
│   │   ├── gunicorn
│   │   ├── pandas
│   │   └── matplotlib
│   │
│   ├── gunicorn                     # Gunicorn configuration (production)
│   └── .env                         # Environment variables (not in repo)
│
├── frontend/                        # React frontend directory
│   ├── public/                      # Static assets
│   │   ├── index.html              # Main HTML file
│   │   ├── logo.png                # Favicon
│   │   └── robots.txt              # Search engine crawler directives
│   │
│   ├── src/                        # React components and styles
│   │   ├── index.jsx               # Application entry point
│   │   │   └── Renders StockDashboard into root div
│   │   │
│   │   ├── StockDashboard.jsx      # Main dashboard component
│   │   │   ├── Company list (55+ companies)
│   │   │   ├── Company dropdown selector
│   │   │   ├── Recharts LineChart visualization
│   │   │   ├── Axios API integration
│   │   │   └── Loading/error states
│   │   │
│   │   ├── App.test.js             # Test file for App component
│   │   ├── index.css               # Global styles
│   │   ├── reportWebVitals.js      # Performance monitoring
│   │   └── setupTests.js           # Test configuration
│   │
│   ├── package.json                # NPM dependencies and scripts
│   │   ├── React: 19.2.4
│   │   ├── Recharts: 3.7.0
│   │   ├── Axios: 1.13.4
│   │   └── Testing libraries
│   │
│   └── .env                        # Environment variables (optional)
```