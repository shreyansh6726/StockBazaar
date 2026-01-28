# StockBazaar - Real-Time Stock Market Dashboard

A full-stack web application for visualizing and analyzing real-time stock market data across major global companies. StockBazaar provides an intuitive interface to explore historical stock price trends, powered by the Alpha Vantage API with a modern React frontend and Python Flask backend.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Using the Dashboard](#using-the-dashboard)
  - [Using the CLI Tool](#using-the-cli-tool)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Key Components](#key-components)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Supported Companies](#supported-companies)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Author](#author)

---

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

## Configuration

### CORS Configuration

The backend is configured to accept requests from:
- **Production**: `https://stock-bazaar-one.vercel.app`
- **Development**: `http://localhost:3000`

To modify CORS settings, edit `backend/app.py`:

```python
CORS(app, origins=["https://stock-bazaar-one.vercel.app", "http://localhost:3000"])
```

### Environment Variables

#### Backend `.env` File

```env
# Alpha Vantage API Key (required)
ALPHA_VANTAGE_KEY=demo  # Replace with your actual key

# Server Configuration (optional)
PORT=5000
FLASK_ENV=development  # or production
```

#### Frontend `.env` File (Optional)

```env
# If you need to override the API base URL
REACT_APP_API_BASE_URL=http://localhost:5000
```

---

## Usage

### Running the Application

#### Local Development (Recommended for Testing)

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
# Backend running at http://localhost:5000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
# Frontend running at http://localhost:3000
```

Then open `http://localhost:3000` in your browser.

#### Production Deployment

- **Frontend**: Deployed on Vercel ([stock-bazaar-one.vercel.app](https://stock-bazaar-one.vercel.app))
- **Backend**: Deployed on Render ([stockbazaar.onrender.com](https://stockbazaar.onrender.com))

---

### Using the Dashboard

#### Step 1: Select a Company

1. Open the StockBazaar dashboard in your browser
2. Click the dropdown menu that shows "-- Select a Company --"
3. Browse through 55+ companies or search by typing the ticker symbol

#### Step 2: View the Chart

1. Select any company from the list
2. The dashboard fetches real-time data from the Alpha Vantage API
3. An interactive line chart displays the closing prices over the last trading days
4. The chart automatically updates with the latest data

#### Step 3: Interact with the Chart

- **Hover**: Move your mouse over the chart to see exact closing prices and dates
- **Zoom**: Use browser zoom for detailed inspection
- **Select Different Company**: Choose another company from the dropdown to compare

#### Example Companies Available

**Technology**: Apple (AAPL), Microsoft (MSFT), Google (GOOG), Meta (META), NVIDIA (NVDA)

**Finance**: JPMorgan (JPM), Visa (V), Mastercard (MA), Bank of America (BAC)

**Energy**: ExxonMobil (XOM), Chevron (CVX), Shell (SHEL), BP (BP)

**Healthcare**: Johnson & Johnson (JNJ), Eli Lilly (LLY), Pfizer (PFE), Merck (MRK)

**Retail & Consumer**: Walmart (WMT), Amazon (AMZN), McDonald's (MCD), Coca-Cola (KO)

---

### Using the CLI Tool

The `stock.py` CLI utility provides command-line access to stock data with graphical visualization.

#### Step 1: Prepare Environment

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

#### Step 2: Run the CLI Tool

```bash
python stock.py
```

#### Step 3: Follow the Menu

```
--- Select a Company to Check Share Price ---
1. AAPL    2. MSFT    3. GOOG    4. META    5. NVDA
6. TSM     7. AVGO    8. ADBE    9. CSCO    10. ORCL
... (55+ companies total)

Enter the number (1-55) of the company: 1
```

#### Step 4: View the Chart

The CLI tool will:
1. Fetch historical daily stock data for the selected company
2. Display a matplotlib chart showing the stock's closing price trend
3. The chart displays:
   - Company name and ticker symbol in the title
   - X-axis: Date of the trading day
   - Y-axis: Closing price in USD
   - Green line: Historical price trend

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

---

## API Endpoints

### Health Check

**Endpoint**: `GET /`

**Purpose**: Verify backend server is running

**Response**:
```
Backend is running!
```

### Fetch Stock Data

**Endpoint**: `GET /api/stock/<symbol>`

**Purpose**: Retrieve historical daily stock data for a company

**Parameters**:
- `symbol` (required): Stock ticker symbol (e.g., AAPL, MSFT, GOOG)

**Example Request**:
```bash
curl http://localhost:5000/api/stock/AAPL
```

**Response** (Success - 200 OK):
```json
{
  "Meta Data": {
    "1. Information": "Daily Prices",
    "2. Symbol": "AAPL",
    "3. Last Refreshed": "2024-01-26",
    "4. Output Size": "Full size",
    "5. Time Zone": "US/Eastern"
  },
  "Time Series (Daily)": {
    "2024-01-26": {
      "1. open": "189.45",
      "2. high": "191.30",
      "3. low": "188.92",
      "4. close": "190.15",
      "5. volume": "52500000"
    },
    "2024-01-25": {
      "1. open": "188.50",
      "2. high": "190.22",
      "3. low": "188.10",
      "4. close": "189.80",
      "5. volume": "48200000"
    }
    // ... more historical data
  }
}
```

**Response** (Error - 500 Internal Server Error):
```json
{
  "error": "API rate limit exceeded or invalid API key"
}
```

**Important Notes**:
- Alpha Vantage free tier: 5 requests per minute
- Full size output includes 20+ years of historical data
- Data updates daily after market close (4 PM EST)
- Rate limited API key will return a note in response

---

## Key Components

### Frontend Components

#### StockDashboard.jsx

The main React component that powers the dashboard.

**Key Features**:
- **State Management**: Uses React hooks (useState, useEffect)
- **Company Mapping**: Array of 55+ companies with ticker and full names
- **Data Fetching**: Axios integration with async/await
- **Error Handling**: Graceful handling of failed API requests
- **Chart Rendering**: Recharts LineChart with interactive tooltips
- **Responsive Design**: CSS-in-JS styling with color variables

**Code Highlights**:

```jsx
const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://stockbazaar.onrender.com' 
    : 'http://localhost:5000';

const StockDashboard = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('');
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchData = async (symbol) => {
        setLoading(true);
        const res = await axios.get(`${API_BASE_URL}/api/stock/${symbol}`);
        const timeSeries = res.data['Time Series (Daily)'];
        
        // Format data for Recharts
        const formattedData = Object.keys(timeSeries).map(date => ({
            date,
            close: parseFloat(timeSeries[date]['4. close'])
        })).reverse();
        
        setData(formattedData);
        setLoading(false);
    };
};
```

---

### Backend Components

#### app.py

The main Flask backend application.

**Key Features**:
- **Flask App**: Creates and configures the Flask application
- **CORS Support**: Enables cross-origin requests
- **Environment Variables**: Loads API key from .env file
- **API Route**: Single endpoint for fetching stock data
- **Error Handling**: Try-catch for API failures

**Code Highlights**:

```python
from flask import Flask, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, origins=["https://stock-bazaar-one.vercel.app", 
                   "http://localhost:3000"])

API_KEY = os.getenv('ALPHA_VANTAGE_KEY')

@app.route('/api/stock/<symbol>')
def get_stock(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get('https://www.alphavantage.co/query', 
                           params=params)
    return jsonify(response.json())
```

#### stock.py

CLI utility for command-line stock analysis.

**Key Features**:
- **Interactive Menu**: Display 55+ companies in a formatted menu
- **User Input**: Get company selection from user
- **API Integration**: Fetch data from Alpha Vantage
- **Data Processing**: Convert JSON to Pandas DataFrame
- **Visualization**: Create line chart with Matplotlib

**Code Highlights**:

```python
def show_menu():
    # Display company list in 5 columns
    for i, symbol in enumerate(symbols, 1):
        print(f"{i:2}. {symbol}", end="\t" if i % 5 != 0 else "\n")

def get_user_choice():
    # Get valid company selection from user
    choice = int(input(f"Enter the number (1-{len(symbols)}): "))
    return symbols[choice - 1]

# Visualization
df = pd.DataFrame.from_dict(time_series, orient='index')
plt.plot(df.index, df['close'], color='green', linewidth=2)
plt.title(f'Stock Performance: {selected_symbol}')
plt.show()
```

---

## Deployment

### Frontend Deployment (Vercel)

1. **Push code to GitHub**:
```bash
git push origin main
```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Select the GitHub repository
   - Configure settings:
     - Framework: React
     - Root Directory: ./frontend
     - Build Command: `npm run build`
     - Output Directory: `build`

3. **Environment Variables** (in Vercel Dashboard):
   - Set any necessary environment variables
   - Auto-detects Next.js/React apps

4. **Deploy**:
   - Vercel automatically deploys on push to main branch
   - Builds with `npm run build`
   - Hosted at: `https://stock-bazaar-one.vercel.app`

### Backend Deployment (Render)

1. **Push code to GitHub**:
```bash
git push origin main
```

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Click "New Web Service"
   - Connect GitHub repository
   - Configure settings:
     - Name: `stockbazaar`
     - Environment: Python
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
     - Root Directory: `backend`

3. **Environment Variables** (in Render Dashboard):
   ```
   ALPHA_VANTAGE_KEY=your_api_key_here
   ```

4. **Deploy**:
   - Render automatically deploys on push
   - Builds with requirements.txt
   - Hosted at: `https://stockbazaar.onrender.com`

---

## Environment Variables

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required: Alpha Vantage API Key
ALPHA_VANTAGE_KEY=your_32_character_api_key_here

# Optional: Server Configuration
PORT=5000
FLASK_ENV=development  # or production
DEBUG=False
```

### Frontend Environment Variables (Optional)

Create a `.env` file in the `frontend/` directory:

```env
# Optional: Override API base URL
REACT_APP_API_BASE_URL=http://localhost:5000
```

**Important**: 
- Never commit `.env` files to version control
- Environment variables are loaded automatically in development
- For production, set variables in deployment platform dashboard
- API keys must be kept secret

---

## Supported Companies

StockBazaar supports 55+ major global companies across 7 sectors:

### Technology (Top 13)
AAPL (Apple), MSFT (Microsoft), GOOG (Google), META (Meta), NVDA (NVIDIA), TSM (TSMC), AVGO (Broadcom), ADBE (Adobe), CSCO (Cisco), ORCL (Oracle), NFLX (Netflix), AMD (AMD), ASML (ASML)

### E-Commerce & Retail (5)
AMZN (Amazon), WMT (Walmart), COST (Costco), HD (Home Depot), MCD (McDonald's)

### Consumer Goods & Food (5)
PG (Procter & Gamble), KO (Coca-Cola), PEP (PepsiCo), LVMUY (LVMH), NSRGY (Nestlé)

### Energy & Commodities (5)
XOM (ExxonMobil), CVX (Chevron), SHEL (Shell), BP (BP), VALE (Vale)

### Financial Services (9)
JPM (JPMorgan), V (Visa), MA (Mastercard), BAC (Bank of America), WFC (Wells Fargo), BRK.B (Berkshire), HSBC (HSBC), GS (Goldman Sachs), AXP (American Express), TCEHY (Tencent)

### Healthcare & Pharma (6)
JNJ (J&J), LLY (Eli Lilly), UNH (UnitedHealth), MRK (Merck), PFE (Pfizer), NVO (Novo Nordisk), AZN (AstraZeneca), ROG (Roche)

### Industrial & Manufacturing (6)
BA (Boeing), GE (GE), MMM (3M), CAT (Caterpillar), TOYOF (Toyota), DDAIF (Mercedes-Benz), SIE (Siemens)

To add more companies, edit the `companyMap` array in `StockDashboard.jsx` and the `symbols` list in `stock.py`.

---

## Troubleshooting

### Issue: "Backend is not running" Error

**Symptoms**: Cannot connect to backend API

**Solutions**:
1. Verify backend is running: `python app.py`
2. Check if port 5000 is available: `netstat -ano | findstr :5000`
3. Verify CORS configuration in `app.py`
4. Check browser console for exact error message

### Issue: "API rate limit exceeded"

**Symptoms**: Error message about rate limits

**Causes**: Alpha Vantage free tier limit (5 requests/minute)

**Solutions**:
1. Wait 1 minute before making another request
2. Upgrade to paid API plan
3. Implement request caching in backend

### Issue: "No data available for this symbol"

**Symptoms**: Chart doesn't display data

**Causes**:
- Invalid stock symbol
- Alpha Vantage API key not set
- API rate limit exceeded

**Solutions**:
1. Verify stock symbol is in the supported list
2. Check `.env` file has correct API key
3. Ensure API key is active and valid

### Issue: CORS Errors in Browser Console

**Symptoms**: "Access to XMLHttpRequest blocked by CORS policy"

**Causes**: Frontend URL not in CORS allowed origins

**Solutions**:
1. Add your frontend URL to CORS origins in `app.py`:
   ```python
   CORS(app, origins=["http://yourdomain.com", ...])
   ```
2. Verify frontend is running on correct port
3. Check that headers are properly set

### Issue: "Port 5000 already in use"

**Symptoms**: Address already in use error

**Solutions**:
1. Kill process using port 5000:
   ```bash
   # Windows
   taskkill /F /PID <pid>
   
   # macOS/Linux
   lsof -ti:5000 | xargs kill -9
   ```
2. Use different port: `python app.py --port 5001`

### Issue: npm Dependencies Installation Fails

**Symptoms**: npm install errors

**Solutions**:
1. Clear npm cache: `npm cache clean --force`
2. Delete node_modules: `rm -rf node_modules`
3. Reinstall: `npm install`
4. Update npm: `npm install -g npm@latest`

### Issue: Python Dependencies Installation Fails

**Symptoms**: pip install errors

**Solutions**:
1. Verify Python version: `python --version`
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Install with no cache: `pip install -r requirements.txt --no-cache-dir`

---

## Future Enhancements

### Planned Features

1. **Advanced Analytics**
   - Moving averages (50-day, 200-day)
   - RSI and MACD indicators
   - Volatility analysis
   - Earnings forecasts

2. **User Accounts**
   - User authentication with JWT
   - Save favorite companies/watchlists
   - Portfolio tracking
   - User preferences/settings

3. **Comparative Analysis**
   - Compare multiple stocks on one chart
   - Sector comparison
   - Year-to-date performance
   - Risk metrics

4. **Enhanced Data**
   - Real-time intraday data
   - Options chain data
   - Dividend history
   - Historical splits

5. **Frontend Improvements**
   - Dark mode support
   - Advanced filtering
   - Export charts to PDF
   - Mobile app version

6. **Backend Improvements**
   - Database caching (Redis)
   - Request rate limiting
   - Logging and monitoring
   - WebSocket for real-time updates

7. **DevOps**
   - Docker containerization
   - CI/CD pipeline (GitHub Actions)
   - Automated testing
   - Performance monitoring

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

You are free to:
- Use the software for any purpose
- Modify and distribute the software
- Include the software in proprietary applications

Please include a copy of the license and attribution.

---

## Author

**Shreyansh Srivastava**

- GitHub: [@shreyansh6726](https://github.com/shreyansh6726)
- Repository: [StockBazaar](https://github.com/shreyansh6726/StockBazaar)

---

## Support & Contributing

### Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages in browser console
3. Check backend logs in terminal
4. Open an issue on GitHub

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## Acknowledgments

- **Alpha Vantage**: Free financial data API
- **React**: UI framework
- **Flask**: Backend framework
- **Recharts**: Interactive charting library
- **Vercel & Render**: Hosting platforms

---

## Quick Reference

### Common Commands

```bash
# Backend
cd backend
python -m venv venv
.\venv\Scripts\Activate  # Windows
pip install -r requirements.txt
python app.py  # Development
gunicorn app:app  # Production

# Frontend
cd frontend
npm install
npm start  # Development
npm run build  # Production
npm test  # Run tests

# CLI Tool
python stock.py
```

### Useful URLs

- Local Frontend: `http://localhost:3000`
- Local Backend: `http://localhost:5000`
- Production Frontend: `https://stock-bazaar-one.vercel.app`
- Production Backend: `https://stockbazaar.onrender.com`
- Alpha Vantage: `https://www.alphavantage.co/`

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Active & Maintained
