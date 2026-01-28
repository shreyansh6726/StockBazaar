import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Determine the API Base URL based on environment
// Replace 'your-render-app-name' with your actual Render service name after deployment
const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://your-render-app-name.onrender.com' 
    : 'http://localhost:5000';

const companyMap = [
    { symbol: "AAPL", name: "Apple Inc." },
    { symbol: "MSFT", name: "Microsoft Corporation" },
    { symbol: "GOOG", name: "Alphabet Inc. (Google)" },
    { symbol: "META", name: "Meta Platforms, Inc." },
    { symbol: "NVDA", name: "NVIDIA Corporation" },
    { symbol: "TSM", name: "Taiwan Semiconductor Manufacturing" },
    { symbol: "AVGO", name: "Broadcom Inc." },
    { symbol: "ADBE", name: "Adobe Inc." },
    { symbol: "CSCO", name: "Cisco Systems, Inc." },
    { symbol: "ORCL", name: "Oracle Corporation" },
    { symbol: "NFLX", name: "Netflix, Inc." },
    { symbol: "AMD", name: "Advanced Micro Devices, Inc." },
    { symbol: "ASML", name: "ASML Holding N.V." },
    { symbol: "AMZN", name: "Amazon.com, Inc." },
    { symbol: "TSLA", name: "Tesla, Inc." },
    { symbol: "WMT", name: "Walmart Inc." },
    { symbol: "COST", name: "Costco Wholesale Corporation" },
    { symbol: "HD", name: "The Home Depot, Inc." },
    { symbol: "PG", name: "Procter & Gamble Co." },
    { symbol: "KO", name: "The Coca-Cola Company" },
    { symbol: "PEP", name: "PepsiCo, Inc." },
    { symbol: "MCD", name: "McDonald's Corporation" },
    { symbol: "LVMUY", name: "LVMH Moët Hennessy Louis Vuitton" },
    { symbol: "NSRGY", name: "Nestlé S.A." },
    { symbol: "2222.SR", name: "Saudi Arabian Oil Co. (Aramco)" },
    { symbol: "XOM", name: "Exxon Mobil Corporation" },
    { symbol: "CVX", name: "Chevron Corporation" },
    { symbol: "SHEL", name: "Shell plc" },
    { symbol: "BP", name: "BP p.l.c." },
    { symbol: "VALE", name: "Vale S.A." },
    { symbol: "JPM", name: "JPMorgan Chase & Co." },
    { symbol: "V", name: "Visa Inc." },
    { symbol: "MA", name: "Mastercard Incorporated" },
    { symbol: "BAC", name: "Bank of America Corp." },
    { symbol: "WFC", name: "Wells Fargo & Company" },
    { symbol: "BRK.B", name: "Berkshire Hathaway Inc." },
    { symbol: "HSBC", name: "HSBC Holdings plc" },
    { symbol: "GS", name: "The Goldman Sachs Group, Inc." },
    { symbol: "AXP", name: "American Express Company" },
    { symbol: "TCEHY", name: "Tencent Holdings Limited" },
    { symbol: "JNJ", name: "Johnson & Johnson" },
    { symbol: "LLY", name: "Eli Lilly and Company" },
    { symbol: "UNH", name: "UnitedHealth Group Incorporated" },
    { symbol: "MRK", name: "Merck & Co., Inc." },
    { symbol: "PFE", name: "Pfizer Inc." },
    { symbol: "NVO", name: "Novo Nordisk A/S" },
    { symbol: "AZN", name: "AstraZeneca PLC" },
    { symbol: "ROG", name: "Roche Holding AG" },
    { symbol: "TOYOF", name: "Toyota Motor Corporation" },
    { symbol: "BA", name: "The Boeing Company" },
    { symbol: "GE", name: "General Electric Company" },
    { symbol: "MMM", name: "3M Company" },
    { symbol: "CAT", name: "Caterpillar Inc." },
    { symbol: "DDAIF", name: "Mercedes-Benz Group AG" },
    { symbol: "SIE", name: "Siemens AG" }
];

const StockDashboard = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('');
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(false);

    // Color Palette Constants from Color Hunt
    const colors = {
        primary: "#6F8F72",    // Forest Green (Lines/Text)
        secondary: "#F2A65A",  // Muted Orange (Accents/Loading)
        bgMain: "#BFC6C4",     // Sage Grey (Page Background)
        bgCard: "#E8E2D8",     // Warm Beige (Container Background)
        textDark: "#2F3E33"    // Darker variant for readability
    };

    const currentCompany = companyMap.find(c => c.symbol === selectedSymbol);

    const fetchData = async (symbol) => {
        if (!symbol) return;
        setLoading(true);
        try {
            const res = await axios.get(`${'https://stockbazaar.onrender.com'}/api/stock/${symbol}`);
            const timeSeries = res.data['Time Series (Daily)'];
            
            if (timeSeries) {
                const formattedData = Object.keys(timeSeries).map(date => ({
                    date,
                    close: parseFloat(timeSeries[date]['4. close'])
                })).reverse();
                setData(formattedData);
            } else {
                console.error("API response missing time series data", res.data);
                setData([]);
            }
        } catch (err) {
            console.error("Error fetching data:", err);
            setData([]);
        }
        setLoading(false);
    };

    useEffect(() => {
        fetchData(selectedSymbol);
    }, [selectedSymbol]);

    return (
        <div style={{ 
            minHeight: '100vh',
            backgroundColor: colors.bgMain, 
            padding: '40px 20px', 
            fontFamily: 'Segoe UI, Roboto, sans-serif', 
            textAlign: 'center',
            color: colors.textDark 
        }}>
            <h1 style={{ marginBottom: '10px', fontWeight: '800', letterSpacing: '-1px' }}>StockBazaar</h1>
            
            <div style={{ marginBottom: '30px' }}>
                <select 
                    value={selectedSymbol} 
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    style={{ 
                        padding: '12px 20px', 
                        borderRadius: '30px', 
                        border: `2px solid ${colors.primary}`, 
                        backgroundColor: colors.bgCard,
                        fontSize: '16px', 
                        width: '320px', 
                        maxHeight: '200px',
                        cursor: 'pointer',
                        outline: 'none',
                        color: colors.textDark,
                        fontWeight: '600'
                    }}
                >
                    <option value="" disabled>-- Select a Company --</option>
                    {companyMap.map(company => (
                        <option key={company.symbol} value={company.symbol}>
                            {company.symbol} — {company.name}
                        </option>
                    ))}
                </select>
            </div>

            {loading ? (
                <div style={{ padding: '60px' }}>
                    <p style={{ color: colors.secondary, fontSize: '20px', fontWeight: 'bold' }}>Retrieving Market Trends...</p>
                </div>
            ) : data.length > 0 ? (
                <div style={{ 
                    maxWidth: '900px', 
                    margin: '0 auto', 
                    backgroundColor: colors.bgCard, 
                    padding: '30px', 
                    borderRadius: '24px', 
                    boxShadow: '0 10px 30px rgba(0,0,0,0.1)' 
                }}>
                    <div style={{ marginBottom: '25px', textAlign: 'left', paddingLeft: '20px' }}>
                        <h2 style={{ color: colors.primary, margin: '0', fontSize: '28px' }}>{currentCompany?.name}</h2>
                        <span style={{ 
                            display: 'inline-block',
                            marginTop: '5px',
                            backgroundColor: colors.secondary, 
                            color: '#fff', 
                            padding: '4px 12px', 
                            borderRadius: '12px', 
                            fontSize: '12px', 
                            fontWeight: 'bold' 
                        }}>
                            {selectedSymbol}
                        </span>
                    </div>

                    <div style={{ width: '100%', height: 400 }}>
                        <ResponsiveContainer>
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={colors.bgMain} />
                                <XAxis 
                                    dataKey="date" 
                                    tick={{fontSize: 11, fill: colors.primary}} 
                                    axisLine={{stroke: colors.primary}}
                                />
                                <YAxis 
                                    domain={['auto', 'auto']} 
                                    tick={{fontSize: 11, fill: colors.primary}}
                                    axisLine={{stroke: colors.primary}}
                                />
                                <Tooltip 
                                    contentStyle={{ 
                                        backgroundColor: colors.bgCard, 
                                        borderRadius: '12px', 
                                        border: `2px solid ${colors.secondary}`,
                                        color: colors.textDark
                                    }}
                                    itemStyle={{ color: colors.primary }}
                                    formatter={(value) => [`$${value.toFixed(2)}`, "Closing Price"]}
                                />
                                <Line 
                                    type="monotone" 
                                    dataKey="close" 
                                    stroke={colors.primary} 
                                    strokeWidth={4} 
                                    dot={false} 
                                    activeDot={{ r: 8, fill: colors.secondary, stroke: '#fff', strokeWidth: 2 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            ) : (
                <div style={{ marginTop: '100px', opacity: 0.7 }}>
                    <p style={{ fontSize: '18px', color: colors.primary }}>
                        {selectedSymbol 
                            ? "Data currently unavailable. Please check your API limits or server status." 
                            : "Welcome to StockBazaar! Select a company above to view its performance."}
                    </p>
                </div>
            )}
        </div>
    );
};

export default StockDashboard;