"""
UI Styling for Document AI Assistant
Clean Corporate Design with High Contrast
"""

CUSTOM_CSS = """
<style>
    /* Clean Typography */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Main Content - White Background with Dark Text */
    .main {
        background-color: #FFFFFF !important;
    }
    
    /* Force Dark Text on Light Background */
    .main .stMarkdown, 
    .main .stMarkdown p, 
    .main .stMarkdown div,
    .main h1, 
    .main h2, 
    .main h3, 
    .main h4, 
    .main h5, 
    .main h6 {
        color: #1F2937 !important;
    }
    
    /* Title and Caption */
    .main [data-testid="stCaptionContainer"],
    .main [data-testid="caption"] {
        color: #6B7280 !important;
    }
    
    /* Radio Buttons - Dark Labels on Light Background */
    .main [data-testid="stRadio"] label,
    .main [data-testid="stRadio"] div,
    .main [data-testid="stRadio"] span,
    .main [data-testid="stRadio"] p {
        color: #1F2937 !important;
    }
    
    /* All Main Content Text */
    .main p,
    .main span,
    .main div,
    .main label {
        color: #1F2937 !important;
    }
    
    /* Main Area Metrics - Dark Text on White */
    .main [data-testid="stMetricValue"],
    .main [data-testid="stMetricLabel"],
    .main [data-testid="stMetricDelta"] {
        color: #1F2937 !important;
    }
    
    /* Sidebar - Light Gray Background */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Sidebar - Force ALL Text to Dark */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #333333 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: #333333 !important;
    }
    
    /* Sidebar Metrics - Force Dark Text */
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Sidebar Button Text */
    [data-testid="stSidebar"] button {
        color: #FFFFFF !important;
    }
    
    /* Chat Messages - Force Readable Contrast */
    [data-testid="stChatMessage"] {
        padding: 16px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #f0f2f6 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Force Black Text in All Chat Messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span {
        color: #000000 !important;
    }
    
    /* User Message - Subtle Gray */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #F3F4F6 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Assistant Message - White with Border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #FFFFFF !important;
        border: 1px solid #ddd !important;
    }
    
    /* Chunk Display */
    .chunk-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #3B82F6;
        padding: 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #1F2937;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Document AI Assistant",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
