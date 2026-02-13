import streamlit as st
from views.predict import show_predict

st.set_page_config(
    page_title="Plant Health Intelligence System",
    page_icon="🌿",
    layout="wide"
)

# Sidebar
st.sidebar.title("🌿 Plant Health AI")
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Visualization", "Predict", "Settings", "About"]

)

# Routing
if page == "Dashboard":
    from views.dashboard import show_dashboard
    show_dashboard()

elif page == "Visualization":
    from views.visualization import show_visualization
    show_visualization()

elif page == "Settings":
    from views.settings import show_settings
    show_settings()

elif page == "About":
    from views.about import show_about
    show_about()

elif page == "Predict":
    from views.predict import show_predict
    show_predict()



st.markdown("""
---
<center>
Plant Health Intelligence System © 2026  
Built with ❤️ using Streamlit
</center>
""", unsafe_allow_html=True)
