# AVP Data Discovery MVP (Secure, Offline)

## Quick start
```bash
cd /Users/ltraum/Documents/GitHub/avp_discovery_mvp
# (Optional) create a venv if your enclave allows:
python3 -m venv .venv && source .venv/bin/activate

# Install (must already be available in your enclave or via internal mirror):
pip install streamlit pandas numpy scipy plotly pyyaml statsmodels

# Run the app
streamlit run app.py
