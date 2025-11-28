# PII/PHI De-identification Application

A Flask web application for de-identifying Protected Health Information (PHI) and Personally Identifiable Information (PII) from clinical text using a fine-tuned ClinicalBERT model.

## Features

- Automatic detection of PHI/PII entities (names, dates, locations, etc.)
- Surrogate data generation for consistent de-identification
- Web-based interface for easy text processing
- Support for staff vs. patient name classification
- Date shifting for temporal anonymization

## Technology Stack

- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face library for BERT models
- **ClinicalBERT** - Fine-tuned model (`obi/deid_bert_i2b2`) for clinical NER

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/rajellumen/pii-phi.git
cd pii-phi
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
# Create a .env file or set environment variables
FLASK_SECRET_KEY=your-secret-key-here
DEID_API_KEY=your-api-key-here  # Optional, for API endpoint protection
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:9000`

## Deployment on Render

### Prerequisites

1. GitHub repository connected (already done)
2. Render account

### Deployment Steps

1. **Create a new Web Service on Render:**
   - Go to your Render dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `rajellumen/pii-phi`
   - Select the repository

2. **Configure the service:**
   - **Name**: `pii-phi` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Choose based on your needs (Free tier available, but may have limitations)

3. **Set Environment Variables:**
   In the Render dashboard, add these environment variables:
   - `FLASK_SECRET_KEY`: A secure random string (generate one)
   - `DEID_API_KEY`: (Optional) API key for protecting the `/deidentify` endpoint
   - `PORT`: (Auto-set by Render, but you can verify it's available)

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - The first deployment may take longer as it downloads the BERT model

### Important Notes for Render Deployment

- **Model Download**: The first deployment will download the BERT model (~500MB), which may take 5-10 minutes
- **Memory Requirements**: The model requires significant memory. Consider using at least 512MB RAM (Free tier has 512MB)
- **Build Time**: Initial build may take 10-15 minutes due to PyTorch and model downloads
- **Cold Starts**: Free tier services spin down after inactivity, causing a cold start delay

### Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `FLASK_SECRET_KEY` | Yes | Secret key for Flask session management |
| `DEID_API_KEY` | No | API key for `/deidentify` endpoint protection |
| `PORT` | Auto | Port number (automatically set by Render) |

## Usage

1. Navigate to the web interface
2. Paste or type clinical text containing PHI/PII
3. Click "De-identify Text"
4. View the de-identified text with highlighted changes
5. Review the changes log showing all replacements

## Model Information

- **Model**: `obi/deid_bert_i2b2`
- **Purpose**: Named Entity Recognition for clinical text
- **Entity Types**: PATIENT, STAFF, DATE, LOCATION, etc.

## License

[Add your license here]

## Author

Raj Anantharaman  
Email: ranantharaman@ellumen.com

