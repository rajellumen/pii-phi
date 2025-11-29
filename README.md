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

## Deployment on Amazon EC2

### Prerequisites

1. AWS EC2 instance (Ubuntu recommended)
2. Python 3.10+ installed
3. Git installed

### Deployment Steps

1. **Connect to your EC2 instance:**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

2. **Clone the repository:**
```bash
git clone https://github.com/rajellumen/pii-phi.git
cd pii-phi
```

3. **Set up Python virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Set environment variables:**
```bash
export FLASK_SECRET_KEY="your-secret-key-here"
export DEID_API_KEY="your-api-key-here"  # Optional
export PORT=9000  # Or your preferred port
```

6. **Run with Gunicorn (Production):**
```bash
gunicorn -w 2 -b 0.0.0.0:9000 app:app
```

7. **Or run as a systemd service (Recommended):**
   - Create `/etc/systemd/system/pii-phi.service`:
   ```ini
   [Unit]
   Description=PII/PHI De-identification Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/pii-phi
   Environment="PATH=/home/ubuntu/pii-phi/venv/bin"
   Environment="FLASK_SECRET_KEY=your-secret-key-here"
   Environment="DEID_API_KEY=your-api-key-here"
   ExecStart=/home/ubuntu/pii-phi/venv/bin/gunicorn -w 2 -b 0.0.0.0:9000 app:app

   [Install]
   WantedBy=multi-user.target
   ```
   - Enable and start:
   ```bash
   sudo systemctl enable pii-phi
   sudo systemctl start pii-phi
   ```

### Important Notes for EC2 Deployment

- **Security Groups**: Ensure port 9000 (or your chosen port) is open in EC2 Security Groups
- **Model Download**: First run will download the BERT model (~500MB), which may take 5-10 minutes
- **Memory Requirements**: The model requires significant memory. Use at least 2GB RAM instance
- **HTTPS**: Use a reverse proxy (nginx) with SSL certificate for production
- **Firewall**: Configure ufw to allow your port:
  ```bash
  sudo ufw allow 9000/tcp
  ```

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

