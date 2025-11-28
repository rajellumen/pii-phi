from flask import Flask, request, jsonify, render_template
from deidentifier import ClinicalDeidentifier
import os
import re
from html import escape

app = Flask(__name__)

# --- Configuration & Security (CRITICAL for Healthcare Data) ---
# 1. Load Secret Key from Environment Variables
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
# 2. Use HTTPS in Production (Enforced by reverse proxy like Nginx/Load Balancer)
# 3. Implement robust Authentication (e.g., JWT/API Key validation) for /deidentify
API_KEY = os.environ.get('DEID_API_KEY')


# Initialize the Deidentifier (Global singleton for efficiency)
# This loads the large BERT model into memory once.
try:
    deidentifier_service = ClinicalDeidentifier()
except Exception as e:
    print(f"Error loading ClinicalBERT model: {e}")
    deidentifier_service = None

def highlight_phi_in_text(text, changes):
    """
    Highlights PHI entities in the text by wrapping them in HTML span tags.
    Returns the text with highlighted PHI.
    """
    if not changes or not text:
        return escape(text)
    
    highlighted_text = escape(text)
    
    # Create a mapping of original -> change info
    phi_map = {}
    for change in changes:
        original = change['original']
        # Use the original as key, but keep the change info
        if original not in phi_map:
            phi_map[original] = change
    
    # Sort by length (longest first) to avoid partial matches
    sorted_originals = sorted(phi_map.keys(), key=len, reverse=True)
    
    # Highlight each unique PHI entity (find all occurrences)
    # Process longest matches first to avoid partial matches
    for original in sorted_originals:
        change = phi_map[original]
        # Escape the original for regex, but preserve word boundaries
        escaped_original = re.escape(original)
        
        # Try to match as whole word/phrase first
        # For multi-word phrases, use word boundaries
        if ' ' in original or len(original) > 10:
            # For phrases or long words, match exactly
            pattern = escaped_original
        else:
            # For single words, use word boundaries
            pattern = r'\b' + escaped_original + r'\b'
        
        # Replace with highlighted version
        # Use a function to check if match is already inside a highlight
        def make_replacement(m):
            # Get the text before this match
            match_start = m.start()
            text_before = highlighted_text[:match_start]
            
            # Check if we're inside a highlight span by counting tags
            # Find the last unclosed <span class="phi-highlight"> before this position
            last_open = text_before.rfind('<span class="phi-highlight"')
            if last_open != -1:
                # Found an opening tag, check if it's closed before our match
                text_after_open = text_before[last_open:]
                # Count opening and closing tags in the text after the last opening tag
                # If there are more opening tags than closing tags, we're inside a highlight
                open_count = text_after_open.count('<span class="phi-highlight"')
                close_count = text_after_open.count('</span>')
                if open_count > close_count:
                    return m.group(0)  # Already inside a highlight, don't highlight again
            
            return f'<span class="phi-highlight" data-type="{change["type"]}" title="{change["type"]}: {escape(change["original"])} → {escape(change["replacement"])}">{m.group(0)}</span>'
        
        highlighted_text = re.sub(
            pattern,
            make_replacement,
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    return highlighted_text

def highlight_replacements_in_text(text, changes):
    """
    Highlights replacement text in de-identified text.
    """
    if not changes or not text:
        return escape(text)
    
    highlighted_text = escape(text)
    
    # Create a mapping of replacement -> change info
    replacement_map = {}
    for change in changes:
        replacement = change['replacement']
        if replacement not in replacement_map:
            replacement_map[replacement] = change
    
    # Sort by length (longest first) to avoid partial matches
    sorted_replacements = sorted(replacement_map.keys(), key=len, reverse=True)
    
    # Highlight each replacement
    # Process longest matches first to avoid partial matches
    for replacement in sorted_replacements:
        change = replacement_map[replacement]
        # Escape the replacement for regex
        escaped_replacement = re.escape(replacement)
        
        # Try to match as whole word/phrase first
        # For multi-word phrases, match exactly
        if ' ' in replacement or len(replacement) > 10:
            pattern = escaped_replacement
        else:
            # For single words, use word boundaries
            pattern = r'\b' + escaped_replacement + r'\b'
        
        # Replace with highlighted version
        # Use a function to check if match is already inside a highlight
        def make_replacement(m):
            # Get the text before this match
            match_start = m.start()
            text_before = highlighted_text[:match_start]
            
            # Check if we're inside a highlight span by counting tags
            # Find the last unclosed <span class="replacement-highlight"> before this position
            last_open = text_before.rfind('<span class="replacement-highlight"')
            if last_open != -1:
                # Found an opening tag, check if it's closed before our match
                text_after_open = text_before[last_open:]
                # Count opening and closing tags in the text after the last opening tag
                # If there are more opening tags than closing tags, we're inside a highlight
                open_count = text_after_open.count('<span class="replacement-highlight"')
                close_count = text_after_open.count('</span>')
                if open_count > close_count:
                    return m.group(0)  # Already inside a highlight, don't highlight again
            
            return f'<span class="replacement-highlight" data-type="{change["type"]}" title="Replaced: {escape(change["original"])} → {escape(change["replacement"])}">{m.group(0)}</span>'
        
        highlighted_text = re.sub(
            pattern,
            make_replacement,
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    return highlighted_text
    
# --- API Routes ---

# --- API Routes ---

# 1. The Home Page Route (GET)
@app.route('/', methods=['GET'])
def index():
    """Renders the index.html template for the user to input text."""
    # Pass empty variables for the initial load
    return render_template('index.html', 
                           original_text="", 
                           original_highlighted="",
                           deidentified_text="",
                           deidentified_highlighted="",
                           changes=[])

# 2. The De-identification Endpoint (POST)
# We will still use the existing /deidentify route to process the text
@app.route('/deidentify', methods=['POST'])
def deidentify_report():
    # 1. API Key/Authentication Check (Still necessary for robustness)
    # NOTE: You'd typically use a browser session/cookie for a web app, 
    # but we'll keep the API key check here for consistency with the API-only design.
    if request.headers.get('X-API-KEY') != API_KEY:
         # For a web request, return an error to the template
         if 'report_text' not in request.form:
             return jsonify({"error": "Unauthorized"}), 401
    
    if deidentifier_service is None:
        return render_template('index.html', error="De-identification service is unavailable"), 503

    # Use request.form to get data from a standard HTML POST form
    raw_text = request.form.get('report_text', '')

    if not raw_text:
        # Re-render the form if text is empty
        return render_template('index.html', error="Please provide a report for de-identification."), 400

    try:
        # 3. Call the De-identification Logic (Hypothetical update to return changes)
        # NOTE: Your actual deidentifier.py needs to be updated to return a list of changes.
        deidentified_text, changes = deidentifier_service.deidentify_with_changes(raw_text)
        
        # 4. Create highlighted versions of the texts
        original_highlighted = highlight_phi_in_text(raw_text, changes)
        deidentified_highlighted = highlight_replacements_in_text(deidentified_text, changes)
        
        # 5. Re-render the index.html template with the results
        return render_template('index.html', 
                               original_text=raw_text, 
                               original_highlighted=original_highlighted,
                               deidentified_text=deidentified_text,
                               deidentified_highlighted=deidentified_highlighted,
                               changes=changes,
                               error=None), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Internal processing error: {e}")
        print(f"Traceback: {error_trace}")
        return render_template('index.html', 
                               error=f"An internal error occurred: {str(e)}"), 500
if __name__ == '__main__':
    # Use a production-grade WSGI server (like Gunicorn) for real deployment
    # This is for local testing only:
    # Set FLASK_SECRET_KEY and DEID_API_KEY environment variables before running
    app.run(debug=True, host='0.0.0.0', port=9000)