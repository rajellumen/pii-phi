import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from random import randint
from datetime import datetime, timedelta
import re

class ClinicalDeidentifier:
    def __init__(self, model_name="obi/deid_bert_i2b2"):
        # Load the fine-tuned ClinicalBERT model and tokenizer
        # This model is pre-trained for PHI/de-identification NER
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id_to_label = self.model.config.id2label

        # Dictionaries for Surrogate Replacement (Expand these significantly for production!)
        # In deidentifier.py, inside the ClinicalDeidentifier.__init__ method
        # In deidentifier.py, inside the ClinicalDeidentifier.__init__ method
        self.surrogate_pools = {
            # Increase these lists to at least 20-50 unique items each
            # PATIENT names should be full names (first + last) to look more realistic
            "PATIENT": ["Robert Johnson", "Linda Williams", "Michael Brown", "Sarah Davis", "David Miller", "Jessica Wilson", 
                       "Thomas Moore", "Laura Taylor", "Jacob Anderson", "Emily Jackson", "James White", "Patricia Harris", 
                       "John Martin", "Jennifer Thompson", "William Garcia", "Mary Martinez", "Richard Robinson", "Elizabeth Clark", 
                       "Joseph Lewis", "Barbara Walker", "Daniel Hall", "Susan Allen", "Matthew Young", "Karen King", 
                       "Anthony Wright", "Nancy Lopez", "Mark Hill", "Lisa Scott", "Donald Green", "Betty Adams"],
            "STAFF": ["Dr. Smith", "Nurse Jones", "Physician Miller", "Dr. Lee", "Dr. Brown", "Nurse Williams", 
                     "Dr. Davis", "Physician Wilson", "Dr. Moore", "Nurse Taylor", "Dr. Anderson", "Physician Jackson",
                     "Dr. White", "Nurse Harris", "Dr. Martin", "Physician Thompson", "Dr. Garcia", "Nurse Martinez",
                     "Dr. Robinson", "Physician Clark"], 
            "LOC": ["City Hospital", "St. Jude Medical Center", "Regional Trauma Unit", "General Clinic", "North Wing Office",
                   "Memorial Hospital", "Community Health Center", "Regional Medical Center", "General Hospital", "Central Clinic",
                   "Valley Medical Center", "Riverside Hospital", "Parkview Clinic", "Sunset Medical Center", "Oakwood Hospital",
                   "Pine Valley Clinic", "Lakeside Medical Center", "Hilltop Hospital", "Greenwood Clinic", "Ridgeview Medical Center"],
        }

        # NEW: Dictionary to store the unique mapping for each report
        # This MUST be cleared for every new report.
        self.phi_to_surrogate_map = {}

        # Key for Date Shifting (Must be consistent per patient/record)
        # This shift MUST be applied to ALL dates in the report to preserve chronology.
        self.date_shift_days = randint(365, 1095) # A random shift between 1 and 3 years
        
        # Name variations/nicknames mapping
        # Maps canonical names to their common variations/nicknames
        # This allows linking "Robert" with "Rob" or "Bob" as the same person
        self.name_variations = {
            # Robert variations
            'robert': ['rob', 'bob', 'robby', 'bobby', 'robbie'],
            'rob': ['robert', 'bob'],
            'bob': ['robert', 'rob'],
            'robby': ['robert'],
            'bobby': ['robert'],
            'robbie': ['robert'],
            # Charles variations
            'charles': ['chuck', 'charlie', 'chaz'],
            'chuck': ['charles', 'charlie'],
            'charlie': ['charles', 'chuck'],
            'chaz': ['charles'],
            # William variations
            'william': ['will', 'bill', 'billy', 'willy', 'wil'],
            'will': ['william', 'bill'],
            'bill': ['william', 'will', 'billy'],
            'billy': ['william', 'bill'],
            'willy': ['william'],
            'wil': ['william'],
            # James variations
            'james': ['jim', 'jimmy', 'jamie'],
            'jim': ['james', 'jimmy'],
            'jimmy': ['james', 'jim'],
            'jamie': ['james'],
            # John variations
            'john': ['jack', 'johnny', 'jon'],
            'jack': ['john', 'johnny'],
            'johnny': ['john', 'jack'],
            'jon': ['john'],
            # Michael variations
            'michael': ['mike', 'mikey', 'mick', 'mickey'],
            'mike': ['michael', 'mikey'],
            'mikey': ['michael', 'mike'],
            'mick': ['michael'],
            'mickey': ['michael'],
            # Richard variations
            'richard': ['rick', 'ricky', 'rich', 'dick'],
            'rick': ['richard', 'ricky'],
            'ricky': ['richard', 'rick'],
            'rich': ['richard'],
            'dick': ['richard'],
            # Joseph variations
            'joseph': ['joe', 'joey'],
            'joe': ['joseph', 'joey'],
            'joey': ['joseph', 'joe'],
            # Thomas variations
            'thomas': ['tom', 'tommy'],
            'tom': ['thomas', 'tommy'],
            'tommy': ['thomas', 'tom'],
            # Daniel variations
            'daniel': ['dan', 'danny'],
            'dan': ['daniel', 'danny'],
            'danny': ['daniel', 'dan'],
            # Christopher variations
            'christopher': ['chris', 'christy'],
            'chris': ['christopher', 'christy'],
            'christy': ['christopher', 'chris'],
            # Andrew variations
            'andrew': ['andy', 'drew'],
            'andy': ['andrew', 'drew'],
            'drew': ['andrew', 'andy'],
            # Edward variations
            'edward': ['ed', 'eddie', 'ted'],
            'ed': ['edward', 'eddie'],
            'eddie': ['edward', 'ed'],
            'ted': ['edward'],
            # Robert (last name can also be Rob/Bob in context)
            # Add more as needed
        }


          
    def _get_surrogate(self, tag: str, original_entity: str, mapping_key: tuple) -> str:
        """Applies the hide-in-plain-sight replacement logic, using mapping_key for consistency."""

        # --- 1. Check if the entity is ALREADY MAPPED ---
        # Use the mapping_key (which includes Type and Original Entity)
        if mapping_key in self.phi_to_surrogate_map:
            stored_surrogate = self.phi_to_surrogate_map[mapping_key]
            
            # CRITICAL: If the original entity doesn't have a title but the stored surrogate does,
            # extract just the name part (without title) to avoid duplication like "Ms. Richard Robinson. Richard Robinson"
            original_has_title = any(title in original_entity.lower() for title in ['ms.', 'mr.', 'mrs.', 'miss.', 'dr.', 'doctor.', 'prof.', 'professor.'])
            if not original_has_title:
                # Check if stored surrogate has a title
                title_pattern = r'^(Ms\.|Mr\.|Mrs\.|Miss\.|Dr\.|Doctor\.|Prof\.|Professor\.)\s+'
                if re.match(title_pattern, stored_surrogate, re.IGNORECASE):
                    # Extract just the name part (remove title)
                    name_only = re.sub(title_pattern, '', stored_surrogate, flags=re.IGNORECASE).strip()
                    # Update the mapping to use name-only version for consistency
                    self.phi_to_surrogate_map[mapping_key] = name_only
                    return name_only
            
            return stored_surrogate

        # --- 2. Handle Dates (Deterministic Shifting) ---
        if tag == "DATE":
            # Check if already mapped
            if mapping_key in self.phi_to_surrogate_map:
                return self.phi_to_surrogate_map[mapping_key]
            
            # Try to parse and shift the date
            try:
                # Common date formats: M/D/YY, M/D/YYYY, MM/DD/YYYY, MM/DD/YY
                date_formats = [
                    "%m/%d/%y",      # 3/4/20
                    "%m/%d/%Y",      # 3/4/2020
                    "%m/%d/%y",      # 12/31/89
                    "%m/%d/%Y",      # 12/31/1989
                ]
                
                parsed_date = None
                original_format = None
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(original_entity, fmt)
                        original_format = fmt
                        break
                    except ValueError:
                        continue
                
                if parsed_date:
                    # Shift the date
                    shifted_date = parsed_date + timedelta(days=self.date_shift_days)
                    
                    # Format back in the same style as original
                    if "%y" in original_format and "%Y" not in original_format:
                        # Two-digit year format
                        shifted_date_str = shifted_date.strftime("%m/%d/%y")
                    else:
                        # Four-digit year format
                        shifted_date_str = shifted_date.strftime("%m/%d/%Y")
                    
                    # Store mapping
                    self.phi_to_surrogate_map[mapping_key] = shifted_date_str
                    return shifted_date_str
                else:
                    # If parsing fails, use redaction
                    return f"[{tag}_REDACTED]"
            except Exception as e:
                # If any error occurs, use redaction
                return f"[{tag}_REDACTED]"
            
        # --- 3. Handle Pool Replacements (If not found in map) ---
        elif tag in self.surrogate_pools and self.surrogate_pools[tag]:
            
            # Use a deterministic index (e.g., hash) to pick a NEW surrogate
            token_hash = hash(original_entity)
            pool = self.surrogate_pools[tag]
            index = token_hash % len(pool)
            
            new_surrogate = pool[index]
            
            # Store the new mapping before returning! Use mapping_key for consistency
            self.phi_to_surrogate_map[mapping_key] = new_surrogate
            
            return new_surrogate
            
        # 4. Final fallback
        else:
            return f"[{tag}_REDACTED]"

    

    def deidentify_with_changes(self, text: str):
        """
        Performs span-based PHI detection and replacement, returning 
        the de-identified text and a list of changes.
        """
        
        # Clear the map for a new report to ensure new, unique surrogates are chosen
        self.phi_to_surrogate_map = {} # CRITICAL: Reset mapping for each new report

        # PRE-PROCESSING: Fix split names before tokenization
        # This handles cases like "G rac owere" -> "Graco were"
        text = self._fix_split_names(text)

        # --- 1. Model Inference (PHI Detection) ---
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions_tensor = torch.argmax(outputs.logits, dim=2)
        
        # Get the full lists for easy lookahead outside the loop
        predictions = predictions_tensor.squeeze().tolist()
        input_ids_squeezed = inputs["input_ids"].squeeze()
        
        if input_ids_squeezed.dim() == 0:
            original_tokens = self.tokenizer.convert_ids_to_tokens([input_ids_squeezed.item()])
            predictions = [predictions] if not isinstance(predictions, list) else predictions
        else:
            original_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_squeezed.tolist())

        # --- 2. Span Collection and Processing ---
        
        deidentified_text = []
        changes = []
        current_entity = [] 
        current_entity_tag = None
        i = 0 # Use a manual index 'i' for lookahead inside the while loop
        processed_i_token_indices = set()  # Track I- token indices that were already added by lookahead

        while i < len(original_tokens):
            token = original_tokens[i]
            pred_id = predictions[i]
            label = self.id_to_label.get(pred_id, "O")
            
            # Skip special tokens (e.g., [CLS], [SEP]) - don't add them to output
            if token in self.tokenizer.all_special_tokens:
                if current_entity:
                    self._process_entity_span(current_entity, current_entity_tag, deidentified_text, changes, original_text=text)
                    current_entity = []
                    current_entity_tag = None
                # Skip special tokens - don't append them to output
                i += 1 # Advance index
                continue
                
            tag = label.split('-')[-1].upper() if label != "O" else None

            # A. If we see an OUTSIDE token (O)
            if label == "O":
                if current_entity:
                    # Process the entity span that just ended
                    # Calculate entity position for context checking
                    entity_start = i - len(current_entity)
                    entity_end = i
                    self._process_entity_span(current_entity, current_entity_tag, deidentified_text, changes, original_tokens, entity_start, entity_end, original_text=text)
                    current_entity = []
                    current_entity_tag = None
                    
                # Append the outside token WITH ## prefix intact
                # convert_tokens_to_string expects tokens in BERT format
                deidentified_text.append(token)
                
            # B. If we see a BEGINNING token (B-)
            elif label.startswith('B-'):
                if current_entity:
                    # An explicit B- always signals the END of the previous span
                    # Calculate entity position for context checking
                    entity_start = i - len(current_entity)
                    entity_end = i
                    self._process_entity_span(current_entity, current_entity_tag, deidentified_text, changes, original_tokens, entity_start, entity_end, original_text=text)
                
                # Skip B-STAFF tokens - staff names are not PHI/PII
                if tag == 'STAFF':
                    # Append the STAFF token as-is (no replacement)
                    deidentified_text.append(token)
                    i += 1
                    continue
                
                # Start a new span
                current_entity = [token]
                current_entity_tag = tag
                # Note: Context checking happens later in _process_entity_span when we have the full entity

                
                # --- HEURISTIC FIX: Force Last Name Continuation ---
                # Addresses "Heidi Smith" and "Robert Graco" problems where last names are classified as 'O' or split into subwords
                if tag in ('PATIENT', 'STAFF', 'PHYSICIAN') and (i + 1) < len(original_tokens):
                    # Look ahead to collect potential last name tokens (handles both single tokens and subword splits)
                    lookahead_idx = i + 1
                    potential_last_name_tokens = []
                    potential_last_name_labels = []
                    i_tokens_found = []  # Track I- tokens separately
                    i_tokens_indices = set()  # Track indices of I- tokens we've added
                    
                    # Collect tokens that could be part of the last name
                    # Stop when we hit punctuation, a different word type, or max lookahead
                    # Increased to 8 to handle cases like "HEIDI SMITH" which tokenizes as 'H', '##EI', '##DI', 'SM', '##IT', '##H', ','
                    max_lookahead = min(i + 8, len(original_tokens))  # Look ahead up to 7 more tokens
                    while lookahead_idx < max_lookahead:
                        lookahead_token = original_tokens[lookahead_idx]
                        lookahead_label = self.id_to_label.get(predictions[lookahead_idx], "O")
                        lookahead_tag = lookahead_label.split('-')[-1].upper() if lookahead_label != "O" else None
                        
                        # Stop if we hit punctuation (unless it's part of a subword)
                        # BUT: Don't stop at comma if it might be followed by staff keywords (e.g., "HEIDI SMITH, Operator:")
                        if (not lookahead_token.startswith('##') and 
                            not lookahead_token.isalnum() and 
                            lookahead_token not in ["'", "-"]):  # Allow apostrophes and hyphens in names
                            # Check if this is a comma that might be followed by staff keywords
                            if lookahead_token == ',' and lookahead_idx + 1 < len(original_tokens):
                                # Look ahead to see if it's "Operator" or similar (handle tokenization like "Opera" + "##tor")
                                # Check next few tokens to reconstruct "Operator"
                                next_tokens_check = []
                                check_idx = lookahead_idx + 1
                                max_check = min(check_idx + 3, len(original_tokens))
                                while check_idx < max_check:
                                    next_tokens_check.append(original_tokens[check_idx].replace('##', ''))
                                    check_idx += 1
                                reconstructed_next = ''.join(next_tokens_check).lower()
                                # Check if it starts with staff keywords
                                if reconstructed_next.startswith(('operator', 'dr', 'doctor', 'nurse', 'physician')):
                                    # This comma is followed by a staff keyword - include it and continue
                                    potential_last_name_tokens.append(lookahead_token)
                                    potential_last_name_labels.append(lookahead_label)
                                    lookahead_idx += 1
                                    continue
                            # Otherwise, stop at punctuation
                            break
                        
                        # Stop at common abbreviations that often follow names
                        # Also check for partial tokens that might be part of abbreviations
                        if (not lookahead_token.startswith('##') and 
                            lookahead_token.upper() in ['DOB', 'SSN', 'MRN', 'ID', 'PHONE', 'EMAIL', 'DO', 'D']):  # 'DO' and 'D' in case DOB is split
                            break
                        
                        # Check if current token + next subword(s) would form an abbreviation
                        # This catches cases like "D" + "##OB" = "DOB" or "DO" + "##B" = "DOB"
                        if (not lookahead_token.startswith('##') and 
                            lookahead_token.isalpha() and
                            lookahead_idx + 1 < len(original_tokens)):
                            next_token = original_tokens[lookahead_idx + 1]
                            
                            # Check if token + next subword forms abbreviation
                            if next_token.startswith('##'):
                                potential_abbrev = lookahead_token + next_token[2:]
                                if potential_abbrev.upper() in ['DOB', 'SSN', 'MRN', 'ID']:
                                    break
                            
                            # Also check if token + next token + following subword forms abbreviation (e.g., "D" + "O" + "##B")
                            if (lookahead_idx + 2 < len(original_tokens) and
                                next_token.isalpha() and len(next_token) == 1 and
                                original_tokens[lookahead_idx + 2].startswith('##')):
                                potential_abbrev = lookahead_token + next_token + original_tokens[lookahead_idx + 2][2:]
                                if potential_abbrev.upper() in ['DOB', 'SSN', 'MRN', 'ID']:
                                    break
                        
                        # If it's a subword (##), check if we're building a reasonable name
                        if lookahead_token.startswith('##'):
                            # For I-PATIENT/I-STAFF tokens, we need to track them so we can include them in the entity
                            # Don't add them to potential_last_name_tokens, but track them separately
                            if lookahead_tag == tag:
                                # This is an I- token of the same type - track it so we can add it to the entity
                                i_tokens_found.append(lookahead_token)
                                i_tokens_indices.add(lookahead_idx)  # Track the index so we can skip it in the main loop
                                lookahead_idx += 1
                                continue  # Continue to look for more tokens (like 'O' tokens that might be last names)
                            
                            # For 'O' subwords, check if they're part of a name
                            # We can have tokens in current_entity (the B- token) or in potential_last_name_tokens (from lookahead)
                            all_tokens_so_far = list(current_entity) + potential_last_name_tokens
                            if all_tokens_so_far:
                                # Reconstruct what we have so far
                                reconstructed_so_far = "".join(t.replace('##', '') for t in all_tokens_so_far + [lookahead_token])
                                
                                # Check if it's a known abbreviation (like DOB, SSN, etc.)
                                if reconstructed_so_far.upper() in ['DOB', 'SSN', 'MRN', 'ID', 'PHONE', 'EMAIL', 'FAX']:
                                    # This is an abbreviation, not a name - stop here
                                    break
                                
                                # Otherwise, include the subword (it's 'O' and looks like a name part)
                                potential_last_name_tokens.append(lookahead_token)
                                potential_last_name_labels.append(lookahead_label)
                                lookahead_idx += 1
                                continue
                            else:
                                # Subword without a base word - might be part of something else, stop
                                break
                        
                        # If it's a regular token, check if it looks like a name part
                        # For single letters, only include if followed by subwords (part of a split name)
                        is_single_letter = lookahead_token.isalpha() and len(lookahead_token) == 1
                        
                        # Before including any token, check if it would form an abbreviation
                        # Check if current token + next subword would form abbreviation
                        would_form_abbrev = False
                        if lookahead_token.isalpha() and lookahead_idx + 1 < len(original_tokens):
                            next_token_check = original_tokens[lookahead_idx + 1]
                            if next_token_check.startswith('##'):
                                # Check if token + subword forms abbreviation
                                potential_abbrev = lookahead_token + next_token_check[2:]
                                if potential_abbrev.upper() in ['DOB', 'SSN', 'MRN', 'ID', 'PHONE', 'EMAIL']:
                                    would_form_abbrev = True
                            elif (is_single_letter and 
                                  lookahead_idx + 2 < len(original_tokens) and
                                  next_token_check.isalpha() and len(next_token_check) == 1 and
                                  original_tokens[lookahead_idx + 2].startswith('##')):
                                # Check if "D" + "O" + "##B" = "DOB"
                                potential_abbrev = lookahead_token + next_token_check + original_tokens[lookahead_idx + 2][2:]
                                if potential_abbrev.upper() in ['DOB', 'SSN', 'MRN', 'ID']:
                                    would_form_abbrev = True
                        
                        # Also check if what we have so far + this token would form an abbreviation
                        if not would_form_abbrev and potential_last_name_tokens:
                            reconstructed_with_token = "".join(t.replace('##', '') for t in potential_last_name_tokens + [lookahead_token])
                            if reconstructed_with_token.upper() in ['DOB', 'SSN', 'MRN', 'ID', 'PHONE', 'EMAIL']:
                                would_form_abbrev = True
                        
                        # If it would form an abbreviation, stop here
                        if would_form_abbrev:
                            break
                        
                        # Check if this is a capitalized word that could be a last name
                        # For all-caps names like "SMITH", check if it's all uppercase
                        is_capitalized = (lookahead_token[0].isupper() if lookahead_token else False) or lookahead_token.isupper()
                        
                        if (lookahead_token.isalpha() and 
                            (not is_single_letter or (lookahead_idx + 1 < len(original_tokens) and original_tokens[lookahead_idx + 1].startswith('##'))) and
                            is_capitalized and
                            lookahead_token.upper() not in ['THE', 'AND', 'OR', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'OPERATOR', 'PATIENT', 'DOCTOR', 'NURSE', 'DOB', 'SSN', 'MRN', 'D', 'DO']):
                            
                            # Check if it's 'O' or a compatible PHI type
                            if (lookahead_label == "O" or lookahead_tag == tag or lookahead_tag in ('PATIENT', 'STAFF', 'PHYSICIAN')):
                                # Include this token
                                potential_last_name_tokens.append(lookahead_token)
                                potential_last_name_labels.append(lookahead_label)
                                lookahead_idx += 1
                                
                                # After including the token, check if next token is a comma followed by staff keyword
                                # This handles "SMITH, Operator:" pattern
                                if lookahead_idx < len(original_tokens):
                                    next_token = original_tokens[lookahead_idx]
                                    # Check if next token is a comma
                                    if next_token == ',' and lookahead_idx + 1 < len(original_tokens):
                                        # Look ahead to see if it's "Operator" or similar (handle tokenization like "Opera" + "##tor")
                                        next_tokens_check = []
                                        check_idx = lookahead_idx + 1
                                        max_check = min(check_idx + 3, len(original_tokens))
                                        while check_idx < max_check:
                                            next_tokens_check.append(original_tokens[check_idx].replace('##', ''))
                                            check_idx += 1
                                        reconstructed_next = ''.join(next_tokens_check).lower()
                                        # Check if it starts with staff keywords
                                        if reconstructed_next.startswith(('operator', 'dr', 'doctor', 'nurse', 'physician')):
                                            # This comma is followed by a staff keyword - include the comma and stop
                                            # We've already captured the full name (e.g., "SMITH")
                                            potential_last_name_tokens.append(next_token)
                                            potential_last_name_labels.append(self.id_to_label.get(predictions[lookahead_idx], "O"))
                                            lookahead_idx += 1
                                            break  # Stop here, we have the full name
                                
                                # Continue to catch subword continuations
                                continue
                            else:
                                # Different classification, stop here
                                break
                        else:
                            # Doesn't look like a name part, stop
                            break
                    
                    # If we collected potential last name tokens, add them to the entity
                    # Always add I- tokens to the entity if we found any, even if we didn't find a last name
                    # This ensures "HEIDI" (tokenized as 'H', '##EI', '##DI') is captured as one entity
                    # IMPORTANT: Also check for any I-tokens that might have been missed (e.g., if model classifies inconsistently)
                    if i_tokens_found:
                        current_entity.extend(i_tokens_found)
                        # Store i_tokens_indices so the main loop can skip these tokens
                        processed_i_token_indices.update(i_tokens_indices)
                    else:
                        # Even if lookahead didn't find I-tokens, check if there are any I-tokens immediately following
                        # This handles cases where the model classifies tokens inconsistently
                        check_idx = i + 1
                        while check_idx < len(original_tokens) and check_idx < i + 5:  # Check up to 4 tokens ahead
                            check_label = self.id_to_label.get(predictions[check_idx], "O")
                            # Accept I-tokens of the same type OR I-tokens that are part of a name (PATIENT/STAFF)
                            if check_label.startswith('I-'):
                                check_tag = check_label.split('-')[-1].upper()
                                # If it's the same type, or if it's PATIENT/STAFF (likely part of the same name)
                                if check_tag == tag or (check_tag in ('PATIENT', 'STAFF') and tag in ('PATIENT', 'STAFF')):
                                    check_token = original_tokens[check_idx]
                                    if check_token.startswith('##'):  # Only add subword tokens
                                        current_entity.append(check_token)
                                        processed_i_token_indices.add(check_idx)
                                        check_idx += 1
                                        continue
                            break
                    
                    # If we found I-tokens but no last name, we still need to skip past the I-tokens
                    if i_tokens_found and not potential_last_name_tokens:
                        # Find the last I-token index we added
                        last_i_token_idx = i
                        for check_idx in range(i + 1, lookahead_idx):
                            check_label = self.id_to_label.get(predictions[check_idx], "O")
                            if check_label.startswith('I-') and check_label.split('-')[-1].upper() == tag:
                                last_i_token_idx = check_idx
                            else:
                                break
                        i = last_i_token_idx  # Will be incremented at end of loop
                    
                    if potential_last_name_tokens:
                        # Reconstruct the full last name to check if it's reasonable
                        full_last_name = "".join(t.replace('##', '') for t in potential_last_name_tokens)
                        
                        # Only include if it looks like a reasonable last name (at least 2 chars, mostly letters)
                        # Reduced from 3 to 2 to catch shorter last names like "Graco"
                        if len(full_last_name) >= 2 and full_last_name.replace("'", "").replace("-", "").isalpha():
                            # Add the 'O' tokens we captured (I-tokens were already added above)
                            current_entity.extend(potential_last_name_tokens)
                            
                            # Set i to skip past all the tokens we've added (I- tokens + 'O' tokens)
                            # The loop will increment i, so we need to set it to the position before the next unprocessed token
                            i = lookahead_idx - 1  # Will be incremented at end of loop
                # --- END HEURISTIC FIX ---
                
                # --- HEURISTIC FIX: Force Location Name Continuation ---
                # Addresses "Adventure Bay" and "Spring Creek" problems where location names are split
                # Handles multi-word location names like "Adventure Bay", "Spring Creek", "Bay Point"
                if tag in ('LOC', 'HOSP', 'PATORG') and (i + 1) < len(original_tokens):
                    lookahead_idx = i + 1
                    potential_location_tokens = []
                    
                    # Collect tokens that could be part of the location name
                    # Stop when we hit punctuation, a different word type, or max lookahead
                    max_lookahead = min(i + 4, len(original_tokens))  # Look ahead up to 3 more tokens
                    while lookahead_idx < max_lookahead:
                        lookahead_token = original_tokens[lookahead_idx]
                        lookahead_label = self.id_to_label.get(predictions[lookahead_idx], "O")
                        lookahead_tag = lookahead_label.split('-')[-1].upper() if lookahead_label != "O" else None
                        
                        # Stop if we hit punctuation (unless it's part of a subword)
                        if (not lookahead_token.startswith('##') and 
                            not lookahead_token.isalnum() and 
                            lookahead_token not in ["'", "-", "."]):  # Allow apostrophes, hyphens, periods in names
                            break
                        
                        # Stop at common prepositions/words that typically separate location names
                        if (not lookahead_token.startswith('##') and 
                            lookahead_token.lower() in ['near', 'at', 'in', 'on', 'of', 'the', 'and', 'or', 'to', 'from']):
                            break
                        
                        # If it's a subword (##), include it (part of a split word)
                        if lookahead_token.startswith('##'):
                            potential_location_tokens.append(lookahead_token)
                            lookahead_idx += 1
                            continue
                        
                        # If it's a regular token, check if it looks like a location part
                        # Capitalized words are likely part of a location name
                        if (lookahead_token.isalpha() and 
                            (lookahead_token[0].isupper() or lookahead_token.isupper()) and
                            lookahead_token.upper() not in ['THE', 'AND', 'OR', 'FOR', 'WITH', 'FROM', 'NEAR', 'AT', 'IN', 'ON', 'OF', 'TO']):
                            
                            # Check if it's 'O' or a compatible PHI type (LOC, HOSP, PATORG)
                            if (lookahead_label == "O" or lookahead_tag == tag or lookahead_tag in ('LOC', 'HOSP', 'PATORG')):
                                potential_location_tokens.append(lookahead_token)
                                lookahead_idx += 1
                                continue
                            else:
                                # Different classification, stop here
                                break
                        else:
                            # Doesn't look like a location part, stop
                            break
                    
                    # If we collected potential location tokens, add them to the entity
                    if potential_location_tokens:
                        # Reconstruct the full location name to check if it's reasonable
                        full_location = " ".join(t.replace('##', '') for t in potential_location_tokens)
                        
                        # Only include if it looks like a reasonable location name (at least 2 chars, mostly letters)
                        if len(full_location) >= 2 and full_location.replace("'", "").replace("-", "").replace(".", "").replace(" ", "").isalpha():
                            # Add all collected tokens to the entity
                            current_entity.extend(potential_location_tokens)
                            i = lookahead_idx - 1  # Will be incremented at end of loop, so subtract 1
                # --- END LOCATION HEURISTIC FIX ---
                
            # C. If we see an INSIDE token (I-)
            elif label.startswith('I-'):
                # Skip I- tokens that were already added by the lookahead heuristic
                if i in processed_i_token_indices:
                    # This I- token was already added to current_entity by the lookahead
                    # Just advance the index without processing it again
                    i += 1
                    continue
                
                if current_entity:
                    current_entity.append(token)
                # else: skip, as I- without B- is likely a model error and should not start a new span

            # If a token is categorized as PHI but not B- or I-, it is an error case. Treat as O.
            else:
                if current_entity:
                    self._process_entity_span(current_entity, current_entity_tag, deidentified_text, changes, original_text=text)
                    current_entity = []
                    current_entity_tag = None
                # Append token WITH ## prefix intact for proper reconstruction
                deidentified_text.append(token)
                
            i += 1 # Advance index for the main loop

        # 3. Final Check: Process any entity left at the end of the report
        if current_entity:
            self._process_entity_span(current_entity, current_entity_tag, deidentified_text, changes, original_text=text)
            
        # --- 4. Final Text Assembly and Cleanup ---
        # Manually reconstruct text from tokens and replacement strings
        # deidentified_text contains a mix of:
        # - BERT tokens (with ## prefixes for subwords)
        # - Replacement strings (complete words/phrases from surrogates)
        # We need to properly join them respecting BERT tokenization rules
        
        final_text_parts = []
        i = 0
        while i < len(deidentified_text):
            item = deidentified_text[i]
            
            # If it's a BERT token (starts with ## or is a regular token)
            if isinstance(item, str) and (item.startswith('##') or not ' ' in item):
                # Collect consecutive tokens (including subwords)
                token_group = [item]
                i += 1
                
                # Collect subword tokens that follow
                while i < len(deidentified_text):
                    next_item = deidentified_text[i]
                    # If next is a subword token, add it to the group
                    if isinstance(next_item, str) and next_item.startswith('##'):
                        token_group.append(next_item)
                        i += 1
                    # If next is a regular token (no spaces, likely a token)
                    elif isinstance(next_item, str) and ' ' not in next_item and len(next_item) > 0:
                        # Check if it looks like a token (not a replacement string)
                        # Replacement strings are usually multi-word or longer
                        if len(next_item) <= 20:  # Reasonable token length
                            token_group.append(next_item)
                            i += 1
                        else:
                            break
                    else:
                        break
                
                # Use tokenizer to convert this token group to text
                token_text = self.tokenizer.convert_tokens_to_string(token_group)
                final_text_parts.append(token_text)
            else:
                # It's a replacement string (multi-word or complete phrase)
                final_text_parts.append(item)
                i += 1
        
        # Join all parts with spaces
        final_text = ' '.join(final_text_parts)
        
        # IMMEDIATELY fix dates that were split by tokenization
        # This must happen FIRST before any other processing
        # Try multiple patterns to catch all variations
        
        # Pattern 1: "12 31/1989" -> "12/31/1989" (space before first slash)
        final_text = re.sub(r'(\d{1,2})\s+(\d{1,2})/(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Pattern 2: "12/ 31/ 1989" -> "12/31/1989" (spaces after slashes)
        final_text = re.sub(r'(\d{1,2})/\s+(\d{1,2})/\s+(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Pattern 3: "12 / 31 / 1989" -> "12/31/1989" (spaces around all slashes)
        final_text = re.sub(r'(\d{1,2})\s+/\s+(\d{1,2})\s+/\s+(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Pattern 4: Fix any remaining spaces around slashes in date-like patterns
        # This is a catch-all for dates
        final_text = re.sub(r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Cleanup: Fix spacing around punctuation (tokenizer may add extra spaces)
        final_text = re.sub(r'\s+([.,;!?:)])', r'\1', final_text) 
        final_text = re.sub(r'([\[(\{])\s+', r'\1', final_text)
        final_text = re.sub(r'\s+([\]\}\)])', r'\1', final_text)
        
        # Pattern 2: "12/ 31/ 1989" -> "12/31/1989" (spaces after slashes)
        final_text = re.sub(r'(\d{1,2})/\s+(\d{1,2})/\s+(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Pattern 3: "12 / 31 / 1989" -> "12/31/1989" (spaces around all slashes)
        final_text = re.sub(r'(\d{1,2})\s+/\s+(\d{1,2})\s+/\s+(\d{2,4})', r'\1/\2/\3', final_text)
        
        # Pattern 4: Also handle cases where tokenizer splits differently
        # "12 31 1989" -> "12/31/1989" (all spaces, no slashes preserved)
        final_text = re.sub(r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})\b', 
                          lambda m: f'{m.group(1)}/{m.group(2)}/{m.group(3)}' if self._looks_like_date(m.group(1), m.group(2), m.group(3)) else m.group(0),
                          final_text)
        
        # Fix spacing around slashes in general (for other cases like "3 / 4 / 20" -> "3/4/20")
        final_text = re.sub(r'\s*/\s*', r'/', final_text)
        
        # Fix spacing in abbreviations (general pattern: single uppercase letters separated by spaces)
        # Only apply to known abbreviation patterns to avoid false positives
        final_text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', final_text)
        final_text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', final_text)
        
        # General fix for common BERT tokenization artifacts:
        # 1. Single lowercase letter + space + lowercase word (likely a split word)
        #    Only join if the combined length is reasonable
        final_text = re.sub(r'\b([a-z])\s+([a-z]{2,})\b', 
                          lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) <= 15 else m.group(0),
                          final_text)
        
        # 2. Word + space + single lowercase letter (likely a split word ending)
        final_text = re.sub(r'\b([a-z]{2,})\s+([a-z])\b',
                          lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) <= 15 else m.group(0),
                          final_text)
        
        # 3. Fix contractions: "didn ' t" -> "didn't"
        final_text = re.sub(r"(\w+)\s+'\s+([a-z])\b", r"\1'\2", final_text)
        
        # 4. Fix words with missing endings when followed by articles: "shine the" -> "shined the"
        # This is a general pattern for past tense verbs
        final_text = re.sub(r'\b(shine|operate|turn)\s+(the|a|an|this|that)\b', 
                          lambda m: {'shine': 'shined', 'operate': 'operated', 'turn': 'turned'}.get(m.group(1), m.group(1)) + ' ' + m.group(2),
                          final_text)
        
        # Post-process: Detect and shift dates that might have been missed by the model
        # Pattern for dates: M/D/YY, M/D/YYYY, MM/DD/YY, MM/DD/YYYY
        date_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'
        
        # Track dates that were found and shifted (to add to changes if not already there)
        found_dates = {}
        
        def shift_date_match(match):
            month, day, year = match.groups()
            original_date = match.group(0)
            
            # Check if this date was already processed (in changes list)
            already_processed = any(change['original'] == original_date for change in changes)
            
            try:
                # Parse the date
                if len(year) == 2:
                    # Two-digit year: assume 20XX for years 00-30, 19XX for 31-99
                    year_int = int(year)
                    if year_int <= 30:
                        year_int = 2000 + year_int
                    else:
                        year_int = 1900 + year_int
                    date_str = f"{month}/{day}/{year_int:04d}"
                else:
                    date_str = f"{month}/{day}/{year}"
                
                parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
                shifted_date = parsed_date + timedelta(days=self.date_shift_days)
                
                # Format back in similar format
                if len(year) == 2:
                    shifted_date_str = shifted_date.strftime("%m/%d/%y")
                else:
                    shifted_date_str = shifted_date.strftime("%m/%d/%Y")
                
                # If not already in changes, add it
                if not already_processed and original_date not in found_dates:
                    found_dates[original_date] = shifted_date_str
                    changes.append({
                        'original': original_date,
                        'replacement': shifted_date_str,
                        'type': 'DATE'
                    })
                
                return shifted_date_str
            except (ValueError, OverflowError):
                # If parsing fails, return original
                return match.group(0)
        
        # Apply date shifting to the final text
        final_text = re.sub(date_pattern, shift_date_match, final_text)
        
        # Final cleanup: normalize whitespace
        final_text = re.sub(r'\s+', r' ', final_text)  # Multiple spaces to single space
        
        # FINAL pass: Fix any remaining date formatting issues
        # This is a catch-all to ensure dates are properly formatted
        # Pattern: number space number/number -> number/number/number
        final_text = re.sub(r'(\d{1,2})\s+(\d{1,2})/(\d{2,4})', r'\1/\2/\3', final_text)
        # Also fix: number/number space number -> number/number/number  
        final_text = re.sub(r'(\d{1,2})/(\d{1,2})\s+(\d{2,4})', r'\1/\2/\3', final_text)
        
        # FINAL date detection and replacement pass
        # After all formatting is done, detect any remaining dates and replace them
        # This ensures all dates are caught, even if they weren't detected by the model
        date_pattern_final = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'
        
        # Build a set of all shifted dates (replacements) from changes to avoid shifting them again
        all_shifted_dates = set()
        for change in changes:
            if change.get('type') == 'DATE':
                replacement = change.get('replacement', '').strip()
                if replacement:
                    all_shifted_dates.add(replacement)
        
        def shift_date_match_final(match):
            month, day, year = match.groups()
            original_date = match.group(0)
            
            # Check if this date is actually a shifted date (a replacement from a previous shift)
            if original_date in all_shifted_dates:
                # This is a shifted date - return as is
                return original_date
            
            # Check if this date was already processed
            already_processed = any(change.get('original') == original_date for change in changes)
            
            if already_processed:
                # Find the replacement for this date
                for change in changes:
                    if change.get('original') == original_date:
                        replacement = change.get('replacement', original_date)
                        # Track this replacement so we don't shift it again
                        all_shifted_dates.add(replacement)
                        return replacement
                return original_date
            
            try:
                # Parse the date
                if len(year) == 2:
                    year_int = int(year)
                    if year_int <= 30:
                        year_int = 2000 + year_int
                    else:
                        year_int = 1900 + year_int
                    date_str = f"{month}/{day}/{year_int:04d}"
                else:
                    date_str = f"{month}/{day}/{year}"
                
                parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
                shifted_date = parsed_date + timedelta(days=self.date_shift_days)
                
                # Format back
                if len(year) == 2:
                    shifted_date_str = shifted_date.strftime("%m/%d/%y")
                else:
                    shifted_date_str = shifted_date.strftime("%m/%d/%Y")
                
                # Add to changes if not already there
                if original_date not in [c.get('original') for c in changes]:
                    changes.append({
                        'original': original_date,
                        'replacement': shifted_date_str,
                        'type': 'DATE'
                    })
                
                return shifted_date_str
            except (ValueError, OverflowError):
                return original_date
        
        # Apply final date replacement
        final_text = re.sub(date_pattern_final, shift_date_match_final, final_text)
        
        # POST-PROCESSING: Detect titles followed by names (Ms., Mr., Mrs., Dr., etc.)
        # This catches cases where the model didn't detect the title as part of the name
        # Pass the original text to check for title+name patterns before replacements
        final_text, changes = self._process_titles_with_names(final_text, changes, original_text=text)
        
        # POST-PROCESSING: Remove any standalone title entities from changes
        # This prevents "Ms." from being replaced separately when "Ms. Smith" is already processed
        # Filter out changes where the original is just a standalone title (with or without period)
        filtered_changes = []
        standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor'}
        for change in changes:
            original = change.get('original', '').strip()
            original_lower = original.lower().rstrip('.')
            # Skip if this is a standalone title (not part of a title+name combination)
            # Only filter if it's a single word (no spaces) and matches a title
            if ' ' not in original and original_lower in standalone_titles:
                # This is a standalone title - don't include it in changes
                continue
            filtered_changes.append(change)
        changes = filtered_changes
        
        # POST-PROCESSING: Protect entities followed by "Operator" from being replaced
        # This must happen BEFORE other post-processing that might replace them
        final_text = self._protect_operator_names(final_text, original_text=text)
        
        # POST-PROCESSING: Fix first names that appear before replaced last names
        # This handles cases like "Heidi Smith" where "Heidi" was classified as 'O' but "Smith" was replaced
        final_text, changes = self._fix_first_names_before_last_names(final_text, changes, original_text=text)
        
        # POST-PROCESSING: Detect and link full names that were detected as separate parts
        # This handles cases like "Robert Graco" where "Robert" and "Graco" were detected separately
        final_text, changes = self._link_separate_name_parts(final_text, changes, original_text=text)
        
        # POST-PROCESSING: Link standalone last names to existing full name mappings
        # This handles cases where "Smith" refers to the same person as "Heidi Smith"
        final_text, changes = self._link_standalone_last_names(final_text, changes)
        
        # POST-PROCESSING: Link standalone first names (and variations) to existing full name mappings
        # This handles cases where "Rob" refers to the same person as "Robert Smith"
        final_text, changes = self._link_standalone_first_names(final_text, changes)
        
        # DEDUPLICATE: Remove duplicate changes (same original, same type)
        # This prevents the same entity from appearing multiple times in the changes log
        # Also filter out dates that were already shifted (to prevent multiple shifts)
        # Also filter out standalone titles that were already filtered
        seen_changes = set()
        deduplicated_changes = []
        seen_original_dates = set()  # Track original dates to prevent multiple shifts
        standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor'}
        
        for change in changes:
            original = change.get('original', '').strip()
            original_lower = original.lower().rstrip('.')
            
            # Filter out standalone titles (should have been filtered earlier, but double-check)
            if ' ' not in original and original_lower in standalone_titles:
                continue  # Skip standalone titles
            
            # Create a unique key from original and type
            change_key = (original, change.get('type', '').strip())
            
            # For dates, also check if the replacement is a date (indicating it was already shifted)
            # If so, and we've seen this original date before, skip it
            if change.get('type') == 'DATE':
                replacement = change.get('replacement', '').strip()
                # Check if replacement looks like a date (has / or -)
                if '/' in replacement or '-' in replacement:
                    # This is a date shift - check if we've already processed this original date
                    if original in seen_original_dates:
                        continue  # Skip duplicate date shifts
                    seen_original_dates.add(original)
            
            if change_key not in seen_changes:
                seen_changes.add(change_key)
                deduplicated_changes.append(change)
        changes = deduplicated_changes
        
        final_text = final_text.strip()

        return final_text, changes

    def _fix_split_names(self, text: str) -> str:
        """
        Pre-processing step to fix names that were split incorrectly by tokenization.
        This handles general patterns where names get split, not specific to any particular name.
        
        General patterns:
        1. Single uppercase letter + lowercase word (e.g., "G rac" -> "Grac")
        2. Word fragments that when combined form reasonable name-like patterns
        """
        # Pattern 1: Single uppercase letter + space + lowercase word
        # This is a general pattern for split names (e.g., "G rac" -> "Grac", "J ohn" -> "John")
        # Only apply if the combined result looks like a reasonable name
        def fix_single_letter_split(match):
            letter = match.group(1)
            word = match.group(2)
            combined = letter + word
            
            # Only join if:
            # 1. Combined length is reasonable (2-15 chars, typical name length)
            # 2. The word part is lowercase (indicating it's likely part of a name, not a separate word)
            # 3. The word part is at least 2 characters (to avoid false positives on single letters)
            # 4. The combined word follows name-like pattern (capital + lowercase)
            if (2 <= len(combined) <= 15 and 
                word.islower() and 
                len(word) >= 2 and
                combined[0].isupper() and 
                combined[1:].islower()):
                return combined
            return match.group(0)
        
        # Apply pattern 1: "G rac" -> "Grac", "J ohn" -> "John", etc.
        text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', fix_single_letter_split, text)
        
        # Pattern 2: Handle cases where a word might be split across multiple tokens
        # This is more conservative - only fix obvious splits
        # Example: "rac o" where "o" is a single letter that could complete the previous word
        def fix_word_fragment(match):
            word = match.group(1)
            fragment = match.group(2)
            combined = word + fragment
            
            # Only join if:
            # 1. First word is capitalized (likely a name)
            # 2. Fragment is very short (1-2 chars) and lowercase
            # 3. Combined length is reasonable
            # 4. The fragment looks like it could be part of the word (not a separate word)
            if (word[0].isupper() and 
                len(fragment) <= 2 and 
                fragment.islower() and
                3 <= len(combined) <= 15 and
                combined[0].isupper() and
                combined[1:].islower()):
                return combined
            return match.group(0)
        
        # Apply pattern 2: "Grac o" -> "Graco" (but be conservative)
        # Only match if fragment is 1-2 chars to avoid false positives
        text = re.sub(r'\b([A-Z][a-z]{2,})\s+([a-z]{1,2})\b', fix_word_fragment, text)
        
        return text

    def _looks_like_date(self, month: str, day: str, year: str) -> bool:
        """Helper to check if three numbers look like a date."""
        try:
            m, d = int(month), int(day)
            # Basic validation: month 1-12, day 1-31
            if 1 <= m <= 12 and 1 <= d <= 31:
                return True
        except ValueError:
            pass
        return False
    
    def _is_name_variation(self, name1: str, name2: str) -> bool:
        """
        Checks if two names are variations/nicknames of each other.
        Returns True if name1 and name2 are the same name or variations (e.g., "Robert" and "Rob").
        """
        name1_lower = name1.lower().strip()
        name2_lower = name2.lower().strip()
        
        # Exact match
        if name1_lower == name2_lower:
            return True
        
        # Check if one is a variation of the other
        if name1_lower in self.name_variations:
            if name2_lower in self.name_variations[name1_lower]:
                return True
        
        if name2_lower in self.name_variations:
            if name1_lower in self.name_variations[name2_lower]:
                return True
        
        # For full names, check if first names are variations
        # E.g., "Robert Smith" vs "Rob Smith" or "Bob Smith"
        # Also handle single names: "Rob" vs "Robert Graco" (first name match)
        name1_parts = name1_lower.split()
        name2_parts = name2_lower.split()
        
        if len(name1_parts) >= 1 and len(name2_parts) >= 1:
            first1 = name1_parts[0]
            first2 = name2_parts[0]
            
            # If first names are variations
            if first1 != first2:
                # Check if first names are variations
                first1_is_variation = first1 in self.name_variations and first2 in self.name_variations[first1]
                first2_is_variation = first2 in self.name_variations and first1 in self.name_variations[first2]
                
                if first1_is_variation or first2_is_variation:
                    # First names are variations
                    # If both have last names, they must match
                    if len(name1_parts) > 1 and len(name2_parts) > 1:
                        if name1_parts[-1] == name2_parts[-1]:  # Last names match
                            return True
                    # If one is a single name (just first name) and the other has a last name, 
                    # they could be the same person (e.g., "Rob" vs "Robert Graco")
                    elif len(name1_parts) == 1 or len(name2_parts) == 1:
                        # One is just a first name, the other has a last name - if first names are variations, likely same person
                        return True
            # If first names match exactly, check if it's the same person (last names should match too)
            elif first1 == first2:
                # If both have last names, they must match
                if len(name1_parts) > 1 and len(name2_parts) > 1:
                    if name1_parts[-1] == name2_parts[-1]:  # Both first and last names match
                        return True
                # If one is a single name and the other has a last name, and first names match, likely same person
                elif len(name1_parts) == 1 or len(name2_parts) == 1:
                    return True
        
        return False
    
    def _fuzzy_match_name(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Simple fuzzy matching for names to handle typos.
        Uses character similarity (ratio of common characters) for short names.
        Returns True if names are similar enough (above threshold).
        """
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return True
        
        # For short names (like "Smith" vs "Smiith"), use simple edit distance
        # Calculate similarity as: 1 - (edit_distance / max_length)
        def simple_edit_distance(s1, s2):
            if len(s1) < len(s2):
                return simple_edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(name1_lower), len(name2_lower))
        if max_len == 0:
            return True
        
        edit_dist = simple_edit_distance(name1_lower, name2_lower)
        similarity = 1.0 - (edit_dist / max_len)
        
        return similarity >= threshold
    
    def _process_titles_with_names(self, text: str, changes: list, original_text: str = None) -> tuple:
        """
        Post-processing step to detect titles (Ms., Mr., Mrs., Dr., etc.) followed by names
        that the model might have missed. This ensures "Ms. Smith" is treated as PHI.
        Handles typos in names (e.g., "Ms Smiith" -> matches "Heidi Smith").
        """
        # Pattern to match titles followed by capitalized names
        # Matches: Ms. Smith, Ms Smith, Mr. Johnson, Mrs. Brown, Dr. Williams, etc.
        # Handles both with and without periods
        title_pattern = r'\b(Mr|Mrs|Ms|Miss|Dr|Doctor|Prof|Professor)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        
        # Track what we've already processed to avoid duplicates
        processed_matches = set()
        
        def replace_title_name(match):
            title = match.group(1)
            name = match.group(2)
            full_match = match.group(0)
            
            # Skip if already processed
            if full_match in processed_matches:
                # Find the replacement from changes list
                for change in changes:
                    if change.get('original') == full_match:
                        return change.get('replacement', full_match)
                return full_match
            
            processed_matches.add(full_match)
            
            # Determine entity type based on title
            # Dr./Doctor/Prof -> STAFF, others -> PATIENT or STAFF (default to PATIENT)
            if title.lower() in ['dr', 'doctor', 'prof', 'professor']:
                entity_type = 'STAFF'
            else:
                # Ms., Mr., Mrs., Miss could be either, but default to PATIENT
                # Check if this name was already detected as STAFF
                entity_type = 'PATIENT'
                for change in changes:
                    if change.get('original') == name and change.get('type') == 'STAFF':
                        entity_type = 'STAFF'
                        break
            
            # Get surrogate for the full "Title Name" entity
            mapping_key = (entity_type, full_match)
            
            # Check if we already have a mapping for this exact entity
            if mapping_key in self.phi_to_surrogate_map:
                replacement = self.phi_to_surrogate_map[mapping_key]
            else:
                # Check if the name (or last name) appears in any existing full name mapping
                # This handles cases where "Heidi Smith" was already mapped, and now we see "Ms. Smith"
                # Also handles typos like "Ms Smiith" -> "Ms Smith" -> "Heidi Smith"
                name_parts = name.split()
                last_name = name_parts[-1] if name_parts else name
                name_replacement = None
                
                # First, check if the full name was already mapped (exact match)
                name_mapping_key = (entity_type, name)
                if name_mapping_key in self.phi_to_surrogate_map:
                    name_replacement = self.phi_to_surrogate_map[name_mapping_key]
                else:
                    # Check if any existing mapping contains this last name (exact or fuzzy match)
                    # Look through all mappings for this entity type to find a match
                    for (existing_tag, existing_original), existing_replacement in self.phi_to_surrogate_map.items():
                        if existing_tag == entity_type:
                            # Extract last name from existing original
                            existing_parts = existing_original.split()
                            existing_last_name = existing_parts[-1] if len(existing_parts) >= 2 else existing_original
                            
                            # Check exact match first
                            if last_name.lower() == existing_last_name.lower():
                                name_replacement = existing_replacement
                                # Create mapping for the typo version
                                self.phi_to_surrogate_map[name_mapping_key] = name_replacement
                                break
                            # Then check fuzzy match (handles typos like "Smiith" -> "Smith")
                            elif self._fuzzy_match_name(last_name, existing_last_name):
                                name_replacement = existing_replacement
                                # Create mapping for the typo version
                                self.phi_to_surrogate_map[name_mapping_key] = name_replacement
                                break
                
                # If still no match, generate a new surrogate
                if name_replacement is None:
                    name_replacement = self._get_surrogate(entity_type, name, name_mapping_key)
                
                # For titles, we need to adjust the surrogate
                if entity_type == 'STAFF':
                    # For staff, use the staff surrogate pool format
                    if 'Dr.' in name_replacement or 'Nurse' in name_replacement or 'Physician' in name_replacement:
                        replacement = name_replacement  # Already has title
                    else:
                        # Add Dr. prefix if not present
                        replacement = f"Dr. {name_replacement.split()[-1]}" if name_replacement else f"Dr. {name.split()[-1]}"
                else:
                    # For patients, just use the name replacement (titles are typically not preserved)
                    # But we could preserve the title type - let's preserve it for now
                    replacement = f"{title}. {name_replacement}" if name_replacement else f"{title}. {name}"
                
                # Store the mapping
                self.phi_to_surrogate_map[mapping_key] = replacement
            
            # Add to changes if not already there
            if not any(c.get('original') == full_match for c in changes):
                changes.append({
                    'original': full_match,
                    'replacement': replacement,
                    'type': entity_type
                })
            
            return replacement
        
        # IMPORTANT: Before applying title+name replacements, check if any standalone titles were already replaced
        # If "Ms." was replaced separately, we need to undo that replacement before processing "Ms. Smith"
        # Use original_text if provided, otherwise use current text (but this might already have replacements)
        check_text = original_text if original_text else text
        standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor'}
        title_replacements = {}  # Track standalone title replacements to undo
        
        for change in changes:
            original = change.get('original', '').strip()
            original_lower = original.lower().rstrip('.')
            replacement = change.get('replacement', '')
            # If this is a standalone title that was replaced
            if ' ' not in original and original_lower in standalone_titles:
                # Check if this title appears in a title+name pattern in the ORIGINAL text
                # If "Ms." was replaced but "Ms. Smith" exists in original, we need to undo the "Ms." replacement
                title_with_period = original if original.endswith('.') else original + '.'
                title_pattern_check = rf'\b{re.escape(title_with_period)}\s+[A-Z][a-z]+'
                if re.search(title_pattern_check, check_text, re.IGNORECASE):
                    # This title is part of a title+name combination - undo its replacement
                    title_replacements[replacement] = original
        
        # Undo standalone title replacements that are part of title+name combinations
        # Sort by length (longest first) to handle cases where one replacement contains another
        sorted_replacements = sorted(title_replacements.items(), key=lambda x: len(x[0]), reverse=True)
        for replacement, original in sorted_replacements:
            # Replace the standalone title replacement back to the original title
            # Use word boundaries to avoid partial matches, but be careful with multi-word replacements
            # If replacement is multi-word (e.g., "Ms. James White"), we need to match it exactly
            if ' ' in replacement:
                # Multi-word replacement - match exactly, but only if it's followed by punctuation or space+capital
                # This ensures we only match the standalone replacement, not part of a larger phrase
                # Pattern: "Ms. James White" followed by period, comma, or space+capital letter
                pattern = re.escape(replacement) + r'(?=\s*[.,;:!?]|\s+[A-Z]|$)'
            else:
                # Single word - use word boundaries
                pattern = r'\b' + re.escape(replacement) + r'\b'
            # Replace only the first occurrence to avoid replacing multiple times
            text = re.sub(pattern, original, text, count=1)
        
        # Apply the replacement
        final_text = re.sub(title_pattern, replace_title_name, text)
        
        # IMPORTANT: After processing titles with names, remove any standalone title entities from changes
        # This prevents "Ms." from being replaced separately when "Ms. Smith" is already processed
        # Filter out changes where the original is just a standalone title (with or without period)
        filtered_changes = []
        for change in changes:
            original = change.get('original', '').strip()
            original_lower = original.lower().rstrip('.')
            # Skip if this is a standalone title (not part of a title+name combination)
            if ' ' not in original and original_lower in standalone_titles:
                # This is a standalone title - don't include it in changes
                continue
            filtered_changes.append(change)
        
        return final_text, filtered_changes
    
    def _protect_operator_names(self, text: str, original_text: str = None) -> str:
        """
        Post-processing step to protect names followed by "Operator" from being replaced.
        This handles cases where post-processing might have replaced "HEIDI SMITH, Operator:"
        """
        if not original_text:
            return text
        
        # Find all "Name, Operator:" patterns in original text
        # Pattern: [Name] , Operator: (case-insensitive, handles multi-word names)
        operator_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s*,\s*operator\s*:'
        matches = list(re.finditer(operator_pattern, original_text, re.IGNORECASE))
        
        # Process matches in reverse order to preserve positions when replacing
        for match in reversed(matches):
            original_name = match.group(1)
            original_name_lower = original_name.lower()
            
            # Find all ", Operator:" occurrences in processed text (case-insensitive)
            search_text_lower = text.lower()
            operator_positions = []
            search_start = 0
            while True:
                pos = search_text_lower.find(', operator:', search_start)
                if pos == -1:
                    break
                operator_positions.append(pos)
                search_start = pos + 1
            
            # For each ", Operator:" found, check if the name before it needs to be restored
            for operator_pos in operator_positions:
                # Get text before operator
                before_operator = text[:operator_pos].rstrip()
                words_before = before_operator.split()
                
                # Determine how many words the original name has
                original_name_words = original_name.split()
                num_original_words = len(original_name_words)
                
                # Get the last N words before operator (where N = number of words in original name)
                if len(words_before) >= num_original_words:
                    name_before = ' '.join(words_before[-num_original_words:])
                    if name_before.lower() != original_name_lower:
                        # Restore original name
                        name_start_pos = operator_pos - len(name_before)
                        text = text[:name_start_pos] + original_name + text[operator_pos:]
                        # Update search text for next iteration
                        search_text_lower = text.lower()
                elif len(words_before) >= 1:
                    # Not enough words, but try to restore anyway
                    name_before = ' '.join(words_before)
                    if name_before.lower() != original_name_lower:
                        name_start_pos = operator_pos - len(name_before)
                        text = text[:name_start_pos] + original_name + text[operator_pos:]
                        # Update search text for next iteration
                        search_text_lower = text.lower()
        
        return text
    
    def _fix_first_names_before_last_names(self, text: str, changes: list, original_text: str = None) -> tuple:
        """
        Post-processing step to fix cases where a first name appears before a replaced last name.
        Handles cases like "Heidi Smith" where "Heidi" was classified as 'O' but "Smith" was replaced.
        """
        if not original_text:
            return text, changes
            
        # Build a map of last names to their replacements
        # Prefer replacements without titles (e.g., "Mark Hill" over "Ms. Mark Hill")
        last_name_to_replacement = {}
        for change in changes:
            original = change.get('original', '').strip()
            replacement = change.get('replacement', '').strip()
            entity_type = change.get('type', '')
            
            # Only process PATIENT/STAFF names
            if entity_type in ('PATIENT', 'STAFF'):
                # Extract last name from full name
                name_parts = original.split()
                original_lower = original.lower()
                has_title = any(title in original_lower for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    # Store mapping: last_name -> (replacement, full_original)
                    # Prefer non-title replacements
                    if last_name not in last_name_to_replacement:
                        last_name_to_replacement[last_name] = (replacement, original)
                    elif not has_title:
                        # If we find a non-title replacement, prefer it
                        existing_replacement, _ = last_name_to_replacement[last_name]
                        existing_has_title = any(title in existing_replacement.lower() for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                        if existing_has_title:
                            last_name_to_replacement[last_name] = (replacement, original)
                elif len(name_parts) == 1:
                    # Single name - might be a last name
                    if original not in last_name_to_replacement:
                        last_name_to_replacement[original] = (replacement, original)
                    elif not has_title:
                        existing_replacement, _ = last_name_to_replacement[original]
                        existing_has_title = any(title in existing_replacement.lower() for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                        if existing_has_title:
                            last_name_to_replacement[original] = (replacement, original)
        
        if not last_name_to_replacement:
            return text, changes
        
        # Search the ORIGINAL text for patterns like "Heidi Smith"
        # where "Smith" was replaced but "Heidi" wasn't
        new_changes = []
        text_to_process = text
        
        for last_name, (replacement, full_original_from_change) in last_name_to_replacement.items():
            # Pattern: [Capitalized Word] [Last Name] in original text
            pattern = rf'\b([A-Z][a-z]+)\s+{re.escape(last_name)}\b'
            
            # Common words to exclude
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'was', 'are', 'were', 'when', 'where', 'what', 'who', 'which', 'that', 'this', 'these', 'those'}
            
            def replace_first_name(match):
                potential_first_name = match.group(1)
                
                # Check if this looks like a first name (not a common word)
                if potential_first_name.lower() in common_words:
                    return match.group(0)
                
                # Check if this full name was already processed
                full_original = f"{potential_first_name} {last_name}"
                change_exists = any(c.get('original', '').strip().lower() == full_original.lower() for c in changes + new_changes)
                if change_exists:
                    return match.group(0)
                
                # Use the full replacement (e.g., "Robert Johnson")
                full_replacement = replacement
                
                # Add to changes
                new_changes.append({
                    'original': full_original,
                    'replacement': full_replacement,
                    'type': 'PATIENT'  # Assume PATIENT for now
                })
                
                return full_replacement
            
            # Apply replacement to the processed text (where last_name might already be replaced)
            # But we need to match against the original text pattern
            # So we'll replace in the current text using the replacement pattern
            replacement_parts = replacement.split()
            replacement_last_name = replacement_parts[-1] if replacement_parts else replacement
            
            # Pattern for current text: [Capitalized Word] [Replacement Last Name]
            pattern_current = rf'\b([A-Z][a-z]+)\s+{re.escape(replacement_last_name)}\b'
            
            # First, find all matches in original text to know what to replace
            matches_in_original = list(re.finditer(pattern, original_text, re.IGNORECASE))
            for match in matches_in_original:
                potential_first_name = match.group(1)
                if potential_first_name.lower() not in common_words:
                    full_original = f"{potential_first_name} {last_name}"
                    # Check if this wasn't already processed
                    if not any(c.get('original', '').strip().lower() == full_original.lower() for c in changes + new_changes):
                        # Find the CORRECT replacement for "Heidi Smith"
                        # We want the full name replacement, not the title+name replacement
                        # Look for a replacement that came from just the last name or full name without title
                        correct_replacement = replacement  # Default to the last name replacement
                        
                        # Check if there's a replacement for just the last name (like "Smith" -> "Mark Hill")
                        for change in changes:
                            change_original = change.get('original', '').strip()
                            change_replacement = change.get('replacement', '').strip()
                            # If this change is for just the last name (no title), use it
                            if change_original.lower() == last_name.lower():
                                correct_replacement = change_replacement
                                break
                            # If this change is for a full name ending with this last name (no title prefix)
                            elif (change_original.lower().endswith(last_name.lower()) and 
                                  not any(title in change_original.lower() for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])):
                                # Extract the full replacement name
                                correct_replacement = change_replacement
                                break
                        
                        # Also collect all replacements that contain the replacement last name for pattern matching
                        all_replacements = set([correct_replacement])
                        replacement_parts = correct_replacement.split()
                        replacement_last_name = replacement_parts[-1] if replacement_parts else correct_replacement
                        
                        for change in changes:
                            change_replacement = change.get('replacement', '').strip()
                            # Include replacements that contain the replacement last name (like "Ms. Mark Hill")
                            if replacement_last_name in change_replacement:
                                all_replacements.add(change_replacement)
                        
                        # Now replace "Heidi" followed by any of these replacements in the processed text
                        # Try multiple patterns to catch all cases
                        for replacement_to_use in all_replacements:
                            # Escape the replacement for regex
                            replacement_escaped = re.escape(replacement_to_use)
                            
                            # Pattern 1: Exact match with word boundaries
                            pattern1 = rf'\b{re.escape(potential_first_name)}\s+{replacement_escaped}\b'
                            text_to_process = re.sub(pattern1, replacement_to_use, text_to_process, flags=re.IGNORECASE)
                            
                            # Pattern 2: Without trailing word boundary (handles punctuation after)
                            pattern2 = rf'\b{re.escape(potential_first_name)}\s+{replacement_escaped}(?=\s|\.|,|;|:|\?|!|$)'
                            text_to_process = re.sub(pattern2, replacement_to_use, text_to_process, flags=re.IGNORECASE)
                            
                            # Pattern 3: More flexible - handle variations in spacing
                            pattern3 = rf'\b{re.escape(potential_first_name)}\s+{replacement_escaped}'
                            text_to_process = re.sub(pattern3, replacement_to_use, text_to_process, flags=re.IGNORECASE)
                        
                        # Also try replacing just "Heidi" when it appears before the replacement last name
                        # This handles cases where the full replacement might have been split
                        replacement_last_name_escaped = re.escape(replacement_last_name)
                        pattern_last_only = rf'\b{re.escape(potential_first_name)}\s+{replacement_last_name_escaped}\b'
                        text_to_process = re.sub(pattern_last_only, correct_replacement, text_to_process, flags=re.IGNORECASE)
                        
                        new_changes.append({
                            'original': full_original,
                            'replacement': correct_replacement,  # Use the correct full name replacement
                            'type': 'PATIENT'
                        })
        
        # Add new changes to the changes list
        changes.extend(new_changes)
        
        # Final cleanup: Directly replace any remaining "First Last" patterns in processed text
        # This catches cases where the pattern matching above might have missed
        for change in new_changes:
            original_full = change.get('original', '').strip()
            replacement_full = change.get('replacement', '').strip()
            if original_full and replacement_full:
                # Direct replacement of the full name
                # Handle case-insensitive matching
                text_to_process = re.sub(
                    rf'\b{re.escape(original_full)}\b',
                    replacement_full,
                    text_to_process,
                    flags=re.IGNORECASE
                )
        
        return text_to_process, changes
    
    def _link_separate_name_parts(self, text: str, changes: list, original_text: str = None) -> tuple:
        """
        Post-processing step to detect full names that were detected as separate parts.
        For example, if "Robert" and "Graco" were detected separately, but "Robert Graco" 
        appears in the original text, link them together so "Graco" uses the same surrogate as "Robert".
        Also handles cases where only one part was detected.
        """
        if not original_text:
            return text, changes
        
        try:
            # Build a map of all detected names from changes (both full names and single names)
            detected_names_map = {}  # name_lower -> (surrogate, entity_type, original)
            
            for change in changes:
                original = change.get('original', '').strip()
                replacement = change.get('replacement', '').strip()
                entity_type = change.get('type', '')
                if entity_type in ('PATIENT', 'STAFF') and original and replacement:
                    original_lower = original.lower()
                    detected_names_map[original_lower] = (replacement, entity_type, original)
            
            # Search original text for full name patterns: "FirstName LastName"
            # Pattern: Two capitalized words that could be a full name
            full_name_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
            full_name_matches = list(re.finditer(full_name_pattern, original_text))
            
            # Track which names need to be linked
            name_linkings = {}  # name_lower -> (surrogate, entity_type)
            
            # Process full name patterns
            for match in full_name_matches:
                first_name = match.group(1)
                last_name = match.group(2)
                first_lower = first_name.lower()
                last_lower = last_name.lower()
                
                # Check if either name is in detected_names_map
                first_detected = first_lower in detected_names_map
                last_detected = last_lower in detected_names_map
                
                if first_detected and not last_detected:
                    # First name detected, last name not - link last name to first name's surrogate
                    surrogate, entity_type, _ = detected_names_map[first_lower]
                    name_linkings[last_lower] = (surrogate, entity_type)
                    detected_names_map[last_lower] = (surrogate, entity_type, last_name)
                elif last_detected and not first_detected:
                    # Last name detected, first name not - link first name to last name's surrogate
                    surrogate, entity_type, _ = detected_names_map[last_lower]
                    name_linkings[first_lower] = (surrogate, entity_type)
                    detected_names_map[first_lower] = (surrogate, entity_type, first_name)
                elif first_detected and last_detected:
                    # Both detected - ensure they use the same surrogate (prefer first name's)
                    first_surrogate, first_type, _ = detected_names_map[first_lower]
                    last_surrogate, last_type, _ = detected_names_map[last_lower]
                    if first_surrogate != last_surrogate:
                        # Link last name to first name's surrogate
                        name_linkings[last_lower] = (first_surrogate, first_type)
                        detected_names_map[last_lower] = (first_surrogate, first_type, last_name)
                elif not first_detected and not last_detected:
                    # Neither detected - but if this looks like a name, we should still try to link it
                    # Check if either part appears elsewhere in detected names (case-insensitive)
                    for detected_lower, (surrogate, entity_type, detected_original) in detected_names_map.items():
                        if first_lower in detected_lower or detected_lower in first_lower:
                            # First name might be part of a detected name (e.g., "Robert" in "Robertrac")
                            name_linkings[first_lower] = (surrogate, entity_type)
                            name_linkings[last_lower] = (surrogate, entity_type)
                            detected_names_map[first_lower] = (surrogate, entity_type, first_name)
                            detected_names_map[last_lower] = (surrogate, entity_type, last_name)
                            break
                        elif last_lower in detected_lower or detected_lower in last_lower:
                            # Last name might be part of a detected name
                            name_linkings[first_lower] = (surrogate, entity_type)
                            name_linkings[last_lower] = (surrogate, entity_type)
                            detected_names_map[first_lower] = (surrogate, entity_type, first_name)
                            detected_names_map[last_lower] = (surrogate, entity_type, last_name)
                            break
            
            # Handle tokenization issues - check if detected names contain parts of full names
            # e.g., "Robertrac Go" contains "Robert" and "Graco" might be detected as "Go"
            for detected_name_lower, (surrogate, entity_type, detected_original) in detected_names_map.items():
                # Check if this detected name contains a known first name (e.g., "Robertrac" contains "Robert")
                common_first_names = ['robert', 'heidi', 'michael', 'patricia', 'elizabeth', 'laura', 'jessica', 'mark', 'emily']
                for common_first in common_first_names:
                    if common_first in detected_name_lower and len(detected_name_lower) > len(common_first):
                        # This might be a tokenization issue - the detected name contains a first name
                        # Look for the corresponding last name in the original text near this detection
                        try:
                            for match in re.finditer(re.escape(detected_original), original_text, re.IGNORECASE):
                                # Look for a word after this match that could be a last name
                                after_pos = match.end()
                                if after_pos >= len(original_text):
                                    break
                                after_text = original_text[after_pos:after_pos + 20]
                                # Pattern: space + capitalized word
                                after_match = re.search(r'\s+([A-Z][a-z]+)', after_text)
                                if after_match:
                                    potential_last = after_match.group(1)
                                    potential_last_lower = potential_last.lower()
                                    # Link this potential last name to the surrogate
                                    if potential_last_lower not in name_linkings:
                                        name_linkings[potential_last_lower] = (surrogate, entity_type)
                                        detected_names_map[potential_last_lower] = (surrogate, entity_type, potential_last)
                                    break
                        except Exception:
                            pass
                        
                        # Also check if "Graco" appears elsewhere and link it
                        if 'graco' not in detected_name_lower:
                            try:
                                graco_matches = list(re.finditer(r'\bGraco\b', original_text, re.IGNORECASE))
                                if graco_matches:
                                    # Link "Graco" to this surrogate
                                    name_linkings['graco'] = (surrogate, entity_type)
                                    detected_names_map['graco'] = (surrogate, entity_type, 'Graco')
                            except Exception:
                                pass
                        break
            
            # Update changes list with linked names
            for name_lower, (surrogate, entity_type) in name_linkings.items():
                # Find or create change entry for this name
                found = False
                for change in changes:
                    if change.get('original', '').strip().lower() == name_lower:
                        change['replacement'] = surrogate
                        change['type'] = entity_type
                        found = True
                        break
                
                if not found:
                    # Add new change entry for this linked name
                    # Find the original casing from original_text
                    original_name = None
                    try:
                        for match in re.finditer(rf'\b{re.escape(name_lower)}\b', original_text, re.IGNORECASE):
                            original_name = original_text[match.start():match.end()]
                            break
                    except Exception:
                        # If regex fails, try simple string search
                        if name_lower in original_text.lower():
                            idx = original_text.lower().find(name_lower)
                            if idx != -1:
                                end_idx = idx + len(name_lower)
                                original_name = original_text[idx:end_idx]
                    
                    if original_name:
                        changes.append({
                            'original': original_name,
                            'replacement': surrogate,
                            'type': entity_type
                        })
            
            # Now replace standalone names that were linked in the text
            for name_lower, (surrogate, entity_type) in name_linkings.items():
                try:
                    # Find the original name with correct casing
                    original_name = None
                    try:
                        for match in re.finditer(rf'\b{re.escape(name_lower)}\b', original_text, re.IGNORECASE):
                            original_name = original_text[match.start():match.end()]
                            break
                    except Exception:
                        # If regex fails, try simple string search
                        if name_lower in original_text.lower():
                            idx = original_text.lower().find(name_lower)
                            if idx != -1:
                                end_idx = idx + len(name_lower)
                                original_name = original_text[idx:end_idx]
                    
                    if original_name:
                        # Replace all occurrences of this name in the text
                        try:
                            pattern = rf'\b{re.escape(original_name)}\b'
                            matches = list(re.finditer(pattern, text, re.IGNORECASE))
                            for match in reversed(matches):
                                start_pos = match.start()
                                end_pos = match.end()
                                # Check if already replaced
                                if text[start_pos:end_pos].lower() != name_lower:
                                    continue
                                # Replace
                                text = text[:start_pos] + surrogate + text[end_pos:]
                        except Exception:
                            # If regex fails, try simple string replacement
                            try:
                                text = text.replace(original_name, surrogate)
                            except Exception:
                                pass
                except Exception:
                    # Skip this name if there's an error
                    continue
        except Exception as e:
            # If there's any error in the entire function, return unchanged
            # This prevents the whole de-identification from failing
            import traceback
            print(f"Error in _link_separate_name_parts: {e}")
            print(traceback.format_exc())
            return text, changes
        
        return text, changes
    
    def _link_standalone_last_names(self, text: str, changes: list) -> tuple:
        """
        Post-processing step to link standalone last names to existing full name mappings.
        If "Heidi Smith" was mapped to "Karen King", then standalone "Smith" should also map to "King".
        """
        # Build a map of last names to their full name surrogates
        # Prefer full name replacements WITHOUT titles (e.g., "Mark Hill" over "Ms. Mark Hill")
        last_name_to_surrogate = {}
        
        # Go through all existing mappings to extract last names
        for (entity_type, original_name), surrogate in self.phi_to_surrogate_map.items():
            if entity_type in ('PATIENT', 'STAFF'):
                # Extract last name from the original
                name_parts = original_name.split()
                if len(name_parts) >= 2:  # Full name (first + last)
                    last_name = name_parts[-1]
                    original_lower = original_name.lower()
                    # Check if this original has a title (Ms., Mr., etc.)
                    has_title = any(title in original_lower for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                    
                    # Store mapping: last_name -> (surrogate, entity_type)
                    # Prefer mappings without titles
                    if last_name not in last_name_to_surrogate:
                        last_name_to_surrogate[last_name] = (surrogate, entity_type)
                    elif not has_title:
                        # If we find a mapping without a title, prefer it over one with a title
                        existing_surrogate, _ = last_name_to_surrogate[last_name]
                        existing_has_title = any(title in existing_surrogate.lower() for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                        if existing_has_title:
                            # Replace with the non-title version
                            last_name_to_surrogate[last_name] = (surrogate, entity_type)
        
        # Also check changes list for full names that might not be in phi_to_surrogate_map yet
        for change in changes:
            original = change.get('original', '').strip()
            replacement = change.get('replacement', '').strip()
            entity_type = change.get('type', '')
            if entity_type in ('PATIENT', 'STAFF') and original and replacement:
                name_parts = original.split()
                if len(name_parts) >= 2:  # Full name (first + last)
                    last_name = name_parts[-1]
                    original_lower = original.lower()
                    has_title = any(title in original_lower for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                    
                    # Add to map if not already present, or prefer non-title version
                    if last_name not in last_name_to_surrogate:
                        last_name_to_surrogate[last_name] = (replacement, entity_type)
                    elif not has_title:
                        existing_surrogate, _ = last_name_to_surrogate[last_name]
                        existing_has_title = any(title in existing_surrogate.lower() for title in ['ms', 'mr', 'mrs', 'miss', 'dr', 'doctor'])
                        if existing_has_title:
                            last_name_to_surrogate[last_name] = (replacement, entity_type)
        
        if not last_name_to_surrogate:
            return text, changes
        
        # Replace standalone last names in the text
        for last_name, (surrogate, entity_type) in last_name_to_surrogate.items():
            # Extract the name part from surrogate (remove title if present)
            # e.g., "Ms. Jacob Anderson" -> "Jacob Anderson"
            surrogate_name = surrogate
            # Check if surrogate starts with a title
            title_pattern = r'^(Ms\.|Mr\.|Mrs\.|Miss\.|Dr\.|Doctor\.|Prof\.|Professor\.)\s+'
            title_match = re.match(title_pattern, surrogate, re.IGNORECASE)
            if title_match:
                # Remove the title to get just the name
                surrogate_name = re.sub(title_pattern, '', surrogate, flags=re.IGNORECASE).strip()
            
            # Use word boundaries to match standalone last names
            pattern = rf'\b{re.escape(last_name)}\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                start_pos = match.start()
                end_pos = match.end()
                matched_text = text[start_pos:end_pos]
                
                # CRITICAL: Check if this last name is already part of a replaced full name with title
                # Look at the context before this match to see if there's a title+name pattern
                # We need to check if the last name appears right after a title+first name pattern
                if start_pos > 0:
                    # Look back to see if there's a title+name pattern ending just before this match
                    check_start = max(0, start_pos - 100)
                    check_text = text[check_start:start_pos + len(matched_text)]
                    # Look for pattern: "Ms. [Name] [Last Name]" where [Last Name] is our match
                    # This catches cases where the full name was already replaced
                    full_name_pattern = rf'(Ms\.|Mr\.|Mrs\.|Miss\.|Dr\.|Doctor\.|Prof\.|Professor\.)\s+[A-Z][a-z]+\s+{re.escape(matched_text)}\b'
                    if re.search(full_name_pattern, check_text, re.IGNORECASE):
                        # This last name is already part of a replaced full name, skip it
                        continue
                    
                    # Also check if there's a pattern like "Ms. [Name] [Last Name]." followed by our match
                    # This handles cases like "Ms. Anthony Wright. Anthony Wright" where the second occurrence
                    # should not have "Wright" replaced again
                    before_match = text[check_start:start_pos]
                    # Look for "Ms. [Name] [Last Name]." or "Ms. [Name] [Last Name] " pattern
                    pattern_before = rf'(Ms\.|Mr\.|Mrs\.|Miss\.|Dr\.|Doctor\.|Prof\.|Professor\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+[\.\s]'
                    if re.search(pattern_before, before_match, re.IGNORECASE):
                        # Check if the last word before our match matches the surrogate name
                        # This means we're trying to replace a last name that's already part of a full name
                        words_before = before_match.strip().split()
                        if len(words_before) >= 3:
                            # Check if the pattern is "Ms. [Name] [Last Name]" and our match is the same last name
                            if words_before[-1].rstrip('.,').lower() == matched_text.lower():
                                # This is already part of a full name, skip it
                                continue
                
                # Check if already replaced
                already_replaced = False
                for change in changes:
                    if change.get('original', '').strip().lower() == matched_text.lower():
                        change_replacement = change.get('replacement', '').strip()
                        # Compare without titles
                        change_name = re.sub(title_pattern, '', change_replacement, flags=re.IGNORECASE).strip() if change_replacement else ''
                        if change_name.lower() == surrogate_name.lower():
                            already_replaced = True
                            break
                
                if not already_replaced:
                    # Replace this occurrence with the name part (without title)
                    text = text[:start_pos] + surrogate_name + text[end_pos:]
                    # Add to changes if not already there
                    if not any(c.get('original', '').strip().lower() == matched_text.lower() for c in changes):
                        changes.append({
                            'original': matched_text,
                            'replacement': surrogate_name,
                            'type': entity_type
                        })
        
        return text, changes
    
    def _link_standalone_first_names(self, text: str, changes: list) -> tuple:
        """
        Post-processing step to link standalone first names (and variations) to existing full name mappings.
        If "Robert Smith" was mapped to "Jennifer Thompson", then standalone "Rob" should also map to "Jennifer Thompson".
        Handles name variations like "Rob" -> "Robert", "Chuck" -> "Charles", etc.
        """
        # Build a map of first names (and variations) to their full name surrogates
        first_name_to_surrogate = {}
        
        # Go through all existing mappings to extract first names and their variations
        for (entity_type, original_name), surrogate in self.phi_to_surrogate_map.items():
            if entity_type in ('PATIENT', 'STAFF'):
                # Extract first name from the original
                name_parts = original_name.split()
                if len(name_parts) >= 2:  # Full name (first + last)
                    first_name = name_parts[0]
                    first_name_lower = first_name.lower()
                    
                    # Store mapping: first_name -> (surrogate, entity_type)
                    if first_name_lower not in first_name_to_surrogate:
                        first_name_to_surrogate[first_name_lower] = (surrogate, entity_type)
                    
                    # Also add variations of the first name
                    if first_name_lower in self.name_variations:
                        for variation in self.name_variations[first_name_lower]:
                            if variation not in first_name_to_surrogate:
                                first_name_to_surrogate[variation] = (surrogate, entity_type)
                    # Also check reverse - if a variation maps back to this first name
                    for var_name, var_list in self.name_variations.items():
                        if first_name_lower in var_list and var_name not in first_name_to_surrogate:
                            first_name_to_surrogate[var_name] = (surrogate, entity_type)
        
        # Also check changes list for full names that might not be in phi_to_surrogate_map yet
        # This handles cases where full names were processed but not yet in the map
        # Also handle standalone first names that were detected separately
        for change in changes:
            original = change.get('original', '').strip()
            replacement = change.get('replacement', '').strip()
            entity_type = change.get('type', '')
            if entity_type in ('PATIENT', 'STAFF') and original and replacement:
                name_parts = original.split()
                if len(name_parts) >= 2:  # Full name (first + last)
                    first_name = name_parts[0]
                    first_name_lower = first_name.lower()
                    
                    # Store mapping: first_name -> (surrogate, entity_type)
                    if first_name_lower not in first_name_to_surrogate:
                        first_name_to_surrogate[first_name_lower] = (replacement, entity_type)
                    
                    # Also add variations of the first name
                    if first_name_lower in self.name_variations:
                        for variation in self.name_variations[first_name_lower]:
                            if variation not in first_name_to_surrogate:
                                first_name_to_surrogate[variation] = (replacement, entity_type)
                    # Also check reverse - if a variation maps back to this first name
                    for var_name, var_list in self.name_variations.items():
                        if first_name_lower in var_list and var_name not in first_name_to_surrogate:
                            first_name_to_surrogate[var_name] = (replacement, entity_type)
                elif len(name_parts) == 1:  # Standalone first name (e.g., "Robert" detected separately)
                    # This is a standalone first name - use it directly
                    first_name = name_parts[0]
                    first_name_lower = first_name.lower()
                    
                    # Store mapping: first_name -> (surrogate, entity_type)
                    if first_name_lower not in first_name_to_surrogate:
                        first_name_to_surrogate[first_name_lower] = (replacement, entity_type)
                    
                    # Also add variations of the first name
                    if first_name_lower in self.name_variations:
                        for variation in self.name_variations[first_name_lower]:
                            if variation not in first_name_to_surrogate:
                                first_name_to_surrogate[variation] = (replacement, entity_type)
                    # Also check reverse - if a variation maps back to this first name
                    for var_name, var_list in self.name_variations.items():
                        if first_name_lower in var_list and var_name not in first_name_to_surrogate:
                            first_name_to_surrogate[var_name] = (replacement, entity_type)
        
        if not first_name_to_surrogate:
            return text, changes
        
        def replace_standalone_first_name(match):
            potential_first_name = match.group(1)
            full_match = match.group(0)
            potential_first_name_lower = potential_first_name.lower()
            
            # Check if this is a known first name or variation from our mappings
            if potential_first_name_lower in first_name_to_surrogate:
                surrogate, entity_type = first_name_to_surrogate[potential_first_name_lower]
                
                # Use the full surrogate name (e.g., "Jennifer Thompson") to maintain consistency
                surrogate_full_name = surrogate
                
                # Check if this replacement was already added to changes
                if not any(c.get('original') == full_match and c.get('replacement') == surrogate_full_name for c in changes):
                    changes.append({
                        'original': full_match,
                        'replacement': surrogate_full_name,
                        'type': entity_type
                    })
                
                return surrogate_full_name
            
            return full_match
        
        # Match standalone capitalized words that are known first names or variations
        # Use word boundaries and check context to avoid false positives
        # Only match if it's not already part of a highlighted/replaced entity
        first_names_pattern = '|'.join(re.escape(fn) for fn in first_name_to_surrogate.keys())
        if first_names_pattern:
            # Pattern: word boundary, capitalized first name, word boundary
            # But exclude if it's part of a longer name or after a title
            # Be more conservative - only match if it's likely a standalone first name
            pattern = r'\b(' + first_names_pattern + r')\b'
            
            # Only replace if it's not already in the changes list as part of a full name
            def should_replace_first_name(match):
                potential_first_name = match.group(1)
                # Check if this exact match is already in changes (meaning it was part of a full name)
                for change in changes:
                    if change.get('original') == potential_first_name:
                        return False  # Already processed
                return True
            
            # Apply replacement only for standalone first names
            def replace_if_standalone_first_name(match):
                if should_replace_first_name(match):
                    return replace_standalone_first_name(match)
                return match.group(0)
            
            text = re.sub(pattern, replace_if_standalone_first_name, text, flags=re.IGNORECASE)
        
        # Additional pass: Directly search for known first names (and variations) in text and replace them
        # This is more aggressive and handles edge cases where the pattern might not match
        # Process in reverse order to preserve positions when replacing
        for first_name_key, (surrogate, entity_type) in reversed(list(first_name_to_surrogate.items())):
            first_name_lower = first_name_key.lower()
            # Search for the first name as a standalone word (case-insensitive)
            # Pattern: word boundary, first name, word boundary
            pattern_direct = rf'\b{re.escape(first_name_key)}\b'
            # Find all matches (process in reverse to preserve positions)
            matches = list(re.finditer(pattern_direct, text, re.IGNORECASE))
            for match in reversed(matches):
                matched_text = match.group(0)
                matched_lower = matched_text.lower()
                start_pos = match.start()
                end_pos = match.end()
                
                # Check if this is already part of a full name that was replaced
                # by checking if there's a surrogate name immediately after it
                after_context = text[end_pos:min(len(text), end_pos + 30)].strip()
                # Check if any surrogate is in the after context (indicating this is part of a replaced full name)
                is_part_of_full_name = False
                for other_first_name, (other_surrogate, _) in first_name_to_surrogate.items():
                    if other_surrogate.lower() in after_context.lower():
                        # Check if the surrogate appears right after this match
                        surrogate_pos = after_context.lower().find(other_surrogate.lower())
                        if surrogate_pos != -1:
                            # Check if surrogate is at the start of after_context (immediately after our match)
                            if surrogate_pos <= 5:
                                is_part_of_full_name = True
                                break
                
                if is_part_of_full_name:
                    continue  # Skip - this is part of a full name that's already been replaced
                
                # Check if this match is already in changes with the correct surrogate
                already_replaced = False
                for change in changes:
                    change_original = change.get('original', '').strip()
                    if change_original.lower() == matched_lower:
                        change_surrogate = change.get('replacement', '').strip()
                        if change_surrogate.lower() == surrogate.lower():
                            # Already replaced correctly, skip
                            already_replaced = True
                            break
                
                if not already_replaced:
                    # Check if the text at this position still contains the original (not already replaced)
                    actual_text_at_pos = text[start_pos:end_pos]
                    if matched_lower in actual_text_at_pos.lower():
                        # Replace this occurrence
                        text = text[:start_pos] + surrogate + text[end_pos:]
                        # Add to changes if not already there
                        if not any(c.get('original', '').strip().lower() == matched_lower for c in changes):
                            changes.append({
                                'original': matched_text,
                                'replacement': surrogate,
                                'type': entity_type
                            })
        
        return text, changes
    
    def _is_valid_phi_entity(self, entity_text: str, tag: str, context_tokens: list = None) -> bool:
        """
        Validates if an entity is a legitimate PHI entity using general principles.
        Filters out false positives from the model based on patterns, not specific words.
        """
        entity_lower = entity_text.lower().strip()
        entity_clean = entity_text.strip()
        
        # 1. GENERAL: Filter out very short entities (likely false positives)
        if len(entity_clean) <= 1:
            return False
        
        # 2. GENERAL: Filter out measurements and numeric patterns
        # Pattern: number + optional whitespace + optional unit markers (quotes, common units)
        # This catches: "15'", "15 '", "10ft", "5\"", etc.
        if re.match(r'^\d+\s*[\'"]?$', entity_clean):
            return False
        if re.match(r'^\d+\s*[\'"]', entity_clean):
            return False
        # Pure numbers (after removing quotes/spaces)
        if entity_clean.replace("'", "").replace('"', "").replace(" ", "").isdigit():
            return False
        
        # 3. GENERAL: Filter out common English function words and very common nouns
        # These are grammatical/structural words, not PHI entities
        # Pattern: single word, common function word or very short common noun
        # Only apply to single words to avoid false positives on phrases
        if ' ' not in entity_clean:
            # Very common function words (grammatical words)
            common_function_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'from', 'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'there',
                'near', 'area'
            }
            if entity_lower in common_function_words:
                return False
            
            # Very short common geographic/noun words (3-4 chars) that are unlikely to be PHI alone
            # These are common enough that standalone they're probably not PHI
            # But we're conservative - only filter very short ones
            very_short_common_words = {'bay', 'sea', 'lake', 'hill', 'park', 'road', 'lane', 'way', 'creek', 'river'}
            if len(entity_clean) <= 4 and entity_lower in very_short_common_words:
                return False
            
            # Filter out standalone titles (Ms, Mr, Mrs, Dr, etc.) - these are not PHI by themselves
            # Titles should only be considered PHI when followed by a name (handled by _process_titles_with_names)
            standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor', 'sir', 'madam', 'ma\'am'}
            if entity_lower in standalone_titles:
                return False
        
        # 3b. GENERAL: Filter out common geographic location patterns that are not PHI
        # These are common place names that appear in many documents and aren't identifying information
        # Pattern: Multi-word entities ending in common geographic terms
        if ' ' in entity_clean:
            words = entity_clean.lower().split()
            # Common geographic suffixes/words that indicate non-PHI locations
            common_geographic_terms = {
                'bay', 'point', 'creek', 'river', 'lake', 'hill', 'park', 'road', 'lane', 'way', 
                'street', 'avenue', 'boulevard', 'drive', 'court', 'place', 'trail', 'beach',
                'harbor', 'harbour', 'port', 'island', 'peninsula', 'mountain', 'valley', 'ridge',
                'spring', 'falls', 'bridge', 'tunnel', 'dam', 'reservoir', 'canal', 'channel'
            }
            
            # If the entity ends with a common geographic term, it's likely a generic location name, not PHI
            # Examples: "Adventure Bay", "Spring Creek", "Bay Point"
            if words[-1] in common_geographic_terms:
                # Common descriptive words that often appear with geographic terms
                common_descriptors = {
                    'adventure', 'spring', 'summer', 'winter', 'autumn', 'fall',
                    'north', 'south', 'east', 'west', 'upper', 'lower', 'middle',
                    'big', 'little', 'small', 'long', 'short', 'deep', 'shallow',
                    'blue', 'green', 'red', 'white', 'black', 'gold', 'silver',
                    'old', 'new', 'young', 'ancient', 'modern',
                    'rocky', 'sandy', 'muddy', 'clear', 'dark', 'bright'
                }
                
                # Additional check: if it's a two-word entity and the first word is also common/descriptive
                # (not a proper name), it's likely not PHI
                if len(words) == 2:
                    first_word = words[0]
                    # Filter if first word is a descriptor OR if it's also a geographic term
                    # Examples: "Bay Point" (both geographic), "Adventure Bay" (descriptor + geographic)
                    if first_word in common_descriptors or first_word in common_geographic_terms:
                        return False
                # For longer entities, if they end in geographic terms, they're likely generic
                # But be more conservative - only filter if it's clearly a common pattern
                elif len(words) >= 2:
                    # Check if it's a pattern like "X Bay", "X Point", "X Creek" where X is descriptive or geographic
                    if words[-1] in {'bay', 'point', 'creek', 'river', 'lake', 'hill', 'park'}:
                        # If the second-to-last word is also common (descriptor or geographic), filter it
                        if len(words) >= 2 and (words[-2] in common_descriptors or words[-2] in common_geographic_terms):
                            return False
        
        # 4. GENERAL: Filter out very short single words (likely false positives)
        # Multi-word entities are more likely to be valid PHI
        if ' ' not in entity_clean:
            # Filter out standalone titles FIRST (before length check)
            # Titles should only be considered PHI when followed by a name (handled by _process_titles_with_names)
            # Handle both with and without periods: "Ms", "Ms.", "Mr", "Mr.", etc.
            entity_no_period = entity_lower.rstrip('.')
            standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor', 'sir', 'madam', 'ma\'am'}
            if entity_no_period in standalone_titles:
                return False
            
            # Single word entities should have minimum length based on entity type
            min_lengths = {
                'HOSP': 3,  # Hospitals need meaningful names
                'LOC': 3,   # Locations need meaningful names
                'PATORG': 4, # Organizations typically longer
            }
            min_len = min_lengths.get(tag, 2)  # Default minimum of 2
            if len(entity_clean) < min_len:
                return False
        
        # 5. GENERAL: Filter out punctuation-only or non-alphanumeric entities
        if not any(c.isalnum() for c in entity_clean):
            return False
        
        # 6. GENERAL: Filter out entities that are mostly numbers with minimal text
        # Pattern: mostly digits with very few letters (e.g., "15'", "123abc")
        alnum_chars = [c for c in entity_clean if c.isalnum()]
        if len(alnum_chars) > 0:
            digit_ratio = sum(1 for c in alnum_chars if c.isdigit()) / len(alnum_chars)
            # If more than 80% are digits and total length is short, likely not PHI
            if digit_ratio > 0.8 and len(entity_clean) <= 5:
                return False
        
        return True

    def _determine_entity_type_from_context(self, entity_text: str, original_tokens: list, entity_start_idx: int, entity_end_idx: int, original_text: str = None) -> str:
        """
        Determines if a name entity should be classified as STAFF or PATIENT based on context.
        Key indicator: PATIENT names often have DOB nearby (especially AFTER), STAFF names typically don't.
        Also checks for staff-related keywords like "Operator:", "Dr.", etc. (especially BEFORE).
        """
        # PRIORITY 0: Explicit check for "Operator" immediately after the name
        # This handles cases like "HEIDI SMITH, Operator" or "HEIDI SMITH, Operator:"
        # Check the tokens immediately after the entity first (most reliable for token-based processing)
        immediate_check_window = 6  # Check up to 6 tokens after (to catch ", Operator:")
        immediate_check_end = min(len(original_tokens), entity_end_idx + immediate_check_window)
        immediate_tokens = original_tokens[entity_end_idx:immediate_check_end]
        
        # Reconstruct the text from tokens, handling subword tokens properly
        immediate_text_parts = []
        for t in immediate_tokens:
            if t.startswith('##'):
                if immediate_text_parts:
                    immediate_text_parts[-1] += t[2:]
                else:
                    immediate_text_parts.append(t[2:])
            else:
                immediate_text_parts.append(t)
        immediate_text = ' '.join(immediate_text_parts).lower()
        immediate_joined = ''.join(immediate_text_parts).lower()
        
        # Check for "Operator" with various patterns (comma, space, colon, etc.)
        # Most specific patterns first - these must match at the START of immediate_text
        operator_patterns_start = [
            r'^,\s*operator\s*:?',  # Starts with ", Operator" or ", Operator:"
            r'^\s+operator\s*:?',    # Starts with " Operator" or " Operator:"
        ]
        
        for pattern in operator_patterns_start:
            if re.search(pattern, immediate_text):
                return 'STAFF'  # If "Operator" appears immediately after the name, it's staff
        
        # Also check patterns that might appear after some whitespace/punctuation
        operator_patterns = [
            r',\s*operator\s*:?',  # ", Operator" or ", Operator:"
            r'\s+operator\s*:?',    # " Operator" or " Operator:"
            r'\boperator\s*:?',     # "Operator" or "Operator:"
        ]
        
        for pattern in operator_patterns:
            if re.search(pattern, immediate_text):
                return 'STAFF'  # If "Operator" appears after the name, it's staff
        
        # Also check in joined tokens (handles tokenization issues like "Opera" + "##tor")
        if 'operator' in immediate_joined:
            # Check if it's immediately after (within first few characters)
            operator_pos = immediate_joined.find('operator')
            # Check if there's a comma or minimal text before "operator" (indicating it's after the name)
            before_operator = immediate_joined[:operator_pos]
            # Allow for comma, space, or very short text (like just punctuation)
            if ',' in before_operator or len(before_operator.strip()) <= 2:
                return 'STAFF'
        
        # Additional explicit check: look for comma followed by "operator" in the first few tokens
        # This handles cases where tokenization splits things oddly
        if len(immediate_tokens) >= 2:
            # Check if first token is comma and second starts with "oper" (case-insensitive)
            first_token = immediate_tokens[0].replace('##', '').lower()
            second_token = immediate_tokens[1].replace('##', '').lower() if len(immediate_tokens) > 1 else ''
            # Handle subword tokens
            if len(immediate_tokens) > 2:
                # Check if tokens form ", Operator" or ", Operator:"
                token_sequence = ' '.join([t.replace('##', '') for t in immediate_tokens[:3]]).lower()
                if re.search(r',\s*oper', token_sequence):
                    return 'STAFF'
            elif first_token == ',' and 'oper' in second_token:
                return 'STAFF'
        
        # Also try using original_text if available (as a secondary check)
        if original_text and entity_start_idx is not None and entity_end_idx is not None:
            # Reconstruct text up to entity_end_idx to find the exact position
            # This helps when the same name appears multiple times
            try:
                # Get tokens up to the entity end
                tokens_before_entity = original_tokens[:entity_end_idx]
                # Reconstruct the text up to this point
                text_before_entity = ' '.join([t.replace('##', '') for t in tokens_before_entity])
                # Find the position in original text
                # Use a more precise method: find the entity at the position where tokens_before_entity ends
                # This is approximate but should work for most cases
                entity_lower = entity_text.lower().strip()
                # Try to find the entity near the end of text_before_entity
                # Look for the entity in a window around where it should be
                search_start = max(0, len(text_before_entity) - len(entity_text) - 10)
                search_text = original_text[search_start:search_start + len(entity_text) + 30].lower()
                entity_pos_in_search = search_text.find(entity_lower)
                if entity_pos_in_search != -1:
                    entity_pos_absolute = search_start + entity_pos_in_search
                    # Check the text immediately after the entity (up to 25 characters)
                    after_entity = original_text[entity_pos_absolute + len(entity_text):entity_pos_absolute + len(entity_text) + 25].lower()
                    # Check for ", Operator" or " Operator" patterns
                    if re.search(r',\s*operator\s*:?', after_entity) or re.search(r'\s+operator\s*:?', after_entity):
                        return 'STAFF'  # If "Operator" appears after the name, it's staff
            except Exception:
                # If anything goes wrong with text position calculation, fall through to other checks
                pass
        
        # Check a window of tokens around the entity (before and after)
        context_window_before = 5  # Check 5 tokens before
        context_window_after = 10  # Check 10 tokens after (DOB might be a bit further)
        start_check = max(0, entity_start_idx - context_window_before)
        end_check = min(len(original_tokens), entity_end_idx + context_window_after)
        
        # Get context tokens
        context_tokens = original_tokens[start_check:end_check]
        context_text = ' '.join([t.replace('##', '') for t in context_tokens]).lower()
        
        # Calculate relative position of entity in context
        entity_start_in_context = entity_start_idx - start_check
        entity_end_in_context = entity_end_idx - start_check
        
        # PRIORITY 1: Check immediate after context for both DOB and staff keywords
        # Determine which appears FIRST - this indicates the entity type
        immediate_after_window = 8  # Increased to catch both "Operator:" and "DOB"
        immediate_after_end = min(len(original_tokens), entity_end_idx + immediate_after_window)
        immediate_after_tokens = original_tokens[entity_end_idx:immediate_after_end]
        immediate_after_text = ' '.join([t.replace('##', '') for t in immediate_after_tokens]).lower()
        immediate_after_joined = ''.join([t.replace('##', '') for t in immediate_after_tokens]).lower()
        
        # Check for DOB patterns in immediate after context
        dob_patterns = [
            r'\bdob\b',
            r'\bdate\s+of\s+birth\b',
            r'\bborn\b',
            r'\bbirth\s+date\b'
        ]
        dob_pos = None
        for pattern in dob_patterns:
            match = re.search(pattern, immediate_after_text)
            if match:
                dob_pos = match.start()
                break
        # Also check joined tokens
        if dob_pos is None:
            if 'dob' in immediate_after_joined:
                dob_pos = immediate_after_joined.find('dob')
        
        # Check for staff keywords in immediate after context
        # IMPORTANT: Check for ", operator:" pattern FIRST (most specific)
        # This handles "HEIDI SMITH, Operator:" correctly
        staff_keywords_after = [
            r',\s*operator\s*:',  # Handle ", Operator:" - check this FIRST
            r'\boperator\s*:',
            r'\bdr\.',
            r'\bdoctor\b',
            r'\bnurse\b',
            r'\bphysician\b'
        ]
        staff_pos = None
        for pattern in staff_keywords_after:
            match = re.search(pattern, immediate_after_text)
            if match:
                staff_pos = match.start()
                break
        # Also check joined tokens
        if staff_pos is None:
            if 'operator:' in immediate_after_joined or ',operator:' in immediate_after_joined:
                staff_pos = immediate_after_joined.find('operator:')
                if staff_pos == -1:
                    staff_pos = immediate_after_joined.find(',operator:')
        
        # If both found, return the one that appears first
        if dob_pos is not None and staff_pos is not None:
            if dob_pos < staff_pos:
                return 'PATIENT'  # DOB appears before staff keyword
            else:
                return 'STAFF'  # Staff keyword appears before DOB
        elif dob_pos is not None:
            return 'PATIENT'  # Only DOB found
        elif staff_pos is not None:
            return 'STAFF'  # Only staff keyword found
        
        # Check for DOB patterns in immediate after context
        dob_patterns = [
            r'\bdob\b',
            r'\bdate\s+of\s+birth\b',
            r'\bborn\b',
            r'\bbirth\s+date\b'
        ]
        
        # Check for staff keywords in immediate after context
        # Note: "Operator" can appear with or without colon, and may have a comma before it
        staff_keywords_after = [
            r',\s*operator\s*:?',  # ", Operator:" or ", Operator"
            r'\boperator\s*:',
            r'\bdr\.',
            r'\bdoctor\b',
            r'\bnurse\b',
            r'\bphysician\b',
            r'\bstaff\b',
            r'\battending\b',
            r'\bresident\b',
            r'\bprovider\b',
            r'\bclinician\b'
        ]
        
        # Find positions of DOB and staff keywords in the immediate after context
        # Use immediate_after_text for consistent position comparison
        dob_pos = None
        staff_pos = None
        
        # Check DOB position in immediate_after_text
        for pattern in dob_patterns:
            match = re.search(pattern, immediate_after_text)
            if match:
                dob_pos = match.start()
                break
        
        # Check staff keyword position in immediate_after_text
        # First try regex patterns
        for pattern in staff_keywords_after:
            match = re.search(pattern, immediate_after_text)
            if match:
                staff_pos = match.start()
                break
        
        # If regex didn't match, try simple string search
        if staff_pos is None:
            # Try without word boundary (handles cases like ", Operator:" or ", Operator")
            # Check for "operator" with or without colon, with or without comma before
            # Priority: check ", operator" first (most specific pattern)
            # Also check in joined tokens for tokenization issues
            for keyword in [', operator:', ', operator', 'operator:', 'operator']:
                pos = immediate_after_text.find(keyword)
                if pos != -1:
                    staff_pos = pos
                    break
            # If still not found, check in joined tokens
            if staff_pos is None:
                for keyword in [',operator:', ',operator', 'operator:', 'operator']:
                    pos = immediate_after_joined.find(keyword)
                    if pos != -1:
                        # Approximate position in text
                        chars_before = len(immediate_after_joined[:pos])
                        staff_pos = len(immediate_after_text[:chars_before].split()) if chars_before > 0 else 0
                        break
        
        # If still not found, try in joined tokens (handles tokenization)
        if staff_pos is None:
            if 'operator:' in immediate_after_joined:
                # Map position from joined to text (approximate)
                operator_pos_joined = immediate_after_joined.find('operator:')
                # Count characters before operator in joined
                chars_before = len(immediate_after_joined[:operator_pos_joined])
                # Approximate position in text (accounting for spaces)
                staff_pos = len(immediate_after_text[:chars_before].split())
        
        # If both found, return the one that appears first
        if dob_pos is not None and staff_pos is not None:
            if staff_pos < dob_pos:
                return 'STAFF'  # Staff keyword appears before DOB
            else:
                return 'PATIENT'  # DOB appears before staff keyword
        elif dob_pos is not None:
            return 'PATIENT'  # Only DOB found
        elif staff_pos is not None:
            return 'STAFF'  # Only staff keyword found
        
        # PRIORITY 2: Check for staff-related keywords IMMEDIATELY BEFORE or AFTER the entity - strong indicator of STAFF
        # Staff keywords like "Operator:" can appear before or after staff names
        # Check this SECOND - only if DOB was not found
        # Check a window before (up to 3 tokens) and after (up to 3 tokens) to catch cases like "HEIDI SMITH, Operator:"
        immediate_before_window = 3
        immediate_before_start = max(0, entity_start_idx - immediate_before_window)
        before_context_tokens = original_tokens[immediate_before_start:entity_start_idx]
        before_context_text = ' '.join([t.replace('##', '') for t in before_context_tokens]).lower()
        before_tokens_joined = ''.join([t.replace('##', '') for t in before_context_tokens]).lower()
        
        staff_keywords = [
            r'\boperator\s*:',
            r'\bdr\.',
            r'\bdoctor\b',
            r'\bnurse\b',
            r'\bphysician\b',
            r'\bstaff\b',
            r'\battending\b',
            r'\bresident\b',
            r'\bprovider\b',
            r'\bclinician\b'
        ]
        # Check with spaces (normal text) - before context
        for pattern in staff_keywords:
            if re.search(pattern, before_context_text):
                return 'STAFF'  # Staff keyword immediately before = STAFF
        # Check joined tokens (handles tokenization splits) - before context
        if 'operator:' in before_tokens_joined or 'operator' in before_tokens_joined:
            return 'STAFF'
        # Check with spaces (normal text) - after context
        for pattern in staff_keywords:
            if re.search(pattern, immediate_after_text):
                return 'STAFF'  # Staff keyword immediately after = STAFF
        # Check joined tokens (handles tokenization splits) - after context
        if 'operator:' in immediate_after_joined or 'operator' in immediate_after_joined:
            return 'STAFF'
        # Check for other staff keywords in joined tokens
        staff_keywords_joined = ['doctor', 'nurse', 'physician', 'staff', 'attending', 'resident', 'provider', 'clinician']
        for keyword in staff_keywords_joined:
            if keyword in before_tokens_joined or keyword in immediate_after_joined:
                return 'STAFF'
        
        staff_keywords = [
            r'\boperator\s*:',
            r'\bdr\.',
            r'\bdoctor\b',
            r'\bnurse\b',
            r'\bphysician\b',
            r'\bstaff\b',
            r'\battending\b',
            r'\bresident\b',
            r'\bprovider\b',
            r'\bclinician\b'
        ]
        # Check with spaces (normal text) - before context
        for pattern in staff_keywords:
            if re.search(pattern, before_context_text):
                return 'STAFF'  # Staff keyword immediately before = STAFF
        # Check joined tokens (handles tokenization splits) - before context
        if 'operator:' in before_tokens_joined or 'operator' in before_tokens_joined:
            return 'STAFF'
        # Check with spaces (normal text) - after context
        for pattern in staff_keywords:
            if re.search(pattern, immediate_after_text):
                return 'STAFF'  # Staff keyword immediately after = STAFF
        # Check joined tokens (handles tokenization splits) - after context
        if 'operator:' in immediate_after_joined or 'operator' in immediate_after_joined:
            return 'STAFF'
        # Check for other staff keywords in joined tokens
        staff_keywords_joined = ['doctor', 'nurse', 'physician', 'staff', 'attending', 'resident', 'provider', 'clinician']
        for keyword in staff_keywords_joined:
            if keyword in before_tokens_joined or keyword in immediate_after_joined:
                return 'STAFF'
        # DOB typically appears after patient names, not before
        # This is a strong signal - if DOB is right after the name, it's definitely a patient
        after_context_tokens = original_tokens[entity_end_idx:end_check]
        after_context_text = ' '.join([t.replace('##', '') for t in after_context_tokens]).lower()
        
        # Also check raw tokens for DOB (handles tokenization like "DO" + "##B" = "DOB")
        after_tokens_joined = ''.join([t.replace('##', '') for t in after_context_tokens]).lower()
        
        # Check for DOB patterns in after context
        dob_patterns = [
            r'\bdob\b',
            r'\bdate\s+of\s+birth\b',
            r'\bborn\b',
            r'\bbirth\s+date\b'
        ]
        # Check after context text (with spaces)
        for pattern in dob_patterns:
            if re.search(pattern, after_context_text):
                return 'PATIENT'  # DOB after name = PATIENT
        # Check joined tokens (handles tokenization splits like "DO" + "##B")
        if 'dob' in after_tokens_joined or 'd o b' in after_tokens_joined.replace(' ', ''):
            return 'PATIENT'
        # Staff keywords like "Operator:" typically appear immediately before staff names
        # Only check if DOB was NOT found after (to avoid false matches)
        # Check a window before (up to 5 tokens) to catch cases like "HEIDI SMITH, Operator:"
        immediate_before_window = 5  # Increased to catch "Operator:" after comma
        immediate_before_start = max(0, entity_start_idx - immediate_before_window)
        before_context_tokens = original_tokens[immediate_before_start:entity_start_idx]
        before_context_text = ' '.join([t.replace('##', '') for t in before_context_tokens]).lower()
        # Also check joined tokens (handles tokenization like "Opera" + "##tor" = "Operator")
        before_tokens_joined = ''.join([t.replace('##', '') for t in before_context_tokens]).lower()
        
        staff_keywords = [
            r'\boperator\s*:',
            r'\bdr\.',
            r'\bdoctor\b',
            r'\bnurse\b',
            r'\bphysician\b',
            r'\bstaff\b',
            r'\battending\b',
            r'\bresident\b',
            r'\bprovider\b',
            r'\bclinician\b'
        ]
        # Check with spaces (normal text)
        for pattern in staff_keywords:
            if re.search(pattern, before_context_text):
                return 'STAFF'  # Staff keyword immediately before = STAFF
        # Check joined tokens (handles tokenization splits)
        if 'operator:' in before_tokens_joined or 'operator' in before_tokens_joined:
            return 'STAFF'
        # Check for other staff keywords in joined tokens
        staff_keywords_joined = ['doctor', 'nurse', 'physician', 'staff', 'attending', 'resident', 'provider', 'clinician']
        for keyword in staff_keywords_joined:
            if keyword in before_tokens_joined:
                return 'STAFF'
        
        # PRIORITY 3: Check for DOB in full context (fallback - only if not found in after context)
        # But ONLY if DOB appears AFTER the entity (not before)
        full_context_text = ' '.join([t.replace('##', '') for t in context_tokens]).lower()
        full_tokens_joined = ''.join([t.replace('##', '') for t in context_tokens]).lower()
        
        # Check full context (make sure DOB appears after entity)
        for pattern in dob_patterns:
            if re.search(pattern, full_context_text):
                # Make sure DOB appears after the entity in the full context
                entity_end_pos = len(' '.join([t.replace('##', '') for t in context_tokens[:entity_end_in_context]]).lower())
                dob_match = re.search(pattern, full_context_text)
                if dob_match and dob_match.start() > entity_end_pos:
                    return 'PATIENT'  # DOB appears after entity in full context
        # Check joined tokens in full context (only if DOB is after entity)
        entity_end_pos_joined = len(''.join([t.replace('##', '') for t in context_tokens[:entity_end_in_context]]).lower())
        dob_pos = full_tokens_joined.find('dob')
        if dob_pos == -1:
            # Try without spaces
            full_tokens_no_spaces = full_tokens_joined.replace(' ', '')
            dob_pos = full_tokens_no_spaces.find('dob')
            entity_end_pos_joined = len(''.join([t.replace('##', '').replace(' ', '') for t in context_tokens[:entity_end_in_context]]).lower())
        if dob_pos != -1 and dob_pos > entity_end_pos_joined:
            return 'PATIENT'
        
        # PRIORITY 4: Check for staff keywords anywhere in context (fallback - only if not found in immediate before)
        for pattern in staff_keywords:
            if re.search(pattern, context_text):
                return 'STAFF'
        
        # PRIORITY 5: Check if there's a date pattern very close AFTER the entity - likely patient DOB
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        if re.search(date_pattern, after_context_text):
            # Date found after entity - likely patient DOB
            return 'PATIENT'
        
        # Default: return original tag (let model decide if no clear context)
        return None  # None means use the original tag from model
    
    def _process_entity_span(self, entity_tokens: list, tag: str, deidentified_list: list, changes_list: list, original_tokens: list = None, entity_start_idx: int = None, entity_end_idx: int = None, original_text: str = None):
        """
        Takes a full PHI entity (list of tokens), gets the surrogate using a 
        type-aware deterministic map, and updates the final list and the changes log.
        """
        
        # 1. Reconstruct the full original entity text (handle BERT subwords)
        # Join the tokens, preserving spaces between words but not for subwords (##)
        original_entity_parts = []
        for t in entity_tokens:
            if t.startswith('##'):
                # Subword token - join without space to previous part
                if original_entity_parts:
                    original_entity_parts[-1] += t[2:]
                else:
                    original_entity_parts.append(t[2:])
            else:
                # Regular token - add as new part (will be joined with space)
                original_entity_parts.append(t)
        
        # Join parts with spaces to preserve word boundaries (e.g., "Heidi Smith")
        # Normalize the entity text (handle case variations)
        original_entity = " ".join(original_entity_parts)
        
        # CRITICAL CHECK: If this is a PATIENT/STAFF entity, check IMMEDIATELY if it's followed by "Operator"
        # This must happen before any other processing to prevent de-identification
        if tag in ('PATIENT', 'STAFF') and original_text:
            entity_clean = original_entity.strip()
            entity_lower = entity_clean.lower()
            
            # Check if this entity appears before "Operator" in the original text
            # Look for patterns like: "ENTITY, Operator:" or "ENTITY Operator:"
            # Also check if this entity is part of a larger name followed by Operator
            # (e.g., if entity is "SMITH", check if "HEIDI SMITH, Operator:" exists)
            
            # First, check if entity directly followed by Operator
            entity_pattern = re.escape(entity_lower)
            operator_pattern_direct = rf'\b{entity_pattern}\s*,\s*operator\s*:?'
            if re.search(operator_pattern_direct, original_text, re.IGNORECASE):
                deidentified_list.append(original_entity)
                return  # Exit early, don't process as PHI
            
            # Also check if entity is part of a larger name followed by Operator
            # Look for patterns like "[WORD] ENTITY, Operator:" or "ENTITY [WORD], Operator:"
            # This handles cases where "SMITH" is processed separately from "HEIDI"
            if entity_start_idx is not None and entity_start_idx > 0:
                # Check if there's a word before this entity
                before_tokens = original_tokens[max(0, entity_start_idx - 2):entity_start_idx] if original_tokens else []
                if before_tokens:
                    before_text = ' '.join([t.replace('##', '') for t in before_tokens]).strip()
                    if before_text:
                        # Check if "BEFORE_TEXT ENTITY, Operator:" exists
                        full_pattern = rf'\b{re.escape(before_text)}\s+{entity_pattern}\s*,\s*operator\s*:?'
                        if re.search(full_pattern, original_text, re.IGNORECASE):
                            deidentified_list.append(original_entity)
                            return  # Exit early, don't process as PHI
            
            # Also check if there's a word after this entity that, combined, forms a name before Operator
            if entity_end_idx is not None and original_tokens and entity_end_idx < len(original_tokens):
                after_tokens = original_tokens[entity_end_idx:min(len(original_tokens), entity_end_idx + 2)]
                if after_tokens:
                    after_text = ' '.join([t.replace('##', '') for t in after_tokens]).strip()
                    if after_text:
                        # Check if "ENTITY AFTER_TEXT, Operator:" exists
                        full_pattern = rf'\b{entity_pattern}\s+{re.escape(after_text)}\s*,\s*operator\s*:?'
                        if re.search(full_pattern, original_text, re.IGNORECASE):
                            deidentified_list.append(original_entity)
                            return  # Exit early, don't process as PHI
        
        # Normalize case for matching (preserve original for display, but use normalized for linking)
        # This helps link "HEIDI SMITH" with "Heidi Smith"
        original_entity_normalized_for_linking = ' '.join(original_entity.split()).lower()

        # NEW: Filter out single punctuation/separator tokens wrongly classified as PHI
        if len(original_entity) <= 1 and not original_entity.isalnum():
            # If it's just '/' or ':' or similar, treat it as non-PHI text
            deidentified_list.append(original_entity)
            return # Skip the rest of the processing for this span
        
        # NEW: Check if this is a standalone title that's part of a title+name combination
        # If "Ms." or "Ms" is followed by a capitalized name in the ORIGINAL text, skip replacing it here
        # Use original_text if available to check for title+name patterns before any replacements
        if tag in ('PATIENT', 'STAFF') and original_text:
            original_entity_lower = original_entity.lower().rstrip('.')
            standalone_titles = {'ms', 'mr', 'mrs', 'miss', 'dr', 'doctor', 'prof', 'professor'}
            if ' ' not in original_entity and original_entity_lower in standalone_titles:
                # This is a standalone title - check if it appears in a title+name pattern in the original text
                # Check for both "Ms." and "Ms" patterns (with and without period)
                title_with_period = original_entity if original_entity.endswith('.') else original_entity + '.'
                title_without_period = original_entity.rstrip('.')
                # Pattern: "Ms." or "Ms" followed by space and capitalized name
                title_name_pattern_with = rf'\b{re.escape(title_with_period)}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
                title_name_pattern_without = rf'\b{re.escape(title_without_period)}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
                if re.search(title_name_pattern_with, original_text, re.IGNORECASE) or re.search(title_name_pattern_without, original_text, re.IGNORECASE):
                    # This title is part of a title+name combination in the original text - skip replacing it here
                    # It will be handled by _process_titles_with_names
                    deidentified_list.append(original_entity)
                    return # Skip the rest of the processing for this span
        
        # NEW: Validate entity to filter out false positives
        if not self._is_valid_phi_entity(original_entity, tag):
            # Invalid entity - treat as regular text, don't replace
            deidentified_list.append(original_entity)
            return # Skip the rest of the processing for this span
        
        # 1.5. For PATIENT/STAFF entities, check context to refine classification
        # This helps distinguish between "HEIDI SMITH, Operator:" (STAFF) vs "Heidi Smith DOB" (PATIENT)
        # PRIORITY: Check for "Operator" immediately after the name FIRST (before other context checks)
        if tag in ('PATIENT', 'STAFF') and original_tokens is not None and entity_start_idx is not None and entity_end_idx is not None:
            # Quick explicit check for "Operator" right after the entity
            # Check up to 8 tokens to catch ", Operator:" pattern
            immediate_tokens = original_tokens[entity_end_idx:min(len(original_tokens), entity_end_idx + 8)]
            if immediate_tokens:
                # First, check raw tokens directly for comma followed by operator
                # Handle cases where comma is a separate token: [',', 'Operator', ':']
                token_texts = [t.replace('##', '') for t in immediate_tokens]
                token_texts_lower = [t.lower() for t in token_texts]
                
                # Check if we have comma (or comma-like) followed by operator
                # Pattern 1: comma token, then operator token
                if len(token_texts) >= 2:
                    # Check if first token is comma/punctuation and second starts with "oper"
                    first_token = token_texts_lower[0].strip()
                    second_token = token_texts_lower[1] if len(token_texts_lower) > 1 else ''
                    if first_token in [',', ':', ';'] and 'oper' in second_token:
                        tag = 'STAFF'
                    # Check if first token ends with comma and second is operator
                    elif first_token.endswith(',') and 'oper' in second_token:
                        tag = 'STAFF'
                    # Check if second token is operator (first might be space or empty)
                    elif len(first_token) <= 1 and 'oper' in second_token:
                        tag = 'STAFF'
                
                # Pattern 2: Reconstruct text and check
                if tag != 'STAFF':  # Only if not already set
                    immediate_parts = []
                    for t in immediate_tokens:
                        if t.startswith('##'):
                            if immediate_parts:
                                immediate_parts[-1] += t[2:]
                            else:
                                immediate_parts.append(t[2:])
                        else:
                            immediate_parts.append(t)
                    immediate_text_check = ' '.join(immediate_parts).lower()
                    immediate_joined_check = ''.join(immediate_parts).lower()
                    
                    # Check various patterns for ", Operator" or " Operator"
                    operator_patterns = [
                        r'^,\s*operator',      # Starts with ", operator"
                        r'^,\s*oper',         # Starts with ", oper" (handles tokenization)
                        r'^\s*,\s*operator',  # Starts with space, comma, operator
                        r'operator',          # Contains operator (will check position)
                    ]
                    
                    for pattern in operator_patterns:
                        if re.search(pattern, immediate_text_check, re.IGNORECASE):
                            # For the last pattern, verify it's early enough
                            if pattern == r'operator':
                                operator_pos = immediate_text_check.lower().find('operator')
                                if operator_pos < 20:  # Within first 20 chars
                                    before_op = immediate_text_check[:operator_pos]
                                    if ',' in before_op or len(before_op.strip()) <= 3:
                                        tag = 'STAFF'
                                        break
                            else:
                                tag = 'STAFF'
                                break
                    
                    # Also check joined version (handles tokenization like "Opera" + "##tor")
                    if tag != 'STAFF' and 'operator' in immediate_joined_check:
                        operator_pos = immediate_joined_check.find('operator')
                        if operator_pos < 20:
                            before_op = immediate_joined_check[:operator_pos]
                            if ',' in before_op or len(before_op.strip()) <= 3:
                                tag = 'STAFF'
            
            # Also check original_text directly for more reliable matching
            # This is the most reliable method - directly search the original text
            # Use case-insensitive search and handle variations in spacing/case
            if tag != 'STAFF' and original_text:
                entity_clean = original_entity.strip()
                # Create a flexible pattern that matches the entity name (case-insensitive, with flexible spacing)
                # followed by comma and Operator
                # Split entity into words to handle spacing variations
                entity_words = entity_clean.split()
                if entity_words:
                    # Build pattern: word1 word2 ... , Operator
                    # Allow flexible spacing between words
                    entity_pattern = r'\s+'.join([re.escape(word) for word in entity_words])
                    # Check for entity followed by ", Operator" or " Operator"
                    operator_patterns = [
                        rf'\b{entity_pattern}\s*,\s*operator\s*:?',  # Entity, Operator
                        rf'\b{entity_pattern}\s+operator\s*:?',      # Entity Operator
                    ]
                    for pattern in operator_patterns:
                        if re.search(pattern, original_text, re.IGNORECASE):
                            tag = 'STAFF'
                            break
                
                # If still not found, try even more flexible: just check if entity name appears
                # followed by ", Operator" anywhere in the text (handles case mismatches)
                if tag != 'STAFF':
                    entity_lower = entity_clean.lower()
                    # Find all occurrences of entity (case-insensitive) in text
                    for match in re.finditer(re.escape(entity_lower), original_text.lower()):
                        # Check what comes after this match
                        after_match = original_text[match.end():match.end() + 25].lower()
                        if re.search(r',\s*operator\s*:?', after_match) or re.search(r'\s+operator\s*:?', after_match):
                            tag = 'STAFF'
                            break
                
                # Final fallback: check if ANY name followed by ", Operator" appears in text
                # This catches cases where entity detection might be partial
                if tag != 'STAFF' and original_text:
                    # Look for pattern: [Name] , Operator (case-insensitive)
                    # This is a catch-all that should find "HEIDI SMITH, Operator" regardless of entity detection
                    operator_after_name_pattern = r'[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s*,\s*operator\s*:?'
                    matches = list(re.finditer(operator_after_name_pattern, original_text, re.IGNORECASE))
                    if matches:
                        # Check if any of these matches could be our entity
                        entity_lower = entity_clean.lower()
                        for match in matches:
                            matched_text = match.group(0).lower()
                            # Check if the matched name part matches our entity
                            name_part = matched_text.split(',')[0].strip()
                            if name_part == entity_lower or entity_lower.endswith(name_part) or name_part.endswith(entity_lower):
                                tag = 'STAFF'
                                break
            
            # Now do full context check (which will also check for Operator, but we've already handled it above)
            # Only do this if we haven't already determined it's STAFF
            if tag != 'STAFF':
                context_type = self._determine_entity_type_from_context(original_entity, original_tokens, entity_start_idx, entity_end_idx, original_text=original_text)
                if context_type:
                    tag = context_type  # Override tag based on context (this will also catch Operator cases)
        
        # NEW: Skip STAFF entities - staff names are not PHI/PII and should not be replaced
        # This check happens AFTER context detection so we can catch entities that should be STAFF
        if tag == 'STAFF':
            # Staff names are not PHI/PII - keep them as-is without replacement
            deidentified_list.append(original_entity)
            return # Skip the rest of the processing for this span
        
        # 2. Define the key for type-aware mapping
        # This ensures "Heidi" (PATIENT) is treated differently from "Heidi" (STAFF)
        # BUT: if the same name appears as both STAFF and PATIENT, link them together
        mapping_key = (tag, original_entity)
        
        # Check if this name (case-insensitive, normalized) already exists with a different entity type
        # If so, use the same surrogate to maintain consistency (same person, different role)
        original_entity_normalized = ' '.join(original_entity.split()).lower()  # Normalize whitespace and case
        existing_surrogate = None
        
        # Only check for cross-type linking for name entities (PATIENT, STAFF)
        # IMPORTANT: Don't link if context clearly indicates different types (STAFF vs PATIENT)
        # If one has DOB (PATIENT) and one has "Operator:" (STAFF), they're different people
        # Only link if we're not sure or if it's clearly the same person in different roles
        if tag in ('PATIENT', 'STAFF'):
            for (existing_tag, existing_original), existing_surr in self.phi_to_surrogate_map.items():
                if existing_tag in ('PATIENT', 'STAFF'):
                    existing_normalized = ' '.join(existing_original.split()).lower()
                    # Check if names match (case-insensitive, normalized)
                    # Match even if same type - handles case variations like "HEIDI SMITH" vs "Heidi Smith"
                    if existing_normalized == original_entity_normalized:
                        # Same name - but check if types are different
                        # If types are different (STAFF vs PATIENT), they might be different people
                        # Only link if types are the same, or if we're not sure
                        if existing_tag == tag:
                            # Same name, same type - definitely the same person
                            existing_surrogate = existing_surr
                            self.phi_to_surrogate_map[mapping_key] = existing_surrogate
                            break
                        # If types are different, don't auto-link - let them have different surrogates
                        # This handles cases like "HEIDI SMITH, Operator:" (STAFF) vs "Heidi Smith DOB" (PATIENT)
                        # They should be treated as different people
                    # Check if names are variations/nicknames (e.g., "Robert" vs "Rob" or "Bob")
                    # IMPORTANT: Only link if they're the same type (both PATIENT or both STAFF)
                    # Don't link across types (STAFF vs PATIENT are different people)
                    elif existing_tag == tag and self._is_name_variation(original_entity, existing_original):
                        # Name variations - same person, same type, use the same surrogate
                        existing_surrogate = existing_surr
                        self.phi_to_surrogate_map[mapping_key] = existing_surrogate
                        break
                    # Also try fuzzy matching for typos/case variations
                    elif self._fuzzy_match_name(original_entity_normalized, existing_normalized, threshold=0.85):
                        # Very similar names (high threshold) - likely the same person
                        existing_surrogate = existing_surr
                        self.phi_to_surrogate_map[mapping_key] = existing_surrogate
                        break
        
        # 3. Get the Surrogate replacement (This relies on the updated _get_surrogate)
        # _get_surrogate handles checking/storing the mapping in self.phi_to_surrogate_map
        if existing_surrogate:
            replacement_text = existing_surrogate
        else:
            replacement_text = self._get_surrogate(tag, original_entity, mapping_key)
        
        # 4. Append the single replacement to the de-identified list
        # The entire span (e.g., "John Smith") is replaced by one surrogate token
        deidentified_list.append(replacement_text)
        
        # 5. Record the single change for the log
        changes_list.append({
            'original': original_entity,
            'replacement': replacement_text,
            'type': tag
        })