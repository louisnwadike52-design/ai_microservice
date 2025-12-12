"""
AI Scan Service
Handles image processing and data extraction for various document types
"""

import os
import json
import base64
from typing import Dict, Any, Optional
from datetime import datetime
import re

# Image processing libraries
try:
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Warning: Some dependencies not installed: {e}")
    print("Install with: pip install pillow pytesseract opencv-python numpy")


class AiScanService:
    """Service for processing scanned images and extracting data"""

    def __init__(self):
        """Initialize the AI scan service"""
        self.supported_types = [
            'INVOICE',
            'UTILITY_BILL',
            'QR_CODE',
            'BARCODE',
            'ACCOUNT_DETAILS',
            'GIFT_CARD',
            'RECEIPT',
            'BANK_DETAILS'
        ]

    def process_image(self, image_data: bytes, scan_type: str) -> Dict[str, Any]:
        """
        Process an image and extract relevant data based on scan type

        Args:
            image_data: Raw image bytes
            scan_type: Type of document to scan

        Returns:
            Dictionary containing extracted data and metadata
        """
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {
                    'success': False,
                    'error_message': 'Failed to decode image',
                    'extracted_data': {}
                }

            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Extract text using OCR
            text = self._extract_text(processed_image)

            # Process based on scan type
            extracted_data = self._extract_data_by_type(text, scan_type, image)

            # Generate AI message
            ai_message = self._generate_ai_message(extracted_data, scan_type)

            return {
                'success': True,
                'extracted_data': extracted_data,
                'ai_message': ai_message,
                'confidence_score': extracted_data.get('confidence_score', 0.85)
            }

        except Exception as e:
            return {
                'success': False,
                'error_message': f'Image processing failed: {str(e)}',
                'extracted_data': {}
            }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        return denoised

    def _extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from preprocessed image using OCR

        Args:
            image: Preprocessed image

        Returns:
            Extracted text
        """
        try:
            # Convert numpy array to PIL Image for pytesseract
            pil_image = Image.fromarray(image)

            # Extract text
            text = pytesseract.image_to_string(pil_image, config='--psm 6')

            return text
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""

    def _extract_data_by_type(self, text: str, scan_type: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract specific data based on scan type

        Args:
            text: Extracted text from image
            scan_type: Type of scan
            image: Original image

        Returns:
            Extracted data dictionary
        """
        scan_type = scan_type.upper()

        if scan_type == 'INVOICE':
            return self._extract_invoice_data(text)
        elif scan_type == 'UTILITY_BILL':
            return self._extract_utility_bill_data(text)
        elif scan_type == 'QR_CODE':
            return self._extract_qr_code_data(image)
        elif scan_type == 'BARCODE':
            return self._extract_barcode_data(image)
        elif scan_type == 'ACCOUNT_DETAILS' or scan_type == 'BANK_DETAILS':
            return self._extract_account_details(text)
        elif scan_type == 'GIFT_CARD':
            return self._extract_gift_card_data(text)
        elif scan_type == 'RECEIPT':
            return self._extract_receipt_data(text)
        else:
            return self._extract_generic_data(text)

    def _extract_invoice_data(self, text: str) -> Dict[str, Any]:
        """Extract data from invoice"""
        data = {
            'recipient': self._find_company_name(text),
            'amount': self._find_amount(text),
            'currency': 'USD',
            'reference': self._find_invoice_number(text),
            'due_date': self._find_date(text),
            'description': 'Invoice payment',
            'confidence_score': 0.85
        }
        return data

    def _extract_utility_bill_data(self, text: str) -> Dict[str, Any]:
        """Extract data from utility bill"""
        data = {
            'recipient': self._find_utility_company(text),
            'amount': self._find_amount(text),
            'currency': 'USD',
            'reference': self._find_account_number(text),
            'account_number': self._find_account_number(text),
            'due_date': self._find_date(text),
            'description': 'Utility bill payment',
            'bill_period': self._find_bill_period(text),
            'confidence_score': 0.82
        }
        return data

    def _extract_qr_code_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from QR code"""
        try:
            from pyzbar.pyzbar import decode

            # Decode QR code
            decoded_objects = decode(image)

            if decoded_objects:
                qr_data = decoded_objects[0].data.decode('utf-8')

                # Try to parse as JSON (common for payment QR codes)
                try:
                    parsed_data = json.loads(qr_data)
                    return {
                        'recipient': parsed_data.get('recipient', 'Unknown'),
                        'amount': float(parsed_data.get('amount', 0)),
                        'currency': parsed_data.get('currency', 'USD'),
                        'reference': parsed_data.get('reference', ''),
                        'description': 'QR code payment',
                        'merchant_id': parsed_data.get('merchant_id', ''),
                        'confidence_score': 0.95
                    }
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    return {
                        'recipient': 'QR Code Merchant',
                        'amount': 0.0,
                        'currency': 'USD',
                        'reference': qr_data[:50],
                        'description': 'QR code payment',
                        'raw_data': qr_data,
                        'confidence_score': 0.90
                    }
            else:
                return {
                    'recipient': 'Unknown',
                    'amount': 0.0,
                    'currency': 'USD',
                    'description': 'QR code not detected',
                    'confidence_score': 0.0
                }

        except ImportError:
            print("pyzbar not installed. Install with: pip install pyzbar")
            return {
                'recipient': 'Unknown',
                'amount': 0.0,
                'currency': 'USD',
                'description': 'QR scanner not available',
                'confidence_score': 0.0
            }

    def _extract_barcode_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from barcode"""
        try:
            from pyzbar.pyzbar import decode

            decoded_objects = decode(image)

            if decoded_objects:
                barcode_data = decoded_objects[0].data.decode('utf-8')

                return {
                    'barcode': barcode_data,
                    'barcode_type': decoded_objects[0].type,
                    'description': f'Barcode: {barcode_data}',
                    'confidence_score': 0.95
                }
            else:
                return {
                    'description': 'Barcode not detected',
                    'confidence_score': 0.0
                }

        except ImportError:
            return {
                'description': 'Barcode scanner not available',
                'confidence_score': 0.0
            }

    def _extract_account_details(self, text: str) -> Dict[str, Any]:
        """Extract bank account details"""
        data = {
            'account_number': self._find_account_number(text),
            'routing_number': self._find_routing_number(text),
            'bank_name': self._find_bank_name(text),
            'account_holder': self._find_account_holder(text),
            'account_type': self._find_account_type(text),
            'confidence_score': 0.80
        }
        return data

    def _extract_gift_card_data(self, text: str) -> Dict[str, Any]:
        """Extract gift card data"""
        data = {
            'card_number': self._find_card_number(text),
            'security_code': self._find_security_code(text),
            'merchant': self._find_merchant(text),
            'balance': self._find_amount(text),
            'currency': 'USD',
            'expiry_date': self._find_expiry_date(text),
            'confidence_score': 0.83
        }
        return data

    def _extract_receipt_data(self, text: str) -> Dict[str, Any]:
        """Extract receipt data"""
        data = {
            'recipient': self._find_merchant(text),
            'amount': self._find_amount(text),
            'currency': 'USD',
            'reference': self._find_transaction_id(text),
            'description': 'Receipt payment',
            'items': self._find_line_items(text),
            'confidence_score': 0.81
        }
        return data

    def _extract_generic_data(self, text: str) -> Dict[str, Any]:
        """Extract generic data when type is unknown"""
        data = {
            'amount': self._find_amount(text),
            'currency': 'USD',
            'description': 'Scanned document',
            'extracted_text': text[:500],  # First 500 chars
            'confidence_score': 0.70
        }
        return data

    # Helper methods for extracting specific fields

    def _find_amount(self, text: str) -> float:
        """Find monetary amount in text"""
        # Pattern for amounts like $123.45, 123.45, etc.
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:total|amount|due|balance)[\s:]+\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*\.\d{2})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue

        return 0.0

    def _find_company_name(self, text: str) -> str:
        """Find company name in text"""
        lines = text.strip().split('\n')
        # Usually company name is in first few lines
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and not line.startswith('#'):
                return line
        return 'Unknown Company'

    def _find_invoice_number(self, text: str) -> Optional[str]:
        """Find invoice number"""
        pattern = r'(?:invoice|inv)[\s#:]*([A-Z0-9-]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_account_number(self, text: str) -> Optional[str]:
        """Find account number"""
        pattern = r'(?:account|acct)[\s#:]*(\d{8,16})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_routing_number(self, text: str) -> Optional[str]:
        """Find routing number"""
        pattern = r'(?:routing|aba)[\s#:]*(\d{9})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_date(self, text: str) -> Optional[str]:
        """Find date in various formats"""
        patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _find_utility_company(self, text: str) -> str:
        """Find utility company name"""
        utilities = ['electric', 'power', 'water', 'gas', 'energy', 'utility']
        lines = text.lower().split('\n')

        for line in lines[:10]:
            for utility in utilities:
                if utility in line:
                    return line.strip().title()

        return self._find_company_name(text)

    def _find_bill_period(self, text: str) -> Optional[str]:
        """Find billing period"""
        pattern = r'(?:billing period|period)[\s:]+([A-Za-z]+\s+\d{4})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_bank_name(self, text: str) -> Optional[str]:
        """Find bank name"""
        banks = ['bank', 'credit union', 'federal', 'national']
        lines = text.split('\n')

        for line in lines[:10]:
            for bank_keyword in banks:
                if bank_keyword.lower() in line.lower():
                    return line.strip()

        return None

    def _find_account_holder(self, text: str) -> Optional[str]:
        """Find account holder name"""
        pattern = r'(?:account holder|name)[\s:]+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_account_type(self, text: str) -> Optional[str]:
        """Find account type (checking, savings, etc.)"""
        types = ['checking', 'savings', 'money market', 'business']
        text_lower = text.lower()

        for acc_type in types:
            if acc_type in text_lower:
                return acc_type.title()

        return None

    def _find_card_number(self, text: str) -> Optional[str]:
        """Find card number"""
        pattern = r'(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _find_security_code(self, text: str) -> Optional[str]:
        """Find security code/PIN"""
        pattern = r'(?:pin|code|cvv)[\s:]+(\d{3,4})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_merchant(self, text: str) -> str:
        """Find merchant name"""
        return self._find_company_name(text)

    def _find_expiry_date(self, text: str) -> Optional[str]:
        """Find expiry date"""
        pattern = r'(\d{2}/\d{2,4})'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _find_transaction_id(self, text: str) -> Optional[str]:
        """Find transaction ID"""
        pattern = r'(?:transaction|trans|txn)[\s#:]*([A-Z0-9-]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _find_line_items(self, text: str) -> list:
        """Find line items in receipt"""
        # This is a simplified version - would need more sophisticated parsing
        items = []
        lines = text.split('\n')

        for line in lines:
            # Look for lines with amounts
            if '$' in line or re.search(r'\d+\.\d{2}', line):
                items.append(line.strip())

        return items[:10]  # Return max 10 items

    def _generate_ai_message(self, extracted_data: Dict[str, Any], scan_type: str) -> str:
        """Generate helpful AI message based on extracted data"""
        amount = extracted_data.get('amount', 0.0)
        recipient = extracted_data.get('recipient', 'Unknown')
        confidence = extracted_data.get('confidence_score', 0.0)

        if confidence < 0.5:
            return "I had some difficulty reading this document. Please ensure the image is clear and well-lit. You may need to manually verify the details."

        messages = {
            'INVOICE': f"I've extracted the invoice details. It appears to be for ${amount:.2f} payable to {recipient}. Would you like to proceed with the payment?",
            'UTILITY_BILL': f"I've found your utility bill from {recipient} for ${amount:.2f}. The due date is {extracted_data.get('due_date', 'not specified')}. Shall I help you pay this?",
            'QR_CODE': f"QR code scanned successfully! Payment of ${amount:.2f} to {recipient}. Ready to proceed?",
            'RECEIPT': f"Receipt from {recipient} for ${amount:.2f} has been scanned. Would you like to save or share this?",
            'ACCOUNT_DETAILS': f"I've extracted the bank account details for {extracted_data.get('account_holder', 'the account')}. Please verify these details carefully before proceeding.",
            'GIFT_CARD': f"Gift card detected from {recipient} with a balance of ${amount:.2f}. Would you like to redeem this?"
        }

        return messages.get(scan_type, f"I've extracted the payment details. Amount: ${amount:.2f}, Recipient: {recipient}. How would you like to proceed?")


# Singleton instance
_ai_scan_service = None

def get_ai_scan_service() -> AiScanService:
    """Get or create AI scan service instance"""
    global _ai_scan_service
    if _ai_scan_service is None:
        _ai_scan_service = AiScanService()
    return _ai_scan_service
